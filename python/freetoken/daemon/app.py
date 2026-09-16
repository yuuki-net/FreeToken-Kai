"""The daemon's HTTP control plane. camelCase JSON throughout. Loopback by default; an optional
``X-FT-Token`` shared secret gates everything except the daemon's own ``/health`` liveness probe.

Handlers are ``async`` and push every blocking call to an executor so the event loop never
blocks. Two executors: a small **lifecycle** pool for start/stop/switch, kept separate from the
**proxy/metrics** pool, so a storm of health/metrics polls against a loading serve can never
starve an operator's stop."""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from urllib.parse import quote, urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .accounting import AccountingOutboxError, AccountingPrepareError
from .serve_manager import Conflict
from .version import DAEMON_VERSION


class StartBody(BaseModel):
    model: str
    port: int | None = None
    args: list[str] = []


class StopBody(BaseModel):
    force: bool = False


class SwitchBody(StartBody):
    force: bool = False


class AccountingAckBody(BaseModel):
    receiptId: str


class CheckpointBody(BaseModel):
    id: str
    args: list[str] = []


class CancelBody(BaseModel):
    id: str


class ProfileBody(BaseModel):
    model: str
    port: int | None = None
    args: list[str] = []


class TuneBody(BaseModel):
    model: str
    trials: bool = True
    port: int | None = None


class BenchBody(BaseModel):
    # Raw `ft bench bw` args (e.g. ["--dtype", "nvfp4", "--threshold", "2.5"]); empty = all dtypes.
    args: list[str] = []


def _bench_profile_path(gpu_uuid: str | None) -> str | None:
    # per-GPU profiles and no torch here: the serve's own card when its --gpu names one, else the newest file
    from freetoken.moe.bench_profile import default_profile_path, latest_profile_path  # torch-free

    if gpu_uuid:
        path = default_profile_path(gpu_uuid)
        if os.path.isfile(path):
            return path
    return latest_profile_path()


def _serve_gpu_uuid(args: list[str]) -> str | None:
    """The full UUID a serve's `--gpu` pins, or None when there is none or it cannot be resolved."""
    for i, a in enumerate(args):
        val = a[len("--gpu="):] if a.startswith("--gpu=") else (args[i + 1] if a == "--gpu" and i + 1 < len(args) else None)
        if not val:
            continue
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            resolved = resolve_gpu_uuids([val])
        except ValueError:
            return None
        if resolved:
            return resolved[0]
        # no NVML: a UUID value still keys the profile file (canonical prefix), an index cannot
        return "GPU-" + val[len("GPU-"):] if val.upper().startswith("GPU-") else None
    return None


def _read_bench_profile(path: str | None) -> dict | None:
    if path is None:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _bench_sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _parse_ftbench(line: str) -> dict | None:
    """``FTBENCH <done> <total> <label>`` -> a progress dict (mirrors ft checkpoint's FTCONVERT)."""
    parts = line.split(maxsplit=3)
    if len(parts) < 4 or parts[0] != "FTBENCH":
        return None
    try:
        return {"done": int(parts[1]), "total": int(parts[2]), "label": parts[3]}
    except ValueError:
        return None


class _WriteGuard:
    """Refuses writes a browser may not make, before routing.

    The web console (/ui/) makes this control plane reachable from a browser, and a CORS preflight
    does not stop a body-less cross-site POST such as /engine/stop: writes from another http(s)
    origin are refused (clients that send no Origin, ft daemon and curl, pass). Under ft mgr a
    write must also come from this PC or carry the token (webui/auth.py).

    Plain ASGI, not ``@app.middleware("http")``: that wraps every response in a task group, and the
    endless /engine/logs stream a dashboard holds open then logs a traceback when shutdown cancels it."""

    def __init__(self, app, *, console: bool, write_token: str | None) -> None:
        self.app, self.console, self.write_token = app, console, write_token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
            origin = urlsplit(headers.get("origin") or "")
            if origin.scheme in ("http", "https") and origin.hostname != "tauri.localhost" and (
                origin.netloc != headers.get("host")
            ):
                refusal = {"error": "cross-origin write refused", "code": "cross_origin"}
                return await JSONResponse(status_code=403, content=refusal)(scope, receive, send)
            if self.console:
                from freetoken.webui.auth import may_write

                client = (scope.get("client") or (None,))[0]
                if not may_write(client, headers.get("host"), headers.get("x-ft-token"), self.write_token):
                    refusal = {"error": "operating ft mgr from another PC needs its token", "code": "write_needs_token"}
                    return await JSONResponse(status_code=403, content=refusal)(scope, receive, send)
        await self.app(scope, receive, send)


def build_app(
    *,
    manager,
    ring,
    probe,
    footprint_fn: Callable[[int | None], dict],
    lifecycle_pool: ThreadPoolExecutor,
    proxy_pool: ThreadPoolExecutor,
    default_serve_port: int = 1919,
    token: str | None = None,
    checkpoints=None,
    started_wall: float = 0.0,
    wall_now: Callable[[], float] | None = None,
    shutdown_hook: Callable[[], None] | None = None,
    profiles=None,
    console: bool = False,
    console_cache_dir: str | None = None,
    serve_python: str | None = None,
    write_token: str | None = None,
) -> FastAPI:
    import time as _time

    wall_now = wall_now or _time.time
    app = FastAPI(title="FreeToken daemon", version=DAEMON_VERSION)

    if shutdown_hook is not None:

        @app.on_event("shutdown")
        async def _on_shutdown() -> None:
            # uvicorn fires this on SIGTERM/SIGINT. Run the (blocking) hook off-loop so the grace
            # period in stop() can't wedge the event loop during shutdown.
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, shutdown_hook)
            except Exception:  # noqa: BLE001
                pass

    app.add_middleware(_WriteGuard, console=console, write_token=write_token)

    def require_token(x_ft_token: str | None = Header(default=None)) -> None:
        if token is not None and x_ft_token != token:
            raise HTTPException(status_code=401, detail="invalid or missing X-FT-Token")

    auth = [Depends(require_token)]

    async def run(pool: ThreadPoolExecutor, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, functools.partial(fn, *args))

    def resolve_port(explicit: int | None) -> int:
        if explicit is not None:
            return explicit
        st = manager.status()
        return st.get("port") or default_serve_port

    def accounting_error(exc: Exception) -> JSONResponse:
        code = (
            "accounting_outbox_failed"
            if isinstance(exc, AccountingOutboxError)
            else "accounting_prepare_failed"
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "code": code,
                "enginePreserved": True,
            },
        )

    # ---- daemon self-health (never gated; always answers if the daemon is up) ----

    @app.get("/health")
    async def health():
        st = manager.status()
        return {
            "status": "ok",
            "version": DAEMON_VERSION,
            "uptimeS": int(wall_now() - started_wall) if started_wall else 0,
            "engineRunning": bool(st.get("running")),
        }

    # ---- engine lifecycle ----

    def resolve_model_arg(model: str) -> str:
        """Fail a start with a readable message instead of letting the engine die on a hub 401."""
        from freetoken.webui.models import resolve_model

        try:
            return resolve_model(model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/engine/start", dependencies=auth)
    async def engine_start(body: StartBody):
        port = resolve_port(body.port)
        body.model = resolve_model_arg(body.model)
        try:
            return await run(lifecycle_pool, manager.start, body.model, port, list(body.args))
        except Conflict as exc:
            st = manager.status()
            return JSONResponse(
                status_code=409,
                content={
                    "error": str(exc),
                    "code": "serve_conflict",
                    "currentModel": st.get("model"),
                    "currentPort": st.get("port"),
                },
            )
        except Exception as exc:  # noqa: BLE001 — never propagate a 500-as-crash
            raise HTTPException(status_code=500, detail=f"start failed: {exc}")

    @app.post("/engine/stop", dependencies=auth)
    async def engine_stop(body: StopBody | None = None):
        try:
            return await run(lifecycle_pool, manager.stop, None, bool(body and body.force))
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)

    @app.post("/shutdown", dependencies=auth)
    async def shutdown_daemon(request: Request, body: StopBody | None = None):
        # Tray "Stop daemon" stops everything: stop the engine FIRST so the default detach-on-exit can't
        # leave the ~18GB serve orphaned, THEN bring the daemon down. We reply before uvicorn
        # actually stops (it notices should_exit within ~0.1s) so the client still gets a clean 200.
        try:
            stopped = await run(lifecycle_pool, manager.shutdown, None, bool(body and body.force))
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)
        req = getattr(request.app.state, "request_shutdown", None)
        if req is not None:
            req()
        return {
            "stopping": True,
            "already": stopped.get("already", False),
            "accounting": stopped.get("accounting"),
        }

    @app.post("/engine/switch", dependencies=auth)
    async def engine_switch(body: SwitchBody):
        port = resolve_port(body.port)
        body.model = resolve_model_arg(body.model)
        try:
            return await run(
                lifecycle_pool,
                manager.switch,
                body.model,
                port,
                list(body.args),
                body.force,
            )
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"switch failed: {exc}")

    # ---- durable accounting outbox ----

    @app.get("/accounting/pending", dependencies=auth)
    async def accounting_pending():
        try:
            receipts = await run(lifecycle_pool, manager.pending_accounting)
        except AccountingOutboxError as exc:
            return accounting_error(exc)
        return {"receipts": receipts}

    @app.post("/accounting/ack", dependencies=auth)
    async def accounting_ack(body: AccountingAckBody):
        try:
            return await run(lifecycle_pool, manager.ack_accounting, body.receiptId)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except AccountingOutboxError as exc:
            return accounting_error(exc)

    @app.get("/engine/status", dependencies=auth)
    async def engine_status():
        return manager.status()

    @app.get("/engine/metrics", dependencies=auth)
    async def engine_metrics():
        pid = manager.current_pid()
        return await run(proxy_pool, footprint_fn, pid)

    @app.get("/engine/health", dependencies=auth)
    async def engine_health():
        st = manager.status()
        if not st.get("running"):
            return {"reachable": False, "running": False, "daemon": "up", **_engine_summary(st)}
        port = st.get("port") or default_serve_port
        doc = await run(proxy_pool, probe.health, port)
        # The serve's own health fields (status/model/uptimeS/progress) are authoritative for
        # "how is the model doing?"; the daemon only layers on what only it knows, never clobbering
        # the serve's values.
        doc["running"] = True
        doc["daemon"] = "up"
        doc.setdefault("port", st.get("port"))
        doc.setdefault("pid", st.get("pid"))
        doc.setdefault("lastExitCode", st.get("lastExitCode"))
        return doc

    @app.get("/engine/stats", dependencies=auth)
    async def engine_stats():
        st = manager.status()
        if not st.get("running"):
            return {"reachable": False, "running": False}
        port = st.get("port") or default_serve_port
        doc = await run(proxy_pool, probe.stats, port)
        manager.observe_accounting(doc)
        return doc

    @app.get("/engine/logs", dependencies=auth)
    async def engine_logs(request: Request, since: int = 0):
        return _log_stream(request, ring, since)

    # ---- checkpoint (phase 3; optional) ----

    if checkpoints is not None:

        @app.post("/checkpoint/start", dependencies=auth)
        async def checkpoint_start(body: CheckpointBody):
            # GPU exclusivity: a convert needs the GPU, so stop any serve first.
            await run(lifecycle_pool, manager.stop)
            try:
                return await run(lifecycle_pool, checkpoints.start, body.id, list(body.args))
            except Conflict as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"checkpoint start failed: {exc}")

        @app.post("/checkpoint/cancel", dependencies=auth)
        async def checkpoint_cancel(body: CancelBody):
            return await run(lifecycle_pool, checkpoints.cancel, body.id)

        @app.get("/checkpoint/status", dependencies=auth)
        async def checkpoint_status():
            return checkpoints.status()

    # ---- hardware bandwidth bench (hardware-adaptive config) ----

    @app.post("/bench/run", dependencies=auth)
    async def bench_run(body: BenchBody):
        # GPU exclusivity: the bench allocates transient device memory, so stop any serve first
        # (mirrors /checkpoint/start). Runs `ft bench bw` on the engine HOST (so the profile lands
        # where this daemon's serve reads it) and STREAMS progress back as SSE: `progress` events
        # per measured format, then a terminal `result` (the profile) or `error` event. `body.args`
        # is the raw arg list, so any `ft bench bw` flag (--dtype/--model/--threshold/...) passes
        # through. torch stays out of the daemon (child process), which also frees VRAM on exit.
        await run(lifecycle_pool, manager.stop)

        async def gen():
            env = {**os.environ, "FREETOKEN_BENCH_PROGRESS": "1"}
            argv = [sys.executable, "-m", "freetoken.cli", "bench", "bw", *body.args]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
                )
            except Exception as exc:  # noqa: BLE001
                yield _bench_sse("error", {"message": f"failed to spawn bench: {exc}"})
                return
            tail: collections.deque = collections.deque(maxlen=8)  # last non-progress lines (errors)
            out_path: str | None = None
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                prog = _parse_ftbench(line)
                if prog is not None:
                    yield _bench_sse("progress", prog)
                elif line.startswith("FTBENCH_OUT "):
                    out_path = line[len("FTBENCH_OUT "):]
                elif line:
                    tail.append(line)
            rc = await proc.wait()
            if rc != 0:
                yield _bench_sse("error", {"message": "\n".join(tail) or f"bench exited {rc}"})
                return
            # the file this run wrote (an older engine prints no FTBENCH_OUT: newest file, as before)
            prof = _read_bench_profile(out_path or _bench_profile_path(None))
            if prof is None:
                yield _bench_sse("error", {"message": "bench finished but no profile was written"})
            else:
                yield _bench_sse("result", prof)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/bench/profile", dependencies=auth)
    async def bench_profile():
        def read() -> dict | None:
            return _read_bench_profile(_bench_profile_path(serve_gpu_uuid()))

        def serve_gpu_uuid() -> str | None:
            # the running serve reports the full UUID of its card (/v1/stats gpus); a --gpu given as
            # a UUID prefix would not match the profile file name
            st = manager.status()
            if st.get("running"):
                try:
                    gpus = probe.stats(st.get("port") or default_serve_port).get("gpus") or []
                    if gpus and gpus[0].get("uuid"):
                        return gpus[0]["uuid"]
                except Exception:  # noqa: BLE001 -- the arg below is the fallback
                    pass
            return _serve_gpu_uuid(manager.serve_args())

        return await run(proxy_pool, read)

    def _register_console() -> None:
        console_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ft-mgr-console")
        # ---- kai's manager (ft mgr): everything the web console needs. Plain ft daemon has none of it.
        # A serve started outside the daemon (a launch script, a terminal) still answers on the default
        # serve port. The daemon cannot stop or restart it, but the console can show it: these console-only
        # routes fall back to that port whenever the daemon itself runs nothing. The upstream
        # /engine/health and /engine/stats keep reporting only the daemon's own engine.

        @app.get("/auth", dependencies=auth)
        async def console_auth(request: Request):
            # what this browser may do; the token itself only goes to a browser on this PC
            from freetoken.webui.auth import is_local, token_matches

            client = request.client.host if request.client else None
            local = is_local(client, request.headers.get("host"))
            given = request.headers.get("x-ft-token")
            valid = token_matches(given, write_token)
            return {
                "local": local,
                "write": local or valid,
                "token_given": bool(given),
                "token_valid": valid,
                "token": write_token if local else None,
            }

        @app.get("/engine/config", dependencies=auth)
        async def engine_config():
            st = manager.status()
            return {"model": st.get("model"), "port": st.get("port"), "args": manager.serve_args()}

        def _view_port() -> int:
            st = manager.status()
            return (st.get("port") or default_serve_port) if st.get("running") else default_serve_port

        async def _proxied(path: str):
            return await run(proxy_pool, probe.get, path, _view_port())

        @app.get("/engine/external", dependencies=auth)
        async def engine_external():
            if manager.status().get("running"):
                return {"external": False}
            doc = await run(proxy_pool, probe.health, default_serve_port)
            found = bool(doc.get("reachable")) and doc.get("status") in ("ok", "loading", "error")
            return {"external": found, "port": default_serve_port, "health": doc if found else None}

        @app.get("/engine/view/stats", dependencies=auth)
        async def engine_view_stats():
            if manager.status().get("running"):
                return await engine_stats()
            return await _proxied("/v1/stats")

        @app.get("/engine/requests", dependencies=auth)
        async def engine_requests(since: int = 0, limit: int = 100):
            return await _proxied(f"/v1/requests?since={int(since)}&limit={int(limit)}")

        @app.get("/engine/cache", dependencies=auth)
        async def engine_cache():
            return await _proxied("/v1/cache/status")

        @app.get("/engine/kai/experts", dependencies=auth)
        async def engine_kai_experts(window: str = "300", freq: bool = False):
            return await _proxied(f"/v1/kai/experts?window={quote(window)}&freq={str(freq).lower()}")

        @app.get("/host", dependencies=auth)
        async def host():
            from freetoken.webui.hostmem import host_memory

            st = manager.status()
            return {"memory": await run(console_pool, host_memory, st.get("pid") if st.get("running") else None)}

        @app.get("/models", dependencies=auth)
        async def models_list():
            from freetoken.webui.models import list_models

            return {"models": await run(console_pool, list_models)}

        @app.get("/recommend", dependencies=auth)
        async def recommend(model: str):
            # reads nvidia-smi, /proc and the checkpoint's config; no GPU work, so the proxy pool is fine
            from freetoken.webui.recommend import recommend as build

            try:
                return await run(console_pool, build, model)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"could not build a recommendation: {exc}")

        @app.get("/serve-flags", dependencies=auth)
        async def serve_flags():
            # built in a child process (the parser imports torch); cached per args.py version
            from freetoken.webui.serve_flags import load

            cache = console_cache_dir or os.path.join(os.path.expanduser("~"), ".freetoken", "daemon", "console")
            try:
                return {"flags": await run(lifecycle_pool, load, cache, serve_python or sys.executable)}
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"could not read ft serve flags: {exc}")

        # ---- the benchmark: measure this PC with a model, settle the flags on the numbers (webui/tuner.py)
        from freetoken.webui.tuner import Busy, TuneJob

        def _recommend_quiet(model: str) -> dict:
            from freetoken.webui.recommend import recommend as build

            return build(model)

        tune = TuneJob(
            manager,
            state_dir=os.path.dirname(console_cache_dir) if console_cache_dir else os.path.join(os.path.expanduser("~"), ".freetoken", "mgr"),
            python=serve_python or sys.executable, default_port=default_serve_port, recommend=_recommend_quiet,
        )
        app.state.tune = tune

        @app.post("/tune/start", dependencies=auth)
        async def tune_start(body: TuneBody):
            from freetoken.webui.models import resolve_model

            try:
                model = resolve_model(body.model)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if not manager.status().get("running"):
                # a serve this manager did not start holds the GPU and cannot be stopped from here
                doc = await run(proxy_pool, probe.health, default_serve_port)
                if doc.get("reachable") and doc.get("status") in ("ok", "loading"):
                    return JSONResponse(status_code=409, content={"error": "a server started outside ft mgr is running", "code": "external_serve"})
            try:
                return tune.start(model, body.trials, body.port)
            except Busy as exc:
                return JSONResponse(status_code=409, content={"error": str(exc), "code": "busy"})

        @app.post("/tune/cancel", dependencies=auth)
        async def tune_cancel():
            return tune.cancel()

        @app.get("/tune/status", dependencies=auth)
        async def tune_status(since: int = 0):
            return tune.status(since)

        @app.get("/tune/last", dependencies=auth)
        async def tune_last(model: str):
            from freetoken.webui.models import resolve_model

            try:
                return {"result": tune.last(resolve_model(model))}
            except ValueError:
                return {"result": None}

        if profiles is not None:

            @app.get("/profiles", dependencies=auth)
            async def profiles_list():
                return {"profiles": await run(lifecycle_pool, profiles.list)}

            @app.put("/profiles/{name}", dependencies=auth)
            async def profiles_put(name: str, body: ProfileBody):
                try:
                    return await run(lifecycle_pool, profiles.put, name, body.model, body.port, list(body.args))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))

            @app.delete("/profiles/{name}", dependencies=auth)
            async def profiles_delete(name: str):
                if not await run(lifecycle_pool, profiles.delete, name):
                    raise HTTPException(status_code=404, detail="no such profile")
                return {"deleted": name}


    if console:
        from freetoken.webui import register_webui

        _register_console()
        register_webui(app, "mgr", DAEMON_VERSION)
    return app


def _engine_summary(st: dict) -> dict:
    return {
        "model": st.get("model"),
        "port": st.get("port"),
        "pid": st.get("pid"),
        "uptimeS": st.get("uptimeS", 0),
        "lastExitCode": st.get("lastExitCode"),
    }


def _sse(rec: dict) -> str:
    return f"id: {rec['seq']}\ndata: {json.dumps(rec)}\n\n"


def _sse_gap(dropped: int, from_seq: Any, to_seq: Any) -> str:
    payload = {"kind": "gap", "dropped": dropped, "fromSeq": from_seq, "toSeq": to_seq}
    return f"data: {json.dumps(payload)}\n\n"


def _shutting_down(request: Request) -> bool:
    check = getattr(request.app.state, "shutting_down", None)
    return bool(check and check())


def _log_stream(request: Request, ring, since: int) -> StreamingResponse:
    """SSE log stream with replay + live tail. Correctness points:
      * subscribe BEFORE snapshotting the backlog, then dedupe live records by seq → no gap and
        no duplicate across the replay→live boundary;
      * per-subscriber bounded queue, drop-oldest on overflow via ``call_soon_threadsafe`` (the
        mutation runs on the loop thread, so the reader never blocks) and a client-visible gap
        sentinel so a slow client knows it lost lines;
      * ``id:<seq>`` on every frame + ``Last-Event-ID`` honoured for native EventSource resume;
      * a 15 s heartbeat + ``is_disconnected`` check so an idle client's disconnect is detected
        and the subscriber is always removed in ``finally`` (no leak)."""
    loop = asyncio.get_running_loop()
    lei = request.headers.get("last-event-id")
    if lei and lei.isdigit():
        since = int(lei) + 1  # exclusive next-cursor

    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    drop = {"n": 0, "from": None, "to": None}
    # Records with seq < boundary are already covered by the replayed backlog (they landed in the
    # window between subscribe and the snapshot). Skipping them here keeps the gap counters honest
    # — only genuinely-lost LIVE lines feed drop[]. Safe to set after subscribe: the
    # scheduled _put callbacks only run once this handler yields control, by which point boundary
    # is set.
    boundary = {"v": 0}

    def push(rec: dict) -> None:
        def _put() -> None:
            if rec["seq"] < boundary["v"]:
                return  # already delivered via backlog; don't enqueue or count it as dropped
            if q.full():
                try:
                    old = q.get_nowait()
                    drop["n"] += 1
                    if drop["from"] is None:
                        drop["from"] = old["seq"]
                    drop["to"] = old["seq"]
                except asyncio.QueueEmpty:  # pragma: no cover - race-only
                    pass
            q.put_nowait(rec)

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:  # loop is closing during shutdown
            pass

    ring.subscribe(push)
    backlog, cursor = ring.since(since)
    boundary["v"] = cursor

    async def gen():
        try:
            # If the ring evicted records at/after the client's cursor before it (re)connected,
            # announce that lost prefix so the client knows its history is incomplete.
            oldest = backlog[0]["seq"] if backlog else cursor
            if oldest > since:
                yield _sse_gap(oldest - since, since, oldest - 1)
            for rec in backlog:
                yield _sse(rec)
            last_seq = cursor - 1
            idle = 0.0
            while True:
                # a stream still open when the daemon stops is cut after the graceful timeout, and
                # uvicorn logs that cut as a traceback: end it as soon as shutdown begins
                if await request.is_disconnected() or _shutting_down(request):
                    break
                try:
                    rec = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    idle += 0.5
                    if idle >= 15.0:
                        idle = 0.0
                        yield ": ping\n\n"
                    continue
                idle = 0.0
                if rec["seq"] <= last_seq:
                    continue  # already delivered in backlog
                if drop["n"]:
                    # Snapshot + reset synchronously BEFORE yielding: during the yield the loop
                    # drains more _put callbacks that may mutate drop[], and those must not be
                    # wiped unreported.
                    n, frm, to = drop["n"], drop["from"], drop["to"]
                    drop["n"], drop["from"], drop["to"] = 0, None, None
                    yield _sse_gap(n, frm, to)
                last_seq = rec["seq"]
                yield _sse(rec)
        finally:
            ring.unsubscribe(push)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
