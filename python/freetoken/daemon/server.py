"""``ft daemon`` entrypoint: the persistent, torch-free engine supervisor.

Wires the pieces together, applies the self-preservation policies (single-instance lock, signal
hygiene, degraded start, periodic OOM reapply), re-adopts a still-running serve, and runs uvicorn.
Imports argparse + stdlib + the daemon package only — never torch."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .version import DAEMON_VERSION

logger = logging.getLogger("freetoken.daemon")

DEFAULT_PORT = 1900  # distinct from the serve default (1919)
# kai's manager (`ft mgr`) is the same supervisor plus the web console, on its own port and state dir.
# Not 1900: Windows' SSDP Discovery service holds that port, and under WSL's mirrored networking a
# listener there is unreachable even from inside WSL.
MGR_PORT = 1901
MGR_STATE_DIR = "mgr"
DEFAULT_SERVE_PORT = 1919


def _default_state_dir(console: bool = False) -> str:
    env = os.environ.get("FREETOKEN_DAEMON_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".freetoken", MGR_STATE_DIR if console else "daemon")


def _build_parser(prog: str, console: bool = False) -> argparse.ArgumentParser:
    from .client import CLIENT_VERBS

    p = argparse.ArgumentParser(
        prog=prog,
        description="FreeToken engine supervisor (run the daemon server)",
        epilog=(
            "To CONTROL a running daemon instead of starting one, use a client verb:\n  "
            + " ".join(CLIENT_VERBS)
            + f"\ne.g. `{prog} status`, `{prog} start MODEL --port 1919`, `{prog} logs`. "
            + f"Run `{prog} <verb> --help` for options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="Control-plane bind host (default loopback)")
    port = MGR_PORT if console else DEFAULT_PORT
    p.add_argument("--port", type=int, default=port, help=f"Control port (default {port})")
    p.add_argument("--state-dir", default=_default_state_dir(console), help="Lock/pidfile/log directory")
    p.add_argument("--token", default=os.environ.get("FREETOKEN_DAEMON_TOKEN"), help="Optional X-FT-Token shared secret")
    p.add_argument("--default-serve-port", type=int, default=DEFAULT_SERVE_PORT, help="Port used when /engine/start omits one")
    p.add_argument("--serve-python", default=sys.executable, help="Interpreter used to launch ft serve")
    p.add_argument("--grace", type=float, default=10.0, help="SIGTERM→SIGKILL grace seconds on stop")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Adopted-serve liveness / OOM reapply interval")
    p.add_argument("--oom-child-score", type=int, default=500, help="oom_score_adj written to the serve tree")
    p.add_argument("--no-oom", action="store_true", help="Do not manage oom_score_adj")
    p.add_argument("--auto-restart", action="store_true", help="Restart the serve on crash (default off)")
    p.add_argument("--stop-serve-on-exit", action="store_true", help="Stop the serve when the daemon exits (default: detach, engine outlives the daemon)")
    p.add_argument("--log-capacity", type=int, default=4000, help="Log ring size (lines)")
    p.add_argument("--setsid", action="store_true", help="Detach into a new session at startup (guarded)")
    p.add_argument("--log-level", default="info", help="uvicorn log level")
    return p


def _install_signal_hygiene(setsid: bool) -> None:
    """SIGHUP ignored so a ``wsl.exe --`` relay close can't kill the daemon; SIGPIPE
    ignored so a broken client socket can't. Set BEFORE uvicorn.run — uvicorn only installs
    SIGINT/SIGTERM handlers and restores them on exit, so these SIG_IGNs survive."""
    for name in ("SIGHUP", "SIGPIPE"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):  # pragma: no cover - non-main-thread / unsupported
                pass
    if setsid and hasattr(os, "setsid") and hasattr(os, "getpgrp"):
        # setsid() raises EPERM if we are already a group leader (the systemd case) — guard it.
        # Best-effort; never gates boot.
        try:
            if os.getpgrp() != os.getpid():
                os.setsid()
        except OSError:
            pass


def _start_oom_reaper(manager, interval: float, stop: threading.Event) -> threading.Thread:
    """Periodically rewrite the positive OOM score across the whole serve tree so mp-spawn workers
    that fork after the initial write still become the preferred OOM victim."""

    def _run() -> None:
        while not stop.wait(interval):
            try:
                if manager.current_pid() is not None:
                    manager.reapply_oom()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=_run, name="ft-daemon-oom-reaper", daemon=True)
    t.start()
    return t


def main(argv: Sequence[str] | None = None, *, prog: str = "ft daemon", console: bool = False) -> int:
    """``console=True`` is ``ft mgr``: the same supervisor with kai's web console (/ui/) and the
    profile store on top. Plain ``ft daemon`` stays what upstream ships."""
    args = _build_parser(prog, console).parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [ft-daemon] %(levelname)s %(message)s",
    )

    from .checkpoint import CheckpointManager
    from .logring import LogRing
    from .metrics import FootprintCache
    from .pidfile import AlreadyRunning, ServeStateStore, SingleInstance
    from .profiles import ProfileStore
    from .proxy import ServeProbe
    from .serve_manager import ServeManager
    from .tailer import LogTailer

    state_dir = args.state_dir
    log_dir = os.path.join(state_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # The ONE hard refusal: two daemons cannot co-own one engine. Everything else degrades.
    lock = SingleInstance(os.path.join(state_dir, "daemon.pid"))
    try:
        lock.acquire()
    except AlreadyRunning as exc:
        print(f"ft daemon: {exc}", file=sys.stderr)
        return 1

    _install_signal_hygiene(args.setsid)

    ring = LogRing(capacity=args.log_capacity)
    store = ServeStateStore(os.path.join(state_dir, "serve.json"))
    probe = ServeProbe()

    def tailer_factory(child):
        log_path = getattr(child, "log_path", None)
        if not log_path:
            return None
        return LogTailer(log_path, ring, from_start=not getattr(child, "adopted", False))

    manager = ServeManager(
        ring,
        store,
        tailer_factory=tailer_factory,
        grace_s=args.grace,
        poll_interval_s=args.poll_interval,
        oom_child_score=args.oom_child_score,
        apply_oom=not args.no_oom,
        auto_restart=args.auto_restart,
        python=args.serve_python,
        log_dir=log_dir,
        prepare_stop=probe.prepare_stop,
        read_stats=probe.fresh_stats,
    )
    checkpoints = CheckpointManager(
        ring, python=args.serve_python, log_dir=log_dir, tailer_factory=tailer_factory
    )
    footprint = FootprintCache()

    # Degraded start: re-adoption must never keep the daemon from booting.
    try:
        if manager.readopt():
            logger.info("re-adopted a running serve from %s", store.path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("re-adoption skipped: %s", exc)

    stop_reaper = threading.Event()
    if not args.no_oom:
        _start_oom_reaper(manager, args.poll_interval, stop_reaper)

    lifecycle_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ft-daemon-lifecycle")
    proxy_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ft-daemon-proxy")

    def shutdown_hook() -> None:
        stop_reaper.set()
        if args.stop_serve_on_exit:
            logger.info("stopping serve on daemon exit (--stop-serve-on-exit)")
            manager.stop()
        else:
            # Default: the engine outlives the daemon. Leave it running and
            # persisted so the next daemon re-adopts it; just stop following its log.
            manager.detach()

    from .app import build_app

    app = build_app(
        manager=manager,
        ring=ring,
        probe=probe,
        footprint_fn=footprint.get,
        lifecycle_pool=lifecycle_pool,
        proxy_pool=proxy_pool,
        default_serve_port=args.default_serve_port,
        token=args.token,
        checkpoints=checkpoints,
        started_wall=time.time(),
        shutdown_hook=shutdown_hook,
        profiles=ProfileStore(os.path.join(state_dir, "profiles.json")) if console else None,
        console=console,
        console_cache_dir=os.path.join(state_dir, "console"),
        serve_python=args.serve_python,
    )

    import uvicorn

    # Explicit Server (not uvicorn.run) so POST /shutdown can flip should_exit for a graceful,
    # cross-platform daemon stop — it stops the engine first, then trips this. On Windows there is
    # no clean self-SIGTERM, so this handle is the reliable way to bring the control plane down.
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            # A desktop client holds an INFINITE SSE connection open on /engine/logs; without a bound,
            # POST /shutdown's graceful stop waits forever for it to close — the listen port frees but
            # the process (and its single-instance lock) lingers, so the next auto-spawn can't acquire
            # the lock. Force-close lingering connections a few seconds after should_exit so "Stop
            # daemon" actually exits the process.
            timeout_graceful_shutdown=3,
        )
    )
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)

    logger.info("%s %s listening on %s:%s (state-dir=%s)%s", prog, DAEMON_VERSION, args.host, args.port, state_dir,
                f" — web console: http://{args.host}:{args.port}/ui/" if console else "")
    try:
        server.run()
    finally:
        stop_reaper.set()
        lifecycle_pool.shutdown(wait=False)
        proxy_pool.shutdown(wait=False)
        lock.release()
    return 0
