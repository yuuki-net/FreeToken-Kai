"""The console's benchmark: measure this PC with the model, then settle the flags on measurements.

Two stages, one background job in ``ft mgr`` (torch-free; the GPU work runs in child processes):

1. Hardware (``webui/hwbench.py``, a child process): PCIe, RAM and SSD rates, and the model's
   own expert kernels on the CPU (swept over thread counts) and over PCIe. Sets the MoE strategy
   and the CPU threads the first run starts from.
2. Real runs (optional): ``ft serve`` is started with the recommended flags (``webui/recommend.py``)
   plus the hardware's, and measured. Then every candidate in ``webui/search.py`` that applies is
   tried as one change against the best settings so far, and kept only when the measurement says
   so; last, longer contexts are tried the same way. The candidates are recomputed from the best
   settings after each run, so a kept change can open or close later ones.

What a run measures, on this repository's own documents and code (a nonce first line keeps the
prefix cache out of it): a 16k-token prompt twice, generation on prose three times (median), and
generation on code three times when an MTP head is in play (its acceptance depends on the text).
Which generation figure decides follows the use the person picked: prose, code, or both.

The engine the manager was running is stopped first and started again at the end, whatever
happened. Progress is a list of events the page polls."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.request
from typing import Any, Callable

from . import search

GiB = 1 << 30
# the longest a hardware step has gone without a line (the CPU sweep on many cores) is well under this
HW_QUIET_S = 300.0
CONTEXT_TIERS = (32768, 65536, 131072, 262144)
# long enough for several chunks: two 8k chunks hide what the two-rank overlap and a wider chunk buy
PREFILL_TOKENS = 16384
# three runs, median: on a 2060 two 200-token runs swung 31-36 tok/s between runs that should match
DECODE_TOKENS = 300
DECODE_RUNS = 3
USES = ("both", "prose", "code")
MODES = ("standard", "thorough")


class Busy(Exception):
    pass


# ------------------------------------------------------------------ flags as data
def parse_flags(args: list[str]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a.split("=", 1)
                out.append((k, v))
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                out.append((a, args[i + 1]))
                i += 1
            else:
                out.append((a, None))
        i += 1
    return out


def to_args(pairs: list[tuple[str, str | None]]) -> list[str]:
    argv: list[str] = []
    for k, v in pairs:
        argv.append(k)
        if v is not None:
            argv.append(v)
    return argv


def set_flags(args: list[str], changes: dict[str, str | None], drop: tuple[str, ...] = ()) -> list[str]:
    """``changes`` overrides or adds flags (None = a switch), ``drop`` removes them; order is kept."""
    pairs = [(k, v) for k, v in parse_flags(args) if k not in drop]
    seen = set()
    for i, (k, _) in enumerate(pairs):
        if k in changes:
            pairs[i] = (k, changes[k])
            seen.add(k)
    pairs += [(k, v) for k, v in changes.items() if k not in seen]
    return to_args(pairs)


def has_flag(args: list[str], flag: str) -> bool:
    return any(k == flag for k, _ in parse_flags(args))


# ------------------------------------------------------------------ deciding from measurements
def generation(t: dict, use: str) -> float:
    """The generation figure that decides, for the use the person picked."""
    prose = t.get("decode_tps") or 0.0
    code = t.get("decode_code_tps") or prose
    if use == "prose":
        return prose
    if use == "code":
        return code
    return math.sqrt(prose * code) if prose and code else prose or code


def keep(best: dict, t: dict, c: search.Candidate, use: str) -> bool:
    if not t.get("ok"):
        return False
    g0, g1 = generation(best, use), generation(t, use)
    p0, p1 = best.get("prefill_tps") or 0.0, t.get("prefill_tps") or 0.0
    if c.goal == "prefill":
        return p1 >= c.gain * p0 and g1 >= c.hold_gen * g0
    if c.goal == "either":
        return (g1 >= c.gain * g0 and p1 >= 0.95 * p0) or (p1 >= 1.05 * p0 and g1 >= c.hold_gen * g0)
    return g1 >= c.gain * g0 and p1 >= c.hold_prefill * p0


def context_candidates(geometry: dict, model_max: int | None) -> list[int]:
    """Context tiers above the one running, up to the model's limit and what the cache budget can hold."""
    now = (geometry.get("num_pages") or 0) * (geometry.get("page_size") or 1)
    cap = min(model_max or 262144, ((geometry.get("limits") or {}).get("kv_tokens") or {}).get("max") or 1 << 30)
    return [t for t in CONTEXT_TIERS if now and t >= now * 1.5 and t <= cap]


# kept for callers and tests of the earlier, single-step version
def longer_context(geometry: dict, ranks: list[dict], cache_auto: bool, model_max: int | None,
                   max_expert_share: float = 0.25, headroom: int = 600 << 20) -> dict | None:
    ub = geometry.get("unit_bytes") or {}
    kv_per_token, per_expert = ub.get("kv_per_token") or 0, ub.get("moe_per_expert") or 0
    now = (geometry.get("num_pages") or 0) * (geometry.get("page_size") or 1)
    slots = geometry.get("moe_cache_size") or 0
    if not (now and kv_per_token):
        return None
    frees = [(r.get("gpu") or {}).get("free_bytes") for r in ranks]
    frees = [f for f in frees if f is not None]
    free = max(0, min(frees) - headroom) if frees else 0
    cap = min(model_max or 262144, ((geometry.get("limits") or {}).get("kv_tokens") or {}).get("max") or 1 << 30)
    best = None
    for ctx in CONTEXT_TIERS:
        if ctx < now * 1.5 or ctx > cap:
            continue
        need = (ctx - now) * kv_per_token
        if need <= free:
            lost = 0
        elif cache_auto and slots and per_expert:
            lost = math.ceil((need - free) / per_expert)
            if lost > slots * max_expert_share:
                continue
        else:
            continue
        best = {"tokens": ctx, "now": now, "need_bytes": need, "free_bytes": free, "experts_lost": lost, "experts": slots}
    return best


# ------------------------------------------------------------------ what the runs read
_FALLBACK_WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue"
).split()


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_corpus(root: str | None = None) -> dict:
    """This repository's own prose (README, docs) and code (the package): text a model is actually
    asked to read, for routing and MTP acceptance that look like use. Random words when neither is
    there (a wheel install)."""
    root = root or _repo_root()
    prose, code = [], []
    for name in ["README.md"] + sorted(f"docs/{n}" for n in (os.listdir(os.path.join(root, "docs")) if os.path.isdir(os.path.join(root, "docs")) else [])):
        path = os.path.join(root, name)
        if name.endswith(".md") and os.path.isfile(path):
            try:
                prose.append(open(path, encoding="utf-8").read())
            except OSError:
                pass
    pkg = os.path.join(root, "python", "freetoken")
    for base, dirs, files in sorted(os.walk(pkg)):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".py") and "test" not in f:
                try:
                    code.append(open(os.path.join(base, f), encoding="utf-8").read())
                except OSError:
                    pass
            if sum(map(len, code)) > 400_000:
                break
    words = " ".join(_FALLBACK_WORDS * 200)
    return {"prose": "\n\n".join(prose) or words, "code": "\n\n".join(code) or words}


def excerpt(text: str, chars: int, rng: random.Random) -> str:
    if len(text) <= chars:
        return (text * (chars // max(1, len(text)) + 1))[:chars]
    start = rng.randrange(0, len(text) - chars)
    return text[start:start + chars]


# ------------------------------------------------------------------ talking to the serve
def _http(port: int, path: str, body: dict | None = None, timeout: float = 600.0) -> Any:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "null")


def _stream(port: int, path: str, body: dict):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            yield raw.decode(errors="replace")


def _prompt_tokens(doc: Any) -> tuple[int, int]:
    usage = (doc or {}).get("usage") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return int(usage.get("prompt_tokens") or 0), int(cached)


def _labels():
    for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        yield a
    for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            yield a + b


class TuneJob:
    def __init__(self, manager, *, state_dir: str, python: str, default_port: int,
                 recommend: Callable[[str], dict] | None = None, spawn: Callable[..., Any] | None = None,
                 http: Callable[..., Any] | None = None, stream: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 corpus: dict | None = None):
        self.manager = manager
        self.dir = os.path.join(state_dir, "tune")
        self.python = python
        self.default_port = default_port
        self._recommend = recommend
        self._spawn = spawn or (lambda argv: subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
        self._http = http or _http
        self._stream = stream or _stream
        self._clock = clock
        self._sleep = sleep
        self._corpus = corpus
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._proc = None
        self._thread: threading.Thread | None = None
        self.run_id = 0
        self.state, self.events, self.result, self.error, self.model = "idle", [], None, None, None
        self._seq = 0

    # ---- events
    def _ev(self, event: str, /, **fields) -> None:
        with self._lock:
            self._seq += 1
            self.events.append({"seq": self._seq, "k": event, "t": round(time.time(), 2), **fields})
            if len(self.events) > 8000:
                del self.events[:2000]

    def status(self, since: int = 0) -> dict:
        with self._lock:
            return {"run": self.run_id, "state": self.state, "model": self.model, "error": self.error, "result": self.result,
                    "events": [e for e in self.events if e["seq"] > since], "seq": self._seq}

    # ---- control
    def start(self, model: str, trials: bool = True, port: int | None = None, mode: str = "standard", use: str = "both") -> dict:
        with self._lock:
            if self.state == "running":
                raise Busy("a benchmark is already running")
            # a new run: the page sees another run id and reads its events from the start
            self.run_id += 1
            self.state, self.model, self.events, self.result, self.error, self._seq = "running", model, [], None, None, 0
        self._cancel.clear()
        mode = mode if mode in MODES else "standard"
        use = use if use in USES else "both"
        self._thread = threading.Thread(target=self._run, args=(model, trials, port or self.default_port, mode, use),
                                        name="ft-mgr-benchmark", daemon=True)
        self._thread.start()
        return {"started": True}

    def cancel(self) -> dict:
        self._cancel.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        return {"cancelling": self.state == "running"}

    def last(self, model: str) -> dict | None:
        try:
            with open(self._path(model)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _path(self, model: str) -> str:
        return os.path.join(self.dir, os.path.basename(model.rstrip("/")) + ".json")

    def _check(self) -> None:
        if self._cancel.is_set():
            raise InterruptedError("cancelled")

    # ---- the job
    def _run(self, model: str, trials: bool, port: int, mode: str = "standard", use: str = "both") -> None:
        prev = self.manager.status()
        prev_cfg = (prev.get("model"), prev.get("port"), list(self.manager.serve_args())) if prev.get("running") else None
        result: dict = {"model": model, "port": port, "started": time.time(), "trials": [], "mode": mode, "use": use,
                        "skipped": [{"flag": f, "why": ja, "why_en": en} for f, ja, en in search.SKIPPED]}
        try:
            if prev_cfg:
                self._ev("phase", phase="stop")
                self.manager.stop()
            self._check()
            self._ev("phase", phase="hw")
            hw = self._hardware(model)
            result["hw"] = hw
            self._check()
            rec = self._recommend(model) if self._recommend else {"flags": [], "notes": []}
            args, notes = self._apply_hw(rec, hw)
            result["base_args"] = list(args)
            if trials:
                host = rec.get("host") or {}
                facts = dict(host.get("model") or {}, ram_total=(host.get("memory") or {}).get("total"),
                             weight_bytes=host.get("weight_bytes"))
                args, notes = self._trials(model, port, args, notes, result, facts, hw, mode, use)
            result["args"] = [a for a in args if a != "--moe-collect-stats"]
            result["notes"] = notes
            result["finished"] = time.time()
            os.makedirs(self.dir, exist_ok=True)
            with open(self._path(model), "w") as fh:
                json.dump(result, fh)
            with self._lock:
                self.result, self.state = result, "done"
            self._ev("phase", phase="done")
        except InterruptedError:
            with self._lock:
                self.state = "cancelled"
            self._ev("phase", phase="cancelled")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state, self.error = "error", str(exc)[:1000]
            self._ev("phase", phase="error", message=str(exc)[:1000])
        finally:
            self._proc = None
            self._restore(prev_cfg, trials)

    def _restore(self, prev_cfg, trials: bool) -> None:
        try:
            if prev_cfg:
                self._ev("phase", phase="restore", model=prev_cfg[0])
                self.manager.switch(prev_cfg[0], prev_cfg[1], prev_cfg[2])
            elif trials and self.manager.status().get("running"):
                self.manager.stop()
        except Exception as exc:  # noqa: BLE001
            self._ev("log", msg=f"could not restore the previous server: {exc}")

    def _hardware(self, model: str, quiet_s: float = HW_QUIET_S) -> dict:
        import queue

        proc = self._spawn([self.python, "-m", "freetoken.webui.hwbench", "--model", model])
        self._proc = proc
        out: dict = {}
        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            for raw in proc.stdout:
                lines.put(raw)
            lines.put(None)

        threading.Thread(target=pump, name="ft-mgr-benchmark-hw", daemon=True).start()
        step = None
        while True:
            try:
                line = lines.get(timeout=quiet_s)
            except queue.Empty:
                # a GPU kernel that hangs never returns: stop the child rather than wait forever
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(f"hardware step {step or '?'} sent nothing for {int(quiet_s)} s and was stopped")
            if line is None:
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.pop("type", None)
            if kind == "step":
                step = msg.get("id")
            if kind == "result":
                out = msg
            elif kind == "error":
                raise RuntimeError(msg.get("message") or "hardware measurement failed")
            else:
                self._ev("hw", kind=kind, **msg)
        proc.wait()
        self._check()
        if not out:
            raise RuntimeError(f"hardware measurement ended without a result (exit {proc.returncode})")
        return out

    def _apply_hw(self, rec: dict, hw: dict) -> tuple[list[str], list[dict]]:
        args = list(rec.get("flags") or [])
        notes = {n["flag"]: dict(n, source="rule") for n in rec.get("notes") or []}
        changes: dict[str, str | None] = {}
        drop: tuple[str, ...] = ()
        for n in hw.get("notes") or []:
            if n["flag"] == "--moe-hybrid-max-fetch":
                notes[n["flag"]] = dict(n, source="measured", info=True)  # the default already reads the record
                continue
            changes[n["flag"]] = n["value"]
            notes[n["flag"]] = dict(n, source="measured")
        if changes.get("--moe-strategy") == "offload":
            drop = ("--moe-cpu-layers", "--moe-cpu-threads")
            for f in drop:
                notes.pop(f, None)
        args = set_flags(args, changes, drop)
        return args, list(notes.values())

    # ---- real runs
    def _trials(self, model: str, port: int, args: list[str], notes: list[dict], result: dict,
                facts: dict, hw: dict, mode: str, use: str):
        self._ev("phase", phase="trial")
        labels = _labels()
        mtp_model = bool(facts.get("mtp"))
        base = set_flags(args, {"--moe-collect-stats": None})
        a = self._trial(next(labels), model, port, base, code=mtp_model and use != "prose")
        if not a["ok"] and has_flag(base, "--dtype"):
            # float16 on a pre-Ampere card copies while converting: a tight model may not load with it
            result["trials"].append(a)
            self._ev("log", msg="dtype_retry")
            base = set_flags(base, {}, drop=("--dtype",))
            notes = [n for n in notes if n["flag"] != "--dtype"]
            notes.append({"flag": "--dtype", "value": None, "source": "measured", "removed": True,
                          "why": "float16 では読み込み時の変換で VRAM が足りず起動できなかったので、指定を外しました。",
                          "why_en": "The model did not load with float16 (the conversion needs VRAM it did not have), so the flag was dropped."})
            a = self._trial(next(labels), model, port, base, change={"flag": "--dtype", "value": None, "removed": True},
                            code=mtp_model and use != "prose")
        a["decision"] = "base"
        result["trials"].append(a)
        if not a["ok"]:
            raise RuntimeError(f"the server did not start with the measured flags: {a.get('error')}")
        best, best_args = a, base
        self._ev("plan", items=[{"key": c.key, "what": c.what_ja, "what_en": c.what_en, "change": c.change()}
                                for c in search.plan(best_args, facts, hw, mode)], mode=mode, use=use)

        tried: set[str] = set()
        kept: set[str] = set()
        while True:
            self._check()
            pending = [c for c in search.plan(best_args, facts, hw, mode)
                       if c.key not in tried and c.group not in tried and (c.after is None or c.after in kept)]
            if not pending:
                break
            c = pending[0]
            tried.update({c.key, c.group})
            cand_args = set_flags(best_args, search.merge_changes(best_args, c.changes), c.drop)
            if cand_args == best_args:
                continue
            t = self._trial(next(labels), model, port, cand_args, change=c.change(), key=c.key,
                            what=(c.what_ja, c.what_en), code=mtp_model and use != "prose" and (c.code or has_flag(cand_args, "--spec-mtp")))
            won = keep(best, t, c, use)
            t["decision"] = "kept" if won else ("failed" if not t.get("ok") else "rejected")
            result["trials"].append(t)
            self._ev("decision", trial=t["label"], key=c.key, decision=t["decision"])
            notes = self._note(notes, c, best, t, won, use)
            if won:
                best, best_args = t, cand_args
                kept.add(c.key)

        # longer contexts last, on everything kept: a tier is kept while generation holds 95%
        model_max = facts.get("max_context")
        for tier in context_candidates(best.get("geometry") or {}, model_max):
            self._check()
            c = search.Candidate(f"context_{tier}", {"--kv-reserve-tokens": str(tier), "--max-seq-len-override": str(tier)},
                                 gain=0.95, hold_prefill=0.9,
                                 what_ja=f"コンテキスト長を {tier:,} にする", what_en=f"a context of {tier:,} tokens")
            cand_args = set_flags(best_args, c.changes)
            t = self._trial(next(labels), model, port, cand_args, change=c.change(), key=c.key, what=(c.what_ja, c.what_en),
                            code=mtp_model and use != "prose" and has_flag(cand_args, "--spec-mtp"))
            won = keep(best, t, c, use)
            t["decision"] = "kept" if won else ("failed" if not t.get("ok") else "rejected")
            result["trials"].append(t)
            self._ev("decision", trial=t["label"], key=c.key, decision=t["decision"])
            notes = self._note(notes, c, best, t, won, use)
            if not won:
                break
            best, best_args = t, cand_args

        tokens = next((v for k, v in parse_flags(best_args) if k == "--kv-reserve-tokens"), None)
        if tokens:
            notes = [n for n in notes if n["flag"] != "--max-seq-len-override"]
            notes.append({"flag": "--max-seq-len-override", "value": tokens, "source": "rule",
                          "why": "宣伝する長さと実際に入る長さをそろえます。", "why_en": "The advertised context matches what actually fits."})
        result["chosen"] = best["label"]
        return best_args, notes

    def _note(self, notes: list[dict], c: search.Candidate, best: dict, t: dict, won: bool, use: str) -> list[dict]:
        def figs(x: dict) -> str:
            g = f"{x.get('decode_tps') or 0:.1f}"
            if x.get("decode_code_tps"):
                g += f"（コード {x['decode_code_tps']:.1f}）"
            return f"{x.get('prefill_tps') or 0:.0f} / {g}"

        def figs_en(x: dict) -> str:
            g = f"{x.get('decode_tps') or 0:.1f}"
            if x.get("decode_code_tps"):
                g += f" (code {x['decode_code_tps']:.1f})"
            return f"{x.get('prefill_tps') or 0:.0f} / {g}"

        ch = c.change()
        if won:
            touched = set(c.changes) | set(c.drop)
            notes = [n for n in notes if n["flag"] not in touched]
            why = f"実測で採用: {c.what_ja}。プロンプト処理 / 生成が {figs(best)} → {figs(t)} tok/s。"
            why_en = f"Measured and kept: {c.what_en}. Prompt processing / generation went {figs_en(best)} -> {figs_en(t)} tok/s."
            for flag, value in c.changes.items():
                notes.append({"flag": flag, "value": value, "source": "measured", "why": why, "why_en": why_en})
            for flag in c.drop:
                notes.append({"flag": flag, "value": None, "source": "measured", "removed": True, "why": why, "why_en": why_en})
            return notes
        if t.get("ok"):
            why = f"試して不採用: {c.what_ja}。プロンプト処理 / 生成が {figs(best)} → {figs(t)} tok/s で、採用の条件に届きませんでした。"
            why_en = f"Tried, not kept: {c.what_en}. Prompt processing / generation went {figs_en(best)} -> {figs_en(t)} tok/s, short of the bar."
        else:
            why = f"試して不採用: {c.what_ja}。起動または測定に失敗しました（{(t.get('error') or '')[:120]}）。"
            why_en = f"Tried, not kept: {c.what_en}. It failed to start or to finish the measurement ({(t.get('error') or '')[:120]})."
        notes.append({"flag": ch["flag"], "value": ch.get("value"), "source": "measured", "rejected": True, "why": why, "why_en": why_en,
                      "key": c.key})
        return notes

    def _trial(self, label: str, model: str, port: int, args: list[str], change: dict | None = None, key: str | None = None,
               what: tuple[str, str] | None = None, code: bool = False) -> dict:
        try:
            return self._trial_run(label, model, port, args, change, key, what, code)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 -- e.g. a wider chunk that runs out of VRAM mid-prompt
            self._ev("trial", trial=label, step="measure", state="failed", message=str(exc)[:300])
            return {"label": label, "key": key, "what": what, "args": list(args), "change": change, "ok": False, "error": str(exc)[:300]}

    def _trial_run(self, label: str, model: str, port: int, args: list[str], change: dict | None, key: str | None,
                   what: tuple[str, str] | None, code: bool) -> dict:
        out: dict = {"label": label, "key": key, "what": what, "args": list(args), "change": change, "ok": False}
        self._ev("trial", trial=label, step="load", state="start", args=args, change=change, key=key,
                 what=what[0] if what else None, what_en=what[1] if what else None)
        t0 = self._clock()
        self.manager.switch(model, port, args)
        ready = self._wait_ready(label, port)
        out["load_s"] = round(self._clock() - t0, 1)
        if ready is not True:
            out["error"] = ready
            self._ev("trial", trial=label, step="load", state="failed", message=ready)
            return out
        self._ev("trial", trial=label, step="load", state="done", seconds=out["load_s"])
        if self._corpus is None:
            self._corpus = load_corpus()
        rng = random.Random(hash((label, key)) & 0xFFFFFFFF)
        model_id = ((self._http(port, "/v1/models", timeout=30) or {}).get("data") or [{}])[0].get("id") or "default"
        # warm-up, and how many characters one token of this text is for this tokenizer
        probe_text = excerpt(self._corpus["prose"], 4000, rng)
        probe = self._http(port, "/v1/completions", {"model": model_id, "prompt": probe_text, "max_tokens": 8, "ignore_eos": True})
        chars_per_token = max(1.5, len(probe_text) / max(1, _prompt_tokens(probe)[0] or 1000))

        cache = self._http(port, "/v1/cache/status", timeout=30) or {}
        geometry = cache.get("geometry") or {}
        kv_tokens = (geometry.get("num_pages") or 0) * (geometry.get("page_size") or 1)
        prompt_len = max(512, min(PREFILL_TOKENS, (kv_tokens or PREFILL_TOKENS) - DECODE_TOKENS - 256))

        self._check()
        self._ev("trial", trial=label, step="prefill", state="start", tokens=prompt_len)
        prefill = []
        mixed = self._corpus["prose"] + "\n\n" + self._corpus["code"]
        for i in range(2):
            # a new first line each time: nothing of this prompt is in the prefix cache
            text = f"[{label}-{i}-{rng.getrandbits(40):010x}]\n" + excerpt(mixed, int(prompt_len * chars_per_token * 0.97), rng)
            t = self._clock()
            doc = self._prefill_once(label, port, {"model": model_id, "prompt": text, "max_tokens": 1, "ignore_eos": True}, prompt_len)
            tokens, cached = _prompt_tokens(doc)
            tps = max(0, (tokens or prompt_len) - cached) / max(1e-6, self._clock() - t)
            prefill.append(round(tps, 1))
            self._ev("sample", id=f"{label}.prefill", value=round(tps, 1))
        out["prefill_tps"] = max(prefill)
        self._ev("trial", trial=label, step="prefill", state="done", value=out["prefill_tps"])

        for kind in (("prose", "code") if code else ("prose",)):
            self._check()
            step = "decode" if kind == "prose" else "decode_code"
            self._ev("trial", trial=label, step=step, state="start", tokens=DECODE_TOKENS)
            runs = [self._decode(label, port, model_id, rng, kind) for _ in range(DECODE_RUNS)]
            runs = [r for r in runs if r]
            value = round(sorted(runs)[len(runs) // 2], 2) if runs else None
            out["decode_tps" if kind == "prose" else "decode_code_tps"] = value
            self._ev("trial", trial=label, step=step, state="done", value=value)

        experts = self._http(port, "/v1/kai/experts?window=60", timeout=30) or {}
        layers = [l for l in experts.get("layers") or [] if not l.get("mtp")]
        active, miss = sum(l.get("active") or 0 for l in layers), sum(l.get("miss") or 0 for l in layers)
        cache = self._http(port, "/v1/cache/status", timeout=30) or {}
        out.update({
            "ok": True, "geometry": cache.get("geometry") or geometry, "ranks": experts.get("ranks") or [],
            "kv_tokens": kv_tokens, "expert_slots": geometry.get("moe_cache_size"),
            "hit_rate": round(1 - miss / active, 4) if active else None,
            "gpu_free_bytes": min(((r.get("gpu") or {}).get("free_bytes") or 0) for r in experts.get("ranks") or [{}]),
        })
        self._ev("trial", trial=label, step="measured", state="done",
                 kv_tokens=kv_tokens, hit_rate=out["hit_rate"], expert_slots=out["expert_slots"])
        return out

    def _prefill_counter(self, port: int) -> int | None:
        try:
            ranks = (self._http(port, "/v1/kai/experts?window=60", timeout=5) or {}).get("ranks") or []
        except Exception:  # noqa: BLE001 -- progress is a courtesy, the measurement does not need it
            return None
        vals = [(r.get("counters") or {}).get("prefill_new_tokens") for r in ranks[:1]]
        return vals[0] if vals and vals[0] is not None else None

    def _prefill_once(self, label: str, port: int, body: dict, expected: int) -> Any:
        """One timed prompt. The response only comes at the end, so meanwhile the rank's lifetime
        prefill counter (written per chunk, every couple of seconds) is read to show progress."""
        out: dict = {}

        def send() -> None:
            try:
                out["doc"] = self._http(port, "/v1/completions", body)
            except Exception as exc:  # noqa: BLE001 -- re-raised on the job thread
                out["error"] = exc

        base = self._prefill_counter(port)
        t0 = self._clock()
        worker = threading.Thread(target=send, name="ft-mgr-benchmark-prompt", daemon=True)
        worker.start()
        while True:
            worker.join(1.0)
            if not worker.is_alive():
                break
            now = self._prefill_counter(port) if base is not None else None
            elapsed = max(1e-6, self._clock() - t0)
            done = max(0, now - base) if now is not None else 0
            self._ev("progress", id=f"{label}.prefill", done=min(done, expected), total=expected,
                     rate=round(done / elapsed, 1) if done else None, elapsed=round(elapsed, 1))
        if "error" in out:
            raise out["error"]
        return out.get("doc")

    def _decode(self, label: str, port: int, model_id: str, rng: random.Random, kind: str = "prose") -> float | None:
        source = self._corpus["code" if kind == "code" else "prose"] if self._corpus else " ".join(_FALLBACK_WORDS)
        lead = "# Continue this Python module.\n" if kind == "code" else "Continue this document.\n\n"
        prompt = f"[{label}-{kind}-{rng.getrandbits(40):010x}]\n{lead}{excerpt(source, 2400, rng)}"
        body = {"model": model_id, "prompt": prompt, "max_tokens": DECODE_TOKENS, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}
        sample_id = f"{label}.decode" if kind == "prose" else f"{label}.decode_code"
        first = last = None
        chunks, usage = 0, None
        for raw in self._stream(port, "/v1/completions", body):
            line = raw.strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                msg = json.loads(line[5:])
            except ValueError:
                continue
            if msg.get("usage"):
                usage = msg["usage"].get("completion_tokens")
            if not any((c.get("text") or "") for c in msg.get("choices") or []):
                continue
            now = self._clock()
            first = first if first is not None else now
            last = now
            chunks += 1
            if chunks % 20 == 0 and last > first:
                self._ev("sample", id=sample_id, value=round((chunks - 1) / (last - first), 2))
            if self._cancel.is_set():
                break
        if first is None or last is None or last <= first or chunks < 2:
            return None
        # a chunk can carry several tokens (MTP) or none (a token that ends mid-character): the usage count is exact
        tokens = usage if usage else chunks
        return (tokens - 1) / (last - first)

    def _wait_ready(self, label: str, port: int, timeout: float = 1800.0) -> bool | str:
        deadline = self._clock() + timeout
        seen_running = False
        while self._clock() < deadline:
            self._check()
            st = self.manager.status()
            if st.get("running"):
                seen_running = True
            elif seen_running or not st.get("starting"):
                code = st.get("lastExitCode")
                return f"the server exited (code {code})" if code is not None else "the server is not running"
            try:
                h = self._http(port, "/health", timeout=5)
            except Exception:  # noqa: BLE001 -- not listening yet
                h = None
            if isinstance(h, dict):
                if h.get("status") == "ok":
                    return True
                if h.get("status") == "error":
                    return h.get("message") or "the server reported an error"
                prog = h.get("progress") or {}
                if prog.get("total_bytes"):
                    self._ev("sample", id=f"{label}.load", value=round(100 * prog.get("done_bytes", 0) / prog["total_bytes"], 1))
            self._sleep(2)
        return "the server did not become ready in time"
