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
happened. Progress is a list of events the page polls.

A run can also pick up from the model's last result (``from_last``): the hardware stage is taken
from that result, its chosen settings are measured once more as today's base, and only the
candidates it has not tried are run -- the ones a newer build added, or the thorough ones after a
standard run. Every result records the keys tried so far, so these runs add up."""

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

from freetoken.version import __version__ as FT_VERSION

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


class NoLastResult(ValueError):
    pass


def prior_trials(prev: dict | None) -> tuple[set[str], set[str]]:
    """(keys tried, keys kept) by a saved result -- its own record, or, for a result saved before
    runs kept one, what its trials say."""
    if not prev:
        return set(), set()
    if "tried" in prev:
        return set(prev["tried"]), set(prev.get("kept") or ())
    trials = [t for t in prev.get("trials") or [] if t.get("key") and t.get("key") != "base_again"]
    return {t["key"] for t in trials}, {t["key"] for t in trials if t.get("decision") == "kept"}


def closed_groups(prev: dict, tried: set[str], facts: dict, hw: dict) -> set[str]:
    """The groups ``tried`` closes. A result records them itself (its ``tried`` holds groups too);
    for an older one they are read off the plans its first and final settings open: a group whose
    member was kept (four prefill pieces) closes the member that would undo it (one piece) even
    though the final settings no longer offer the kept one."""
    groups: dict[str, str] = {}
    for args in (prev.get("base_args"), prev.get("args")):
        if args:
            for c in search.plan(list(args), facts, hw, "thorough"):
                groups.setdefault(c.key, c.group)
    return {groups[k] for k in tried if k in groups}


def last_refusal(prev: dict | None, gpus: list[int]) -> str | None:
    """Why ``prev`` cannot be picked up from, or None when it can."""
    if not prev:
        return "no_last"
    if not prev.get("args") or not prev.get("hw") or not any(t.get("decision") == "base" for t in prev.get("trials") or []):
        return "last_incomplete"  # a hardware-only run, or one saved without its measured base
    if len(prev.get("upstream") or []) != len(gpus):
        return "last_other_gpus"
    return None


def host_facts(rec: dict) -> dict:
    host = rec.get("host") or {}
    return dict(host.get("model") or {}, ram_total=(host.get("memory") or {}).get("total"),
                weight_bytes=host.get("weight_bytes"))


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
def with_benchmark(rec: dict, result: dict | None) -> dict:
    """The recommendation for a model that was benchmarked on this PC: the benchmark's flags and
    reasons, which were measured, win over the rules; a rule the benchmark did not cover (added
    since) stays. Changes the benchmark tried and did not keep are not recommended, and neither is a
    note that only explains a default (``--moe-hybrid-max-fetch -1`` reads the measurement record by
    itself): every recommended flag is one the benchmark's own flags carry, so a profile saved from
    the result shows nothing left to apply."""
    if not result or not result.get("args") or result.get("error"):
        return rec
    args = dict(parse_flags(result["args"]))

    def carried(n: dict) -> bool:
        if n.get("removed"):
            return n["flag"] not in args
        return n["flag"] in args and (n.get("value") is None or str(args[n["flag"]]) == str(n["value"]))

    measured = [n for n in result.get("notes") or [] if not n.get("rejected") and not n.get("info") and carried(n)]
    covered = {n["flag"] for n in result.get("notes") or []}
    notes = measured + [n for n in rec.get("notes") or [] if n["flag"] not in covered]
    return dict(rec, flags=list(result["args"]), notes=notes,
                benchmark={"finished": result.get("finished"), "mode": result.get("mode"), "use": result.get("use"),
                           "version": result.get("freetoken_version"), "current": FT_VERSION,
                           "trials": len(result.get("trials") or [])})


PROMPT_SEED = 20260917  # every run reads the same excerpts: on two 3060s one 16k prompt ran at 509 tok/s, another at 636
# a prompt whose server stopped working: on the 2060 an MTP run lost its backend ("device not ready")
# while the process and its front end stayed up, and the request waited out its ten minutes
PROMPT_STALL_S = 300  # no chunk finished: even a 16k chunk at 60 tok/s is done by then
PROMPT_UNREACHABLE_S = 60  # the progress counter cannot be read at all
DECODE_SKIP = 60  # chunks not timed: the expert cache is still filling for this prompt
UPSTREAM_QUIET_S = 600  # ft bench bw prints a line per format; one format can take a few minutes


def bench_profiles() -> dict:
    """The ``ft bench bw`` profiles the engine reads, one per GPU, with who measured them and with
    which version. A profile from the desktop app or an older ``ft bench bw`` has no version."""
    from freetoken.moe.bench_profile import _cache_dir

    d = os.path.join(_cache_dir(), "benchbw")
    out = []
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                p = json.load(fh)
        except (OSError, ValueError):
            continue
        g = p.get("gpu") or {}
        out.append({"index": g.get("index"), "gpu": g.get("name"), "uuid": g.get("uuid"), "epoch": p.get("epoch"),
                    "version": p.get("freetoken_version"), "by": p.get("measured_by"),
                    "formats": sorted((p.get("dtypes") or {}).keys())})
    return {"current": FT_VERSION, "profiles": out}


def stamp_profile(path: str, by: str = "ft bench bw (ft mgr benchmark)") -> None:
    """Record the version that measured a profile; ``bench_profile`` ignores keys it does not know."""
    with open(path) as fh:
        prof = json.load(fh)
    prof["freetoken_version"], prof["measured_by"] = FT_VERSION, by
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(prof, fh, indent=2)
    os.replace(tmp, path)


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def generation(t: dict, use: str) -> float:
    """The generation figure that decides, for the use the person picked."""
    prose = t.get("decode_tps") or 0.0
    code = t.get("decode_code_tps") or prose
    if use == "prose":
        return prose
    if use == "code":
        return code
    return math.sqrt(prose * code) if prose and code else prose or code


def generations(best: dict, t: dict, use: str) -> tuple[float, float]:
    """The two generation figures to compare, measured the same way. Only MTP runs (and the base of
    an MTP model) measure code as well; a run without it is compared on prose alone. Comparing the
    base's code figure with a run's prose figure dropped a 16-bit KV cache that doubled prompt
    processing on two 3060s (18.34 code against 17.75 prose)."""
    if best.get("decode_code_tps") and t.get("decode_code_tps"):
        return generation(best, use), generation(t, use)
    return best.get("decode_tps") or 0.0, t.get("decode_tps") or 0.0


def keep(best: dict, t: dict, c: search.Candidate, use: str) -> bool:
    """A change is kept when either figure gains while the other holds, whatever it was tried for:
    a 16-bit KV cache is tried for generation, but on two 3060s it doubled prompt processing."""
    if not t.get("ok"):
        return False
    g0, g1 = generations(best, t, use)
    p0, p1 = best.get("prefill_tps") or 0.0, t.get("prefill_tps") or 0.0
    gen_gain = c.gain if c.goal == "gen" else 1.03
    prefill_gain = c.gain if c.goal == "prefill" else 1.05
    return (g1 >= gen_gain * g0 and p1 >= c.hold_prefill * p0) or (p1 >= prefill_gain * p0 and g1 >= c.hold_gen * g0)


FIGURES = ("prefill_tps", "decode_tps", "decode_code_tps")


def close_on_generation(best: dict, t: dict, c: search.Candidate, use: str, band: float = 0.8) -> bool:
    """Not kept only because generation fell short of holding, by no more than the swing seen
    between starts of the same settings, while prompt processing gained enough to keep it. Worth
    starting the base again for; a run that gained nothing is not, however its generation moved."""
    if not t.get("ok"):
        return False
    g0, g1 = generations(best, t, use)
    p0, p1 = best.get("prefill_tps") or 0.0, t.get("prefill_tps") or 0.0
    prefill_gain = c.gain if c.goal == "prefill" else 1.05
    return bool(g0) and p1 >= prefill_gain * p0 and band * g0 <= g1 < c.hold_gen * g0


def slower_of(a: dict, b: dict) -> dict:
    """Two runs of the same settings as one: the later run, with the slower of each figure, so one
    lucky run cannot become the bar every later candidate is measured against."""
    out = dict(b)
    for k in FIGURES:
        if a.get(k) and b.get(k):
            out[k] = min(a[k], b[k])
    return out


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


def load_tokenizer(model: str):
    """The model's own tokenizer, to cut prompts to an exact token count; None when it cannot be
    loaded (the characters-per-token estimate is used then)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001
        return None


def fit_tokens(tokenizer, text: str, tokens: int) -> str:
    """``text`` cut to ``tokens`` tokens. Characters per token ranges from 3.0 to 4.1 across excerpts
    of this repository, so an estimate from a probe either overflows the context (HTTP 400) or
    measures a shorter prompt than the other runs did."""
    if tokenizer is None:
        return text
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= tokens else tokenizer.decode(ids[:tokens])
    except Exception:  # noqa: BLE001
        return text


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
    with urllib.request.urlopen(req, timeout=180) as resp:  # per read: tokens arrive every second or so
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


def _visible_gpus() -> list[int]:
    from .recommend import gpus

    return [g["index"] for g in gpus()]


class TuneJob:
    def __init__(self, manager, *, state_dir: str, python: str, default_port: int,
                 recommend: Callable[[str], dict] | None = None, spawn: Callable[..., Any] | None = None,
                 http: Callable[..., Any] | None = None, stream: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 corpus: dict | None = None, tokenizer: Callable[[str], Any] = load_tokenizer,
                 gpu_indices: Callable[[], list[int]] | None = None):
        self.manager = manager
        self.dir = os.path.join(state_dir, "tune")
        self.python = python
        self.default_port = default_port
        self._recommend = recommend
        self._spawn = spawn or (lambda argv, env=None: subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                                         text=True, env=env))
        self._gpu_indices = gpu_indices or _visible_gpus
        self._http = http or _http
        self._stream = stream or _stream
        self._clock = clock
        self._sleep = sleep
        self._corpus = corpus
        self._load_tokenizer, self._tokenizer = tokenizer, None
        self._run_seconds: list[float] = []
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
    def start(self, model: str, trials: bool = True, port: int | None = None, mode: str = "standard", use: str = "both",
              from_last: bool = False) -> dict:
        prev = None
        if from_last:
            trials = True  # picking up is only ever about the runs
            prev = self.last(model)
            why = last_refusal(prev, self._gpu_indices() or [0])
            if why:
                raise NoLastResult(why)
        with self._lock:
            if self.state == "running":
                raise Busy("a benchmark is already running")
            # a new run: the page sees another run id and reads its events from the start
            self.run_id += 1
            self.state, self.model, self.events, self.result, self.error, self._seq = "running", model, [], None, None, 0
        self._cancel.clear()
        mode = mode if mode in MODES else "standard"
        use = use if use in USES else "both"
        self._thread = threading.Thread(target=self._run, args=(model, trials, port or self.default_port, mode, use, prev),
                                        name="ft-mgr-benchmark", daemon=True)
        self._thread.start()
        return {"started": True}

    def pending(self, model: str, mode: str = "standard") -> dict:
        """What a run picking up from the last result would try, before any measurement: the
        candidates the last result's settings open that it has not tried. Ones that only open
        after a change is kept cannot be known in advance and are not listed."""
        prev = self.last(model)
        why = last_refusal(prev, self._gpu_indices() or [0])
        if why:
            return {"available": False, "reason": why, "items": []}
        mode = mode if mode in MODES else "standard"
        rec = self._recommend(model) if self._recommend else {}
        tried, kept = prior_trials(prev)
        facts, hw = host_facts(rec), prev.get("hw") or {}
        tried |= closed_groups(prev, tried, facts, hw)
        args = list(prev["args"])
        plan = search.plan(args, facts, hw, mode)
        # a candidate the settings already carry (the chunk ceiling four pieces set) is skipped by the
        # run without a start, so it is not listed either
        items = [{"key": c.key, "what": c.what_ja, "what_en": c.what_en, "change": c.change()} for c in plan
                 if c.key not in tried and c.group not in tried and (c.after is None or c.after in kept)
                 and set_flags(args, search.merge_changes(args, c.changes), c.drop) != args]
        return {"available": True, "items": items, "last": {"finished": prev.get("finished"), "mode": prev.get("mode"),
                                                            "version": prev.get("freetoken_version")}}

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
    def _run(self, model: str, trials: bool, port: int, mode: str = "standard", use: str = "both",
             last: dict | None = None) -> None:
        prev = self.manager.status()
        prev_cfg = (prev.get("model"), prev.get("port"), list(self.manager.serve_args())) if prev.get("running") else None
        result: dict = {"model": model, "port": port, "started": time.time(), "trials": [], "mode": mode, "use": use,
                        "freetoken_version": FT_VERSION,
                        "skipped": [{"flag": f, "why": ja, "why_en": en} for f, ja, en in search.SKIPPED]}
        try:
            if prev_cfg:
                self._ev("phase", phase="stop")
                self.manager.stop()
            self._check()
            rec = None
            if last is not None:
                # picking up: the hardware stage and the settings it led to are the last result's
                result["from_last"] = {"started": last.get("started"), "finished": last.get("finished"),
                                       "mode": last.get("mode"), "version": last.get("freetoken_version")}
                # the runs this result builds on, oldest first, so the record of what was measured stays whole
                result["earlier"] = list(last.get("earlier") or []) + [
                    {"started": last.get("started"), "finished": last.get("finished"), "mode": last.get("mode"),
                     "version": last.get("freetoken_version"), "chosen": last.get("chosen"), "trials": last.get("trials") or []}]
                result["upstream"], hw = last.get("upstream"), last["hw"]
                result["hw"], result["base_args"] = hw, list(last.get("base_args") or last["args"])
                args, notes = list(last["args"]), [dict(n) for n in last.get("notes") or []]
                self._ev("phase", phase="from_last", finished=last.get("finished"))
            else:
                # upstream's own measurement first, exactly as the desktop app runs it: the engine reads
                # this profile (hybrid or offload, the fetch split), so it is settled before anything else
                self._ev("phase", phase="upstream")
                result["upstream"] = self._upstream()
                self._check()
                self._ev("phase", phase="hw")
                hw = self._hardware(model)
                result["hw"] = hw
                self._check()
                rec = self._recommend(model) if self._recommend else {"flags": [], "notes": []}
                args, notes = self._apply_hw(rec, hw)
                result["base_args"] = list(args)
            if trials:
                if rec is None:
                    rec = self._recommend(model) if self._recommend else {"flags": [], "notes": []}
                facts = host_facts(rec)
                prior = None
                if last is not None:
                    done, kept = prior_trials(last)
                    prior = (done | closed_groups(last, done, facts, hw), kept)
                args, notes = self._trials(model, port, args, notes, result, facts, hw, mode, use, prior=prior)
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
                self._switch(prev_cfg[0], prev_cfg[1], prev_cfg[2])
            elif trials and self.manager.status().get("running"):
                self.manager.stop()
        except Exception as exc:  # noqa: BLE001
            self._ev("log", msg=f"could not restore the previous server: {exc}")

    def _switch(self, model: str, port: int, args: list[str]) -> None:
        """Start ``args``, replacing what runs. A run that took the server down (the 2060's driver
        once reported "device not ready" under MTP) leaves a front end that cannot answer the stop's
        accounting request; without force every later run then failed on that stop."""
        try:
            self.manager.switch(model, port, args)
        except Exception as exc:  # noqa: BLE001
            self._ev("log", msg=f"switch retried with force: {str(exc)[:200]}")
            self.manager.switch(model, port, args, force=True)

    def _lines(self, proc, quiet_s: float, where: Callable[[], str]):
        """The child's output lines; a child that stays quiet for ``quiet_s`` is stopped, since a GPU
        kernel that hangs never returns."""
        import queue

        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            for raw in proc.stdout:
                lines.put(raw)
            lines.put(None)

        threading.Thread(target=pump, name="ft-mgr-benchmark-child", daemon=True).start()
        while True:
            try:
                line = lines.get(timeout=quiet_s)
            except queue.Empty:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(f"{where()} sent nothing for {int(quiet_s)} s and was stopped")
            if line is None:
                return
            yield line.strip()

    def _upstream(self, quiet_s: float = UPSTREAM_QUIET_S) -> list[dict]:
        indices = self._gpu_indices() or [0]
        self._ev("hw", kind="plan", steps=[{"id": f"upstream{i}", "unit": "%", "max": 100} for i in indices])
        out = []
        for i in indices:
            self._check()
            sid = f"upstream{i}"
            self._ev("hw", kind="step", id=sid)
            argv = [self.python, "-m", "freetoken.cli", "bench", "bw", "--gpu", str(i)]
            env = dict(os.environ, FREETOKEN_BENCH_PROGRESS="1")
            try:
                proc = self._spawn(argv, env=env)
            except TypeError:  # a spawn that takes no environment
                proc = self._spawn(argv)
            self._proc = proc
            path, tail = None, []
            for line in self._lines(proc, quiet_s, lambda: f"ft bench bw on GPU {i}"):
                if line.startswith("FTBENCH_OUT "):
                    path = line[len("FTBENCH_OUT "):]
                elif line.startswith("FTBENCH "):
                    parts = line.split(maxsplit=3)
                    try:
                        done, total = int(parts[1]), int(parts[2])
                    except (IndexError, ValueError):
                        continue
                    self._ev("hw", kind="sample", id=sid, value=round(100 * done / max(1, total)),
                             label=parts[3] if len(parts) > 3 else "")
                elif line:
                    tail = (tail + [line])[-8:]
            code = proc.wait()
            self._check()
            if code:
                raise RuntimeError(f"ft bench bw on GPU {i} failed (exit {code}): " + " / ".join(tail)[-600:])
            if path:
                try:
                    stamp_profile(path)
                except (OSError, ValueError):
                    pass
            self._ev("hw", kind="done", id=sid, value=None, version=FT_VERSION)
            out.append({"gpu": i, "path": path, "version": FT_VERSION, "epoch": int(time.time())})
        return out

    def _hardware(self, model: str, quiet_s: float = HW_QUIET_S) -> dict:
        proc = self._spawn([self.python, "-m", "freetoken.webui.hwbench", "--model", model])
        self._proc = proc
        out: dict = {}
        step = None
        for line in self._lines(proc, quiet_s, lambda: f"hardware step {step or '?'}"):
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
                facts: dict, hw: dict, mode: str, use: str, prior: tuple[set[str], set[str]] | None = None):
        self._ev("phase", phase="trial")
        self._tokenizer = self._load_tokenizer(model)
        labels = _labels()
        mtp_model = bool(facts.get("mtp"))
        # every run of an MTP model is measured on code too (unless the use is prose): a verdict on
        # generation compares the same kind of figure, and MTP is judged against runs that had it
        measure_code = mtp_model and use != "prose"
        base = set_flags(args, {"--moe-collect-stats": None})
        a = self._trial(next(labels), model, port, base, code=measure_code)
        if not a["ok"] and a.get("stage") == "load" and has_flag(base, "--dtype"):
            # float16 on a pre-Ampere card copies while converting: a tight model may not load with it
            result["trials"].append(a)
            self._ev("log", msg="dtype_retry")
            base = set_flags(base, {}, drop=("--dtype",))
            notes = [n for n in notes if n["flag"] != "--dtype"]
            notes.append({"flag": "--dtype", "value": None, "source": "measured", "removed": True,
                          "why": "float16 では読み込み時の変換で VRAM が足りず起動できなかったので、指定を外しました。",
                          "why_en": "The model did not load with float16 (the conversion needs VRAM it did not have), so the flag was dropped."})
            a = self._trial(next(labels), model, port, base, change={"flag": "--dtype", "value": None, "removed": True},
                            code=measure_code)
        a["decision"] = "base"
        result["trials"].append(a)
        result["skipped"] = search.unavailable(facts, hw) + result.get("skipped", [])
        if not a["ok"]:
            raise RuntimeError(f"the server did not start with the measured flags: {a.get('error')}")
        best, best_args = a, base
        # picking up: what the earlier runs tried counts as tried here (and closes its group), and what
        # they kept opens the candidates that wait on it
        prior_tried, prior_kept = prior or (set(), set())
        tried: set[str] = set(prior_tried)  # candidate keys and the groups they close
        kept: set[str] = set(prior_kept)
        kept_now = False
        first = True
        while True:
            self._check()
            pending = [c for c in search.plan(best_args, facts, hw, mode)
                       if c.key not in tried and c.group not in tried and (c.after is None or c.after in kept)]
            # the page lists what is still to come, and about how long it takes; sent again before every run
            self._ev("plan", items=[{"key": x.key, "what": x.what_ja, "what_en": x.what_en, "change": x.change()} for x in pending],
                     mode=mode, use=use, update=not first,
                     eta_s=self._eta(len(pending), 0 if prior is not None and not kept_now else
                                     len(context_candidates(best.get("geometry") or {}, facts.get("max_context")))))
            first = False
            if not pending:
                break
            c = pending[0]
            tried.update({c.key, c.group})
            cand_args = set_flags(best_args, search.merge_changes(best_args, c.changes), c.drop)
            if cand_args == best_args:
                continue
            t = self._trial(next(labels), model, port, cand_args, change=c.change(), key=c.key,
                            what=(c.what_ja, c.what_en), code=measure_code)
            won = keep(best, t, c, use)
            result["trials"].append(t)
            if not won and close_on_generation(best, t, c, use):
                # generation on the 2060 swung 28-38 tok/s between starts of the same settings: before a
                # change is dropped for generation alone, the best settings are started again, and the
                # change is judged against that fresh figure
                t["decision"] = "rebase"
                self._ev("decision", trial=t["label"], key=c.key, decision="rebase")
                again = self._trial(next(labels), model, port, best_args, change=None, key="base_again",
                                    what=("今の設定をもう一度測る（生成の揺れを確かめる）", "the current settings once more (does generation swing?)"),
                                    code=measure_code)
                again["decision"] = "base"
                result["trials"].append(again)
                self._ev("decision", trial=again["label"], key="base_again", decision="base")
                if again.get("ok"):
                    best = dict(best, **{k: again[k] for k in FIGURES if again.get(k)})
                    won = keep(best, t, c, use)
            measured = t
            if won:
                # a single run can be lucky (the 2060 once gave 37 tok/s for settings that then gave 32), and
                # so can the run that checks it be unlucky (four prefill pieces on the 2060: 733 tok/s,
                # then 638 with generation at 24.7 against 30-31 everywhere else). Start it again; when
                # the two disagree, a third start decides, and two runs out of three must clear the bar
                runs = [t]
                t["decision"] = "recheck"
                self._ev("decision", trial=t["label"], key=c.key, decision="recheck")
                for n, (ja, en) in enumerate((("もう一度測って確かめる", "measured again to confirm"),
                                              ("結果が割れたので 3 回目で決める", "the two disagree: a third run decides"))):
                    again = self._trial(next(labels), model, port, cand_args, change=c.change(), key=c.key,
                                        what=(f"{c.what_ja}（{ja}）", f"{c.what_en} ({en})"), code=measure_code)
                    again["recheck"] = True
                    result["trials"].append(again)
                    runs.append(again)
                    passed = [x for x in runs if keep(best, x, c, use)]
                    if len(passed) >= 2 or len(runs) - len(passed) >= 2:
                        break
                    again["decision"] = "tiebreak"
                    self._ev("decision", trial=again["label"], key=c.key, decision="tiebreak")
                won = len(passed) >= 2
                measured = slower_of(passed[0], passed[1]) if won else runs[-1]
                for x in runs[:-1]:
                    x["decision"] = "kept" if won else "rejected"
                    self._ev("decision", trial=x["label"], key=c.key, decision=x["decision"])
                t = runs[-1]
            t["decision"] = "kept" if won else ("failed" if not t.get("ok") else "rejected")
            self._ev("decision", trial=t["label"], key=c.key, decision=t["decision"])
            notes = self._note(notes, c, best, measured, won, use)
            if won:
                best, best_args = measured, cand_args
                kept.add(c.key)
                kept_now = True

        # longer contexts last, on everything kept: a tier is kept while generation holds 95%. Picking up
        # with nothing newly kept, the last result's tiers still stand and are not run again
        model_max = facts.get("max_context")
        tiers = context_candidates(best.get("geometry") or {}, model_max) if (prior is None or kept_now) else []
        if has_flag(best_args, "--num-tokens"):
            # the KV is sized explicitly (one card: Flash-Next on a 12 GB card, recommend.py): the tiers
            # size it through --kv-reserve-tokens, which an explicit size ignores, so they would only
            # move the advertised length away from what fits
            tiers = []
        for tier in tiers:
            self._check()
            c = search.Candidate(f"context_{tier}", {"--kv-reserve-tokens": str(tier), "--max-seq-len-override": str(tier)},
                                 gain=0.95, hold_prefill=0.9,
                                 what_ja=f"コンテキスト長を {tier:,} にする", what_en=f"a context of {tier:,} tokens")
            cand_args = set_flags(best_args, c.changes)
            t = self._trial(next(labels), model, port, cand_args, change=c.change(), key=c.key, what=(c.what_ja, c.what_en),
                            code=measure_code)
            tried.add(c.key)
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
        # what every run so far has tried (keys and the groups they closed) and kept, for the next run
        # that picks up from this one
        result["tried"], result["kept"] = sorted(tried), sorted(kept)
        return best_args, notes

    def _eta(self, candidates: int, tiers: int) -> int | None:
        """Seconds left: runs so far at their median length, a quarter more for the second (and
        third) measurements of what turns out faster, and the context tiers at the end."""
        if not self._run_seconds:
            return None
        per = sorted(self._run_seconds)[len(self._run_seconds) // 2]
        return int(per * (candidates * 1.25 + tiers))

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
        t0 = self._clock()
        try:
            return self._trial_run(label, model, port, args, change, key, what, code)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 -- e.g. a wider chunk that runs out of VRAM mid-prompt
            self._ev("trial", trial=label, step="measure", state="failed", message=str(exc)[:300])
            return {"label": label, "key": key, "what": what, "args": list(args), "change": change, "ok": False, "error": str(exc)[:300]}
        finally:
            self._run_seconds.append(self._clock() - t0)

    def _trial_run(self, label: str, model: str, port: int, args: list[str], change: dict | None, key: str | None,
                   what: tuple[str, str] | None, code: bool) -> dict:
        out: dict = {"label": label, "key": key, "what": what, "args": list(args), "change": change, "ok": False}
        self._ev("trial", trial=label, step="load", state="start", args=args, change=change, key=key,
                 what=what[0] if what else None, what_en=what[1] if what else None)
        t0 = self._clock()
        self._switch(model, port, args)
        ready = self._wait_ready(label, port)
        out["load_s"] = round(self._clock() - t0, 1)
        if ready is not True:
            out["error"], out["stage"] = ready, "load"
            self._ev("trial", trial=label, step="load", state="failed", message=ready)
            return out
        self._ev("trial", trial=label, step="load", state="done", seconds=out["load_s"])
        if self._corpus is None:
            self._corpus = load_corpus()
        rng = random.Random(PROMPT_SEED)
        model_id = ((self._http(port, "/v1/models", timeout=30) or {}).get("data") or [{}])[0].get("id") or "default"
        # warm-up, and how many characters one token of the prompt's mixed text is for this tokenizer
        mixed = self._corpus["prose"] + "\n\n" + self._corpus["code"]
        probe_text = excerpt(mixed, 6000, rng)
        probe = self._http(port, "/v1/completions", {"model": model_id, "prompt": probe_text, "max_tokens": 8, "ignore_eos": True})
        chars_per_token = max(1.5, len(probe_text) / max(1, _prompt_tokens(probe)[0] or 1000))

        cache = self._http(port, "/v1/cache/status", timeout=30) or {}
        geometry = cache.get("geometry") or {}
        kv_tokens = (geometry.get("num_pages") or 0) * (geometry.get("page_size") or 1)
        limit = min(x for x in (kv_tokens, _int(search.flag_value(args, "--max-seq-len-override")), PREFILL_TOKENS + 1024) if x)
        prompt_len = max(512, min(PREFILL_TOKENS, limit - 1024))

        self._check()
        self._ev("trial", trial=label, step="prefill", state="start", tokens=prompt_len)
        prefill = []
        for i in range(2):
            # a new first line each time: nothing of this prompt is in the prefix cache
            head = f"[{label}-{i}-{os.urandom(5).hex()}]\n"  # nothing of it is in a prefix cache, the rest is the same text
            if self._tokenizer is not None:
                body = fit_tokens(self._tokenizer, excerpt(mixed, int(prompt_len * 5), rng), prompt_len - 32)
            else:
                body = excerpt(mixed, int(prompt_len * chars_per_token * 0.8), rng)
            text = head + body
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
            self._decode(label, port, model_id, rng, kind)  # untimed: routing and the cache settle on this text
            runs = [self._decode(label, port, model_id, rng, kind) for _ in range(DECODE_RUNS)]
            runs = [r for r in runs if r]
            value = round(max(runs), 2) if runs else None
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
        seen, changed, reachable = base, t0, t0
        worker = threading.Thread(target=send, name="ft-mgr-benchmark-prompt", daemon=True)
        worker.start()
        while True:
            worker.join(1.0)
            if not worker.is_alive():
                break
            if not self.manager.status().get("running"):
                raise RuntimeError("the server exited during the prompt")
            now = self._prefill_counter(port) if base is not None else None
            tick = self._clock()
            if base is not None:
                if now is None:
                    if tick - reachable > PROMPT_UNREACHABLE_S:
                        raise RuntimeError(f"the server stopped answering for {int(PROMPT_UNREACHABLE_S)} s during the prompt")
                else:
                    reachable = tick
                    if now != seen:
                        seen, changed = now, tick
                    elif tick - changed > PROMPT_STALL_S:
                        raise RuntimeError(f"the prompt made no progress for {int(PROMPT_STALL_S)} s")
            elapsed = max(1e-6, tick - t0)
            done = max(0, now - base) if now is not None else 0
            self._ev("progress", id=f"{label}.prefill", done=min(done, expected), total=expected,
                     rate=round(done / elapsed, 1) if done else None, elapsed=round(elapsed, 1))
        if "error" in out:
            raise out["error"]
        return out.get("doc")

    def _decode(self, label: str, port: int, model_id: str, rng: random.Random, kind: str = "prose") -> float | None:
        source = self._corpus["code" if kind == "code" else "prose"] if self._corpus else " ".join(_FALLBACK_WORDS)
        lead = "# Continue this Python module.\n" if kind == "code" else "Continue this document.\n\n"
        prompt = f"[{label}-{kind}-{os.urandom(5).hex()}]\n{lead}{excerpt(source, 2400, rng)}"
        body = {"model": model_id, "prompt": prompt, "max_tokens": DECODE_TOKENS, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}
        sample_id = f"{label}.decode" if kind == "prose" else f"{label}.decode_code"
        first = last = None
        chunks, usage, skip_t = 0, None, None
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
            if chunks == DECODE_SKIP + 1:
                skip_t = now
            if chunks % 20 == 0 and last > first:
                self._ev("sample", id=sample_id, value=round((chunks - 1) / (last - first), 2))
            if self._cancel.is_set():
                break
        if first is None or last is None or last <= first or chunks < 2:
            return None
        # a chunk can carry several tokens (MTP) or none (a token that ends mid-character): the usage count is exact
        tokens = usage if usage else chunks
        if skip_t is not None and last > skip_t:
            return tokens * (chunks - DECODE_SKIP - 1) / chunks / (last - skip_t)
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
