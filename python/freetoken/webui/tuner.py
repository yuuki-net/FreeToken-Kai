"""The console's benchmark: measure this PC with the model, then settle the flags on measurements.

Two stages, one background job in ``ft mgr`` (torch-free; the GPU work runs in child processes):

1. Hardware (``webui/hwbench.py``, a child process): PCIe, RAM and SSD rates, and the model's
   own expert kernels on the CPU (swept over thread counts) and over PCIe. Decides the MoE
   strategy and the CPU threads.
2. Real runs (optional): start ``ft serve`` with those flags, send a prompt of random words (no
   prefix-cache hits; its token count comes back in the usage) and a fixed generation, then read the
   cache geometry.
   If the free VRAM (or at most a quarter of an auto-sized expert cache) holds a longer context,
   run again with it and keep it only when generation stays within 5% and prompt processing
   within 10%.

The engine the manager was running is stopped first (the measurements need the GPU) and started
again at the end, whatever happened. Progress is a list of events the page polls."""

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

GiB = 1 << 30
# the longest a hardware step has gone without a line (the CPU sweep on many cores) is well under this
HW_QUIET_S = 300.0
CONTEXT_TIERS = (32768, 65536, 131072, 262144)
# long enough for several chunks: two 8k chunks hide what the two-rank overlap and a wider chunk buy
PREFILL_TOKENS = 16384
# three runs, median: on a 2060 two 200-token runs swung 31-36 tok/s between runs that should match
DECODE_TOKENS = 300
DECODE_RUNS = 3


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
def longer_context(geometry: dict, ranks: list[dict], cache_auto: bool, model_max: int | None,
                   max_expert_share: float = 0.25, headroom: int = 600 << 20) -> dict | None:
    """The longest context tier the measured free VRAM holds, taking at most ``max_expert_share``
    of an auto-sized expert cache for the rest. None when nothing longer fits."""
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


def keep_faster_prefill(a: dict, b: dict, gain: float = 1.05, decode_share: float = 0.97) -> bool:
    return bool(b.get("ok")) and (b.get("prefill_tps") or 0) >= gain * (a.get("prefill_tps") or 0) and \
        (b.get("decode_tps") or 0) >= decode_share * (a.get("decode_tps") or 0)


def keep_faster_decode(a: dict, b: dict, gain: float = 1.03, prefill_share: float = 0.9) -> bool:
    return bool(b.get("ok")) and (b.get("decode_tps") or 0) >= gain * (a.get("decode_tps") or 0) and \
        (b.get("prefill_tps") or 0) >= prefill_share * (a.get("prefill_tps") or 0)


def strategy_ratio(hw: dict) -> float | None:
    """CPU expert compute over the slowest GPU's expert transfer, from the hardware bench."""
    m = hw.get("measurements") or {}
    cpu = (m.get("cpu_moe") or {}).get("best_gbs")
    links = [g.get("gbs") for g in (m.get("gather") or {}).values() if g and g.get("gbs")]
    return cpu / min(links) if cpu and links else None


def keep_longer(a: dict, b: dict, decode_share: float = 0.95, prefill_share: float = 0.9) -> bool:
    return bool(b.get("ok")) and (b.get("decode_tps") or 0) >= decode_share * (a.get("decode_tps") or 0) and \
        (b.get("prefill_tps") or 0) >= prefill_share * (a.get("prefill_tps") or 0)


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


_WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue "
    "side kind head house service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office door health person "
    "art war history party result change morning reason research girl guy moment air teacher force education "
    "river stone light green quiet early open simple strong small large young old long short heavy clear warm"
).split()


def _random_text(words: int, rng: random.Random) -> str:
    # ft serve takes text prompts only; random words keep the prefix cache from shortening the run
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def _prompt_tokens(doc: Any) -> tuple[int, int]:
    usage = (doc or {}).get("usage") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return int(usage.get("prompt_tokens") or 0), int(cached)


class TuneJob:
    def __init__(self, manager, *, state_dir: str, python: str, default_port: int,
                 recommend: Callable[[str], dict] | None = None, spawn: Callable[..., Any] | None = None,
                 http: Callable[..., Any] | None = None, stream: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
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
            if len(self.events) > 5000:
                del self.events[:1000]

    def status(self, since: int = 0) -> dict:
        with self._lock:
            return {"run": self.run_id, "state": self.state, "model": self.model, "error": self.error, "result": self.result,
                    "events": [e for e in self.events if e["seq"] > since], "seq": self._seq}

    # ---- control
    def start(self, model: str, trials: bool = True, port: int | None = None) -> dict:
        with self._lock:
            if self.state == "running":
                raise Busy("a benchmark is already running")
            # a new run: the page sees another run id and reads its events from the start
            self.run_id += 1
            self.state, self.model, self.events, self.result, self.error, self._seq = "running", model, [], None, None, 0
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(model, trials, port or self.default_port),
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
    def _run(self, model: str, trials: bool, port: int) -> None:
        prev = self.manager.status()
        prev_cfg = (prev.get("model"), prev.get("port"), list(self.manager.serve_args())) if prev.get("running") else None
        result: dict = {"model": model, "port": port, "started": time.time(), "trials": []}
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
                model_max = ((rec.get("host") or {}).get("model") or {}).get("max_context")
                args, notes = self._trials(model, port, args, notes, result, model_max, hw)
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
                model_max: int | None = None, hw: dict | None = None):
        """One change per run, each kept only when the measurements say so: the prefill chunk
        budget, a longer context, and the other MoE strategy when the hardware bench was close."""
        self._ev("phase", phase="trial")
        letters = iter("ABCDEFGHIJ")
        base = set_flags(args, {"--moe-collect-stats": None})
        label = next(letters)
        a = self._trial(label, model, port, base)
        if not a["ok"] and has_flag(base, "--dtype"):
            # float16 on a pre-Ampere card copies while converting: a tight model may not load with it
            result["trials"].append(a)
            self._ev("log", msg="dtype_retry")
            base = set_flags(base, {}, drop=("--dtype",))
            notes = [n for n in notes if n["flag"] != "--dtype"]
            notes.append({"flag": "--dtype", "value": None, "source": "measured", "removed": True,
                          "why": "float16 では読み込み時の変換で VRAM が足りず起動できなかったので、指定を外しました。",
                          "why_en": "The model did not load with float16 (the conversion needs VRAM it did not have), so the flag was dropped."})
            label = next(letters)
            a = self._trial(label, model, port, base, change={"flag": "--dtype", "value": None})
        result["trials"].append(a)
        if not a["ok"]:
            raise RuntimeError(f"the server did not start with the measured flags: {a.get('error')}")
        best, best_args = a, base

        def attempt(changes: dict, drop: tuple = ()) -> tuple[dict, list[str]]:
            self._check()
            cand_args = set_flags(best_args, changes, drop)
            change = next(iter(changes.items())) if changes else (drop[0], None)
            t = self._trial(next(letters), model, port, cand_args, change={"flag": change[0], "value": change[1]})
            result["trials"].append(t)
            return t, cand_args

        def speeds(x: dict) -> str:
            return f"{x.get('prefill_tps') or 0:.0f} / {x.get('decode_tps') or 0:.1f}"

        # 1. the prefill chunk budget: a wider chunk is the largest prefill lever this fork measured
        if not has_flag(best_args, "--prefill-chunk-budget"):
            t, t_args = attempt({"--prefill-chunk-budget": "0.75"})
            if keep_faster_prefill(best, t):
                notes.append({"flag": "--prefill-chunk-budget", "value": "0.75", "source": "measured",
                              "why": f"実測で決めました: 0.75 にするとプロンプト処理 / 生成が {speeds(best)} → {speeds(t)} tok/s。プロンプト処理が速くなり、生成は落ちないので採ります。",
                              "why_en": f"Measured: at 0.75 prompt processing / generation went {speeds(best)} -> {speeds(t)} tok/s. Prompts are faster and generation holds, so it is kept."})
                best, best_args = t, t_args
            else:
                notes.append({"flag": "--prefill-chunk-budget", "value": "0.75", "source": "measured", "rejected": True,
                              "why": (f"試しました: 0.75 ではプロンプト処理 / 生成が {speeds(best)} → {speeds(t)} tok/s で、5% 以上速くならなかったので既定（0.55）のままにします。"
                                      if t.get("ok") else "試しました: 0.75 では起動または測定に失敗したので、既定（0.55）のままにします。"),
                              "why_en": (f"Tried: at 0.75 prompt processing / generation went {speeds(best)} -> {speeds(t)} tok/s, not 5% faster, so the default (0.55) stays."
                                         if t.get("ok") else "Tried: 0.75 failed to start or to finish the measurement, so the default (0.55) stays.")})

        # 2. a longer context, into the VRAM the best run left free
        cand = longer_context(best.get("geometry") or {}, best.get("ranks") or [], has_flag(best_args, "--moe-cache-auto"), model_max)
        result["candidate"] = cand
        now = best.get("kv_tokens") or 0
        if cand:
            tokens = str(cand["tokens"])
            t, t_args = attempt({"--kv-reserve-tokens": tokens, "--max-seq-len-override": tokens})
            if keep_longer(best, t):
                why = (f"実測で決めました: コンテキスト長 {now:,} では プロンプト処理 / 生成が {speeds(best)} tok/s、{cand['tokens']:,} では {speeds(t)} tok/s。"
                       "長くしても速さがほぼ変わらないので長い方にします。")
                why_en = (f"Measured: at {now:,} tokens prompt processing / generation was {speeds(best)} tok/s, at {cand['tokens']:,} {speeds(t)}. "
                          "The longer context costs almost nothing, so it is kept.")
                best, best_args = t, t_args
            elif t.get("ok"):
                why = (f"実測で決めました: コンテキスト長を {cand['tokens']:,} にするとプロンプト処理 / 生成が {speeds(best)} → {speeds(t)} tok/s でした。"
                       f"生成 95%・プロンプト処理 90% を下回るので、{now:,} のままにします。")
                why_en = (f"Measured: at {cand['tokens']:,} tokens prompt processing / generation went {speeds(best)} -> {speeds(t)} tok/s, "
                          f"below the 95% / 90% kept for a longer context, so {now:,} stays.")
            else:
                why = f"コンテキスト長 {cand['tokens']:,} では起動できなかったので、{now:,} のままにします。"
                why_en = f"The server did not start with {cand['tokens']:,} tokens, so {now:,} stays."
        else:
            why = f"起動後の空き VRAM では、コンテキスト長 {now:,} より長いものが入りませんでした（エキスパートの枠を 1/4 以上削る必要がある）。"
            why_en = f"After the start the free VRAM held no context longer than {now:,} tokens without giving up more than a quarter of the expert cache."
        tokens = next((v for k, v in parse_flags(best_args) if k == "--kv-reserve-tokens"), None)
        notes = [n for n in notes if n["flag"] not in ("--kv-reserve-tokens", "--max-seq-len-override")]
        if tokens:
            notes.append({"flag": "--kv-reserve-tokens", "value": tokens, "why": why, "why_en": why_en, "source": "measured"})
            notes.append({"flag": "--max-seq-len-override", "value": tokens, "source": "rule",
                          "why": "宣伝する長さと実際に入る長さをそろえます。", "why_en": "The advertised context matches what actually fits."})

        # 3. the other MoE strategy, only when the kernels alone could not tell them apart
        ratio = strategy_ratio(hw or {})
        strategy = next((v for k, v in parse_flags(best_args) if k == "--moe-strategy"), None)
        if ratio is not None and 1.5 <= ratio <= 3.0 and strategy in ("hybrid", "offload"):
            other = "offload" if strategy == "hybrid" else "hybrid"
            drop = ("--moe-cpu-layers", "--moe-cpu-threads") if other == "offload" else ()
            t, t_args = attempt({"--moe-strategy": other}, drop)
            if keep_faster_decode(best, t):
                notes = [n for n in notes if n["flag"] not in ("--moe-strategy",) + drop]
                notes.append({"flag": "--moe-strategy", "value": other, "source": "measured",
                              "why": f"実測で決めました: CPU と転送の差が {ratio:.1f} 倍と小さかったので両方を起動して比べ、{other} で生成が {best['decode_tps']:.1f} → {t['decode_tps']:.1f} tok/s になりました。",
                              "why_en": f"Measured: the CPU was only {ratio:.1f}x the transfer, so both were started; {other} took generation {best['decode_tps']:.1f} -> {t['decode_tps']:.1f} tok/s."})
                best, best_args = t, t_args

        result["chosen"] = best["label"]
        return best_args, notes

    def _trial(self, label: str, model: str, port: int, args: list[str], change: dict | None = None) -> dict:
        try:
            return self._trial_run(label, model, port, args, change)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 -- e.g. a wider chunk that runs out of VRAM mid-prompt
            self._ev("trial", trial=label, step="measure", state="failed", message=str(exc)[:300])
            return {"label": label, "args": list(args), "change": change, "ok": False, "error": str(exc)[:300]}

    def _trial_run(self, label: str, model: str, port: int, args: list[str], change: dict | None) -> dict:
        out: dict = {"label": label, "args": list(args), "change": change, "ok": False}
        self._ev("trial", trial=label, step="load", state="start", args=args, change=change)
        t0 = self._clock()
        self.manager.switch(model, port, args)
        ready = self._wait_ready(label, port)
        out["load_s"] = round(self._clock() - t0, 1)
        if ready is not True:
            out["error"] = ready
            self._ev("trial", trial=label, step="load", state="failed", message=ready)
            return out
        self._ev("trial", trial=label, step="load", state="done", seconds=out["load_s"])
        rng = random.Random(0xBE7C + ord(label))
        model_id = ((self._http(port, "/v1/models", timeout=30) or {}).get("data") or [{}])[0].get("id") or "default"
        # warm-up, and how many tokens a word of this vocabulary is for this tokenizer
        probe = self._http(port, "/v1/completions", {"model": model_id, "prompt": _random_text(400, rng), "max_tokens": 8, "ignore_eos": True})
        per_word = max(0.5, (_prompt_tokens(probe)[0] or 400) / 400)

        cache = self._http(port, "/v1/cache/status", timeout=30) or {}
        geometry = cache.get("geometry") or {}
        kv_tokens = (geometry.get("num_pages") or 0) * (geometry.get("page_size") or 1)
        prompt_len = max(512, min(PREFILL_TOKENS, (kv_tokens or PREFILL_TOKENS) - DECODE_TOKENS - 64))

        self._check()
        self._ev("trial", trial=label, step="prefill", state="start", tokens=prompt_len)
        prefill = []
        for _ in range(2):
            t = self._clock()
            doc = self._prefill_once(label, port, {"model": model_id, "prompt": _random_text(int(prompt_len / per_word), rng),
                                                   "max_tokens": 1, "ignore_eos": True}, prompt_len)
            tokens, cached = _prompt_tokens(doc)
            tps = max(0, (tokens or prompt_len) - cached) / max(1e-6, self._clock() - t)
            prefill.append(round(tps, 1))
            self._ev("sample", id=f"{label}.prefill", value=round(tps, 1))
        out["prefill_tps"] = max(prefill)
        self._ev("trial", trial=label, step="prefill", state="done", value=out["prefill_tps"])

        self._check()
        self._ev("trial", trial=label, step="decode", state="start", tokens=DECODE_TOKENS)
        decode = [self._decode(label, port, model_id, rng) for _ in range(DECODE_RUNS)]
        decode = [d for d in decode if d]
        out["decode_tps"] = round(sorted(decode)[len(decode) // 2], 2) if decode else None
        self._ev("trial", trial=label, step="decode", state="done", value=out["decode_tps"])

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

    def _decode(self, label: str, port: int, model_id: str, rng: random.Random) -> float | None:
        body = {"model": model_id, "prompt": _random_text(100, rng), "max_tokens": DECODE_TOKENS, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}
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
                self._ev("sample", id=f"{label}.decode", value=round((chunks - 1) / (last - first), 2))
            if self._cancel.is_set():
                break
        if first is None or last is None or last <= first or chunks < 2:
            return None
        # a chunk can carry no text (a token that ends mid-character): the usage count is exact
        tokens = usage if usage and usage >= chunks else chunks
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
