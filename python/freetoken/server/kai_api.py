"""The web console's view of the per-rank snapshots written by ``scheduler/webstats.py``.

``kai_block`` becomes ``/v1/stats``'s ``kai`` field (dashboard); ``/v1/kai/experts`` is the per-layer
detail the tuning page turns into suggestions. Both only read small JSON files, so they never touch
the scheduler, and a missing or stale rank shows up as missing rather than as zeros."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

from fastapi import FastAPI

from freetoken.webui.stats_path import stats_dir


_RANK_FILE = re.compile(r"rank\d+\.json")
_FREQ_FILE = re.compile(r"rank\d+\.experts\.json")


def read_expert_freq(port: int | None, window: str) -> list[dict] | None:
    """Per-rank routing counts (layers x experts) over the window, or None when not collected."""
    d = stats_dir(port)
    if not d or not os.path.isdir(d):
        return None
    out = []
    for name in sorted(os.listdir(d)):
        if not _FREQ_FILE.fullmatch(name):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        src = (doc.get("windows") or {}).get(window) or doc.get("cumulative") or {}
        if src.get("freq"):
            out.append({"rank": doc.get("rank"), "layer_range": doc.get("layer_range"),
                        "seconds": src.get("seconds"), "freq": src["freq"]})
    return sorted(out, key=lambda r: r.get("rank") or 0) or None


def read_ranks(port: int | None) -> list[dict]:
    d = stats_dir(port)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not _RANK_FILE.fullmatch(name):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda r: r.get("rank", 0))


def _window(rank: dict, prefer: str = "60") -> dict | None:
    w = rank.get("windows") or {}
    return w.get(prefer) or w.get("300")


def _meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                key = {"MemTotal": "mem_total", "MemAvailable": "mem_available",
                       "SwapTotal": "swap_total", "SwapFree": "swap_free"}.get(k)
                if key:
                    out[key] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def kai_block(state: Any) -> dict | None:
    config = getattr(state, "config", None)
    ranks = read_ranks(getattr(config, "server_port", None))
    if not ranks:
        return None
    now = time.time()
    active = miss = faults = secs = 0
    bank_bytes = 0
    new_tok = cached_tok = 0
    spec_steps = spec_acc = 0
    collect = False
    for r in ranks:
        w = _window(r)
        if not w:
            continue
        secs = max(secs, w.get("seconds", 0))
        active += sum(w.get("layer_active") or [])
        miss += sum(w.get("layer_miss") or [])
        faults += w.get("major_faults", 0)
        bank_bytes += w.get("bank_read_bytes", 0)
        new_tok += w.get("prefill_new_tokens", 0)
        cached_tok += w.get("prefill_cached_tokens", 0)
        spec_steps += w.get("spec_steps", 0)
        spec_acc += w.get("spec_accepted", 0)
        collect = collect or bool((r.get("moe") or {}).get("collect_stats"))
    moe = None
    if any(r.get("moe") for r in ranks):
        moe = {
            "collect_stats": collect,
            "gpu_hit_rate": (1 - miss / active) if (collect and active) else None,
            "major_faults_per_min": faults * 60 / secs if secs else None,
            "ssd_read_bytes_per_s": bank_bytes / secs if secs else None,
        }
    spec = None
    k = max((r.get("spec_k", 0) for r in ranks), default=0)
    if k:
        tps = spec_acc / spec_steps if spec_steps else None
        spec = {"k": k, "tokens_per_step": tps, "accept_rate": ((tps - 1) / k) if tps else None}
    gpus = []
    for r in ranks:
        g = r.get("gpu")
        if not g:
            continue
        used = g["total_bytes"] - g["free_bytes"]
        lr = r.get("layer_range")
        gpus.append({
            "rank": r.get("rank"), "index": g.get("index"), "name": g.get("name"),
            "total_bytes": g["total_bytes"], "used_bytes": used, "reserved_bytes": g["reserved_bytes"],
            "layers": [lr[0], lr[1] - 1] if lr else None,
            "age_s": round(now - r.get("time", now), 1),
        })
    return {
        "window_s": round(secs, 1),
        "moe": moe,
        "spec": spec,
        "prefill": {"chunk": ranks[0].get("prefill_chunk"), "auto": bool(getattr(config, "prefill_chunk_budget", None))},
        "prefix_reuse_rate": cached_tok / (new_tok + cached_tok) if (new_tok + cached_tok) else None,
        "kv_cache_dtype": getattr(config, "kv_cache_dtype", None),
        "gpus": gpus,
    }


def experts_doc(state: Any, window: str = "300", freq: bool = False) -> dict:
    config = getattr(state, "config", None)
    ranks = read_ranks(getattr(config, "server_port", None))
    layers = []
    per_rank = []
    for r in ranks:
        w = (r.get("windows") or {}).get(window) or {}
        cum = r.get("cumulative") or {}
        src = w if w.get("layer_active") else cum
        moe = r.get("moe") or {}
        lr = r.get("layer_range") or [0, 0]
        act, mis = src.get("layer_active") or [], src.get("layer_miss") or []
        calls, fet = src.get("layer_calls") or [], src.get("layer_fetched") or []
        n = len(act)
        mtp = bool(r.get("spec_k")) and n > 0 and r.get("rank") == r.get("size", 1) - 1
        for i in range(n):
            layers.append({
                "rank": r.get("rank"), "gpu": (r.get("gpu") or {}).get("index"),
                "local": i, "mtp": mtp and i == n - 1,
                "active": act[i], "miss": mis[i], "calls": calls[i] if i < len(calls) else None,
                "fetched": fet[i] if i < len(fet) else None,
            })
        per_rank.append({
            "rank": r.get("rank"), "gpu": r.get("gpu"), "layer_range": lr, "moe": moe,
            "window": {k: v for k, v in w.items() if not k.startswith("layer_")},
            # lifetime totals: a client timing one request diffs these (the windows slide under it)
            "counters": {k: v for k, v in cum.items() if not k.startswith("layer_")},
            "kv": r.get("kv"), "prefill_chunk": r.get("prefill_chunk"),
            "age_s": round(time.time() - r.get("time", time.time()), 1),
        })
    model = getattr(config, "model_config", None)
    return {
        "window": window,
        # only on request: 10k+ numbers the suggestions do not need
        "expert_freq": read_expert_freq(getattr(config, "server_port", None), window) if freq else None,
        "ranks": per_rank,
        "layers": layers,
        "host_memory": _meminfo(),
        "config": {
            "moe_cache_size": getattr(config, "moe_cache_size", None),
            "moe_bank_ram": getattr(config, "moe_bank_ram", None),
            "kv_cache_dtype": getattr(config, "kv_cache_dtype", None),
            "max_seq_len": getattr(config, "max_seq_len", None),
            "pp_size": getattr(getattr(config, "tp_info", None), "size", None),
            "spec_mtp": getattr(config, "spec_mtp", None),
            "num_experts": getattr(model, "num_experts", None) if model is not None else None,
            "num_experts_per_tok": getattr(model, "num_experts_per_tok", None) if model is not None else None,
        },
    }


def register_kai_routes(app: FastAPI, get_state: Callable[[], Any]) -> None:
    @app.get("/v1/kai/experts")
    async def kai_experts(window: str = "300", freq: bool = False):
        return experts_doc(get_state(), window if window in ("60", "300") else "300", freq)
