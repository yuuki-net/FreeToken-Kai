"""Per-rank snapshot for the web console (``/ui/``): one small JSON file every couple of seconds.

Why a file instead of the reply stamps ``/v1/stats`` rides on: only rank 0 talks to the tokenizer,
so under ``--pp-size`` the other GPUs' numbers had no way to reach the frontend. Every rank writes
``<dir>/rank<N>.json`` and the frontend reads them all (``server/kai_api.py``).

Nothing here may stall decode. The MoE counters live on the device (accumulated inside the
captured decode graph when ``--moe-collect-stats`` is on), so they are copied into a pinned host
buffer with ``non_blocking=True`` behind a CUDA event and read back on a later tick once the event
has fired. Host-side numbers (VRAM, page faults, bank reads, token totals) are plain reads.

The file also carries windowed deltas (last 60 s / 300 s / since start) of every counter, so the
page does not have to stay open for rates to exist.

The cache also counts how often each expert is routed (``decode_freq``, layers x experts). That
goes to ``rank<N>.experts.json`` every 10 s instead: it is 10k+ numbers, too big to rewrite every
2 s or to keep 150 copies of for the windows. The routing ring (the last steps' sets of active
experts, OffloadMoeCache.route_ring) goes out on the same beat as ``rank<N>.routes.npz``, for the
console's cache-size estimate (server/kai_api.py replays it)."""

from __future__ import annotations

import json
import os
import resource
import time
from collections import deque
from typing import Any

import torch

from freetoken.utils import init_logger
from freetoken.webui.stats_path import stats_dir

logger = init_logger(__name__)

INTERVAL_S = 2.0
WINDOWS_S = (60, 300)
FREQ_INTERVAL_S = 10.0


class WebStatsPublisher:
    def __init__(self, scheduler: Any) -> None:
        self.s = scheduler
        cfg = scheduler.config
        self.rank, self.size = cfg.tp_info.rank, cfg.tp_info.size
        self.dir = stats_dir(getattr(cfg, "server_port", None))
        self.path = os.path.join(self.dir, f"rank{self.rank}.json") if self.dir else None
        self.freq_path = os.path.join(self.dir, f"rank{self.rank}.experts.json") if self.dir else None
        self.routes_path = os.path.join(self.dir, f"rank{self.rank}.routes.npz") if self.dir else None
        self._freq_history: deque[tuple[float, torch.Tensor]] = deque()
        self._freq_last = -FREQ_INTERVAL_S
        self.started = time.time()
        self._last = 0.0
        self._pending: tuple[Any, dict, dict] | None = None  # (event, host_bufs, host_snapshot)
        self._history: deque[tuple[float, dict]] = deque()
        self._host_bufs: dict[str, torch.Tensor] = {}
        self._failed = False
        if self.path:
            try:
                os.makedirs(self.dir, exist_ok=True)
                for stale in (self.path, self.freq_path, self.routes_path):
                    if os.path.exists(stale):
                        os.remove(stale)  # a previous run's file must not pass for this one
            except OSError as exc:
                logger.warning("web console stats disabled: %s", exc)
                self.path = None

    # ------------------------------------------------------------------ tick
    def tick(self, idle: bool = False) -> None:
        if self.path is None or self._failed:
            return
        try:
            if self._pending is not None:
                self._maybe_finish(wait=idle)
            now = time.monotonic()
            if self._pending is None and (idle or now - self._last >= INTERVAL_S):
                self._last = now
                self._launch()
                # idle: nothing is queued on the device, so waiting for the copy costs nothing
                # and the file then holds the final numbers for as long as the server stays idle
                self._maybe_finish(wait=idle)
        except Exception as exc:  # noqa: BLE001 -- a console must never take the engine down
            self._failed = True
            logger.warning("web console stats stopped: %s", exc)

    def _maybe_finish(self, wait: bool) -> None:
        ev = self._pending[0]
        if ev is not None and wait:
            ev.synchronize()
        if ev is None or wait or ev.query():
            self._finish()

    # ------------------------------------------------------------------ launch: queue the device copy
    def _launch(self) -> None:
        eng = self.s.engine
        cache = getattr(eng, "moe_offload_cache", None)
        dev_tensors: dict[str, torch.Tensor] = {}
        if cache is not None and getattr(cache, "collect_stats", False):
            if cache.decode_target == "hybrid":
                dev_tensors = {
                    "active": cache.stat_active_layer, "miss": cache.stat_missing_layer,
                    "calls": cache.stat_steps_layer, "fetched": cache.stat_fetched_layer,
                }
            else:
                from flashlib.kernels.slot_cache import Stat

                dev_tensors = {
                    "active": cache.lru_stats[:, Stat.ACTIVE], "miss": cache.lru_stats[:, Stat.MISS],
                    "calls": cache.lru_stats[:, Stat.CALLS],
                }
        now = time.monotonic()
        if cache is not None and getattr(cache, "collect_decode_freq", False) and now - self._freq_last >= FREQ_INTERVAL_S:
            self._freq_last = now
            dev_tensors["expert_freq"] = cache.decode_freq
            if getattr(cache, "route_ring", None) is not None:
                dev_tensors["route_ring"] = cache.route_ring
                dev_tensors["route_steps"] = cache.route_steps
        ev = None
        if dev_tensors:
            with torch.cuda.stream(eng.stream):
                for k, t in dev_tensors.items():
                    buf = self._host_bufs.get(k)
                    if buf is None or buf.shape != t.shape:
                        buf = self._host_bufs[k] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                    buf.copy_(t, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
        self._pending = (ev, {k: self._host_bufs[k] for k in dev_tensors}, self._host_snapshot(cache))

    def _gpu_name(self, device) -> str:
        if not hasattr(self, "_name"):
            self._name = torch.cuda.get_device_name(device)
        return self._name

    def _host_snapshot(self, cache: Any) -> dict:
        s, eng, cfg = self.s, self.s.engine, self.s.config
        snap: dict[str, Any] = {"t_mono": time.monotonic()}
        if eng.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(eng.device)
            snap["gpu"] = {
                "index": eng.device.index if eng.device.index is not None else torch.cuda.current_device(),
                "name": self._gpu_name(eng.device),
                "total_bytes": int(total), "free_bytes": int(free),
                "reserved_bytes": int(torch.cuda.memory_reserved(eng.device)),
            }
        rep = s.status_reporter
        counters = {
            "major_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt,
            "prefill_new_tokens": rep.total_prefill_new_tokens,
            "prefill_cached_tokens": rep.total_prefill_cached_tokens,
            "decode_tokens": rep.total_decode_tokens,
            "spec_steps": rep.total_spec_steps,
            "spec_accepted": rep.total_spec_accepted,
        }
        reader = getattr(cache, "bank_reader", None) if cache is not None else None
        if reader is not None:
            counters["bank_read_bytes"] = int(getattr(reader, "bytes_read", 0))
            counters["bank_cached_bytes"] = int(getattr(reader, "bytes_cached", 0))
        snap["counters"] = counters
        try:
            used, total = s._kv_usage_pages()
            snap["kv"] = {"used_pages": used, "total_pages": total, "page_size": cfg.page_size}
        except Exception:  # noqa: BLE001
            pass
        snap["pools"] = _pool_bytes(eng)
        snap["prefill_chunk"] = getattr(eng, "_prefill_chunk_logged", None) or getattr(s, "prefill_budget", None)
        sampler = getattr(eng, "_decode_sampler", None)
        if sampler is not None:
            snap["decode_sample"] = sampler.snapshot()
        if cache is not None:
            snap["moe_static"] = {
                "decode_target": cache.decode_target,
                "cache_size": int(getattr(cache, "cache_size", 0) or 0),
                "num_layers": int(cache.num_layers),
                "num_experts": int(cache.num_experts),
                "collect_stats": bool(getattr(cache, "collect_stats", False)),
            }
        return snap

    # ------------------------------------------------------------------ finish: host bufs are valid
    def _finish(self) -> None:
        _ev, bufs, snap = self._pending
        self._pending = None
        bufs = dict(bufs)
        freq = bufs.pop("expert_freq", None)
        if freq is not None:
            self._write_freq(snap["t_mono"], freq.clone())
        ring, ring_steps = bufs.pop("route_ring", None), bufs.pop("route_steps", None)
        if ring is not None and ring_steps is not None:
            self._write_routes(ring, ring_steps)
        cum = dict(snap["counters"])
        for k, b in bufs.items():
            cum["layer_" + k] = [int(x) for x in b.tolist()]
        now = snap["t_mono"]
        self._history.append((now, cum))
        while self._history and now - self._history[0][0] > max(WINDOWS_S) + 2 * INTERVAL_S:
            self._history.popleft()

        windows = {}
        for w in WINDOWS_S:
            base = next((h for h in self._history if now - h[0] <= w + INTERVAL_S / 2), None)
            if base is not None and base[0] < now:
                windows[str(w)] = {"seconds": now - base[0], **_delta(cum, base[1])}

        cfg = self.s.config
        doc = {
            "v": 1, "rank": self.rank, "size": self.size, "pid": os.getpid(),
            "time": time.time(), "started": self.started,
            "layer_range": list(getattr(cfg, "pp_layer_range", None) or []) or None,
            "spec_k": int(getattr(cfg, "spec_mtp", 0) or 0),
            "gpu": snap.get("gpu"), "pools": snap.get("pools"), "kv": snap.get("kv"), "moe": snap.get("moe_static"),
            "prefill_chunk": snap.get("prefill_chunk"),
            "decode_sample": snap.get("decode_sample"),
            "cumulative": cum, "windows": windows,
        }
        _write_json(self.path, doc)

    def _write_freq(self, now: float, freq: torch.Tensor) -> None:
        self._freq_history.append((now, freq))
        while self._freq_history and now - self._freq_history[0][0] > max(WINDOWS_S) + 2 * FREQ_INTERVAL_S:
            self._freq_history.popleft()
        windows = {}
        for w in WINDOWS_S:
            base = next((h for h in self._freq_history if now - h[0] <= w + FREQ_INTERVAL_S / 2), None)
            if base is not None and base[0] < now:
                windows[str(w)] = {"seconds": now - base[0], "freq": (freq - base[1]).tolist()}
        cfg = self.s.config
        _write_json(self.freq_path, {
            "v": 1, "rank": self.rank, "time": time.time(),
            "layer_range": list(getattr(cfg, "pp_layer_range", None) or []) or None,
            "cumulative": {"seconds": now - self._freq_history[0][0], "freq": freq.tolist()},
            "windows": windows,
        })


    def _write_routes(self, ring: torch.Tensor, steps: torch.Tensor) -> None:
        """The routing ring as it stands, with what a replay needs to read it: which layers the
        GPU cache serves (CPU layers never take a slot) and the cache's size."""
        import numpy as np

        cache = self.s.engine.moe_offload_cache
        gpu_layers = [not cache.is_cpu_layer(i) for i in range(cache.num_layers)]
        tmp = f"{self.routes_path}.tmp.npz"
        np.savez(
            tmp,
            ring=ring.numpy(), steps=steps.numpy(), gpu_layers=np.array(gpu_layers, dtype=bool),
            num_experts=np.int64(cache.num_experts), cache_size=np.int64(cache.cache_size),
            decode_target=np.array(cache.decode_target), time=np.float64(time.time()),
        )
        os.replace(tmp, self.routes_path)


def _pool_bytes(engine: Any) -> dict | None:
    """VRAM of this rank's own cache pools, from its allocated tensors (the same figures the
    rebuild log prints). None when they cannot be read; never raises."""
    try:
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        pools, unit = compute_cache_pools(engine), compute_cache_unit_bytes(engine)
        return {
            "kv": pools["num_pages"] * pools["page_size"] * unit["kv_bytes_per_token"]
            + pools["num_swa_pages"] * pools["swa_page_size"] * unit["swa_bytes_per_token"],
            "moe": pools["moe_cache_size"] * unit["moe_bytes_per_expert"],
            "mamba": pools["num_mamba_slots"] * unit["mamba_bytes_per_slot"],
        }
    except Exception:  # noqa: BLE001 -- the dashboard falls back to rank 0's geometry
        return None


def _write_json(path: str, doc: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _delta(cur: dict, base: dict) -> dict:
    out = {}
    for k, v in cur.items():
        b = base.get(k)
        if isinstance(v, list) and isinstance(b, list) and len(b) == len(v):
            out[k] = [x - y for x, y in zip(v, b)]
        elif isinstance(v, (int, float)) and isinstance(b, (int, float)):
            out[k] = v - b
    return out
