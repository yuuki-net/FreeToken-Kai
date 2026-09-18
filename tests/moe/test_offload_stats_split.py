"""--moe-stats-out on plain offload: where the misses went.

``fetched`` / ``cpu`` split the misses between the PCIe fetch and the CPU executor, which only
the hybrid strategy has. On plain offload every miss is fetched, but the split read the hybrid
counter (never advanced there, so 0) and reported every miss as computed on the CPU -- a fork
running Kai concluded from it that offload computed its misses on the CPU. CPU-only.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.moe.offload_cache import OffloadMoeCache, Stat


def _cache(target: str):
    rows = torch.zeros(2, 3, dtype=torch.int64)
    rows[:, Stat.ACTIVE], rows[:, Stat.MISS], rows[:, Stat.CALLS] = 40, 10, 4   # per layer
    return SimpleNamespace(
        decode_target=target, num_layers=2, lru_stats=rows,
        stat_fetched=torch.tensor(0), stat_fetched_layer=torch.zeros(2, dtype=torch.int64),
        prefill_hit_rows=0, prefill_total_rows=0,
    )


def test_plain_offload_fetches_every_miss():
    s = OffloadMoeCache.decode_miss_stats(_cache("gpu"))
    assert s["missing_per_layer"] == 2.5
    assert s["fetched_per_layer"] == 2.5 and s["cpu_per_layer"] == 0.0 and s["fetch_rate"] == 1.0


def test_plain_offload_per_layer_too():
    per = OffloadMoeCache.decode_miss_stats_per_layer(_cache("gpu"))["per_layer"]
    assert [p["fetched_per_step"] for p in per] == [2.5, 2.5]
