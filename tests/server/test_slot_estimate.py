"""webui/slot_estimate.py: replaying the routing ring into an LRU hit-rate curve.

The stack-distance shortcut must give exactly what an LRU of each size would have done, so it is
checked against a plain LRU simulation on the same access order.
"""
from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np

from freetoken.webui import slot_estimate as se


def _lru_hits(keys, slots, warm_at):
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for t, k in enumerate(keys):
        if k in cache:
            cache.move_to_end(k)
            hits += t >= warm_at
        else:
            cache[k] = None
            if len(cache) > slots:
                cache.popitem(last=False)
    return hits / (len(keys) - warm_at)


def _pack(active: set[int], words: int) -> np.ndarray:
    bits = np.zeros(words * 32, dtype=np.uint8)
    bits[list(active)] = 1
    return np.packbits(bits, bitorder="little").view("<i4")


def _doc(steps=200, layers=4, experts=64, ring=128, cpu_layers=(), skew=1.2, seed=0, mtp_layer=None):
    rng = random.Random(seed)
    words = (experts + 31) // 32
    ring_arr = np.zeros((layers, ring, words), dtype=np.int32)
    step_counts = np.zeros(layers, dtype=np.int64)
    weights = [1 / (i + 1) ** skew for i in range(experts)]
    for s in range(steps):
        for l in range(layers):
            reps = 3 if l == mtp_layer else 1
            for _ in range(reps):
                active = set(rng.choices(range(experts), weights=weights, k=6))
                ring_arr[l, step_counts[l] % ring] = _pack(active, words)
                step_counts[l] += 1
    return {
        "ring": ring_arr, "steps": step_counts,
        "gpu_layers": np.array([l not in cpu_layers for l in range(layers)]),
        "num_experts": np.int64(experts), "cache_size": np.int64(40),
        "decode_target": np.array("gpu"), "time": np.float64(0),
    }


def test_stack_distances_match_an_lru_simulation():
    rng = random.Random(1)
    keys = [rng.choice(range(30)) for _ in range(2000)]
    dist = se.stack_distances(keys)
    for slots in (1, 3, 8, 15, 29, 30):
        est = sum(1 for d in dist[500:] if 0 <= d < slots) / (len(keys) - 500)
        assert abs(est - _lru_hits(keys, slots, 500)) < 1e-12


def test_estimate_replays_in_decode_order_and_matches_lru():
    doc = _doc()
    keys, steps, warm_at = se.accesses(doc)
    assert steps == 128  # the ring holds the last 128 of 200 steps
    est = se.estimate(doc)
    for p in est["curve"]:
        assert abs(p["hit"] - _lru_hits(keys, p["slots"], warm_at)) < 1e-12
    assert est["capacity"] == 4 * 64 and est["curve"][-1]["slots"] == est["capacity"]
    hits = [p["hit"] for p in est["curve"]]
    assert hits == sorted(hits)  # LRU has no Belady anomaly


def test_the_oldest_step_in_the_ring_comes_first():
    doc = _doc(steps=10, layers=1, ring=4)
    keys, steps, _ = se.accesses(doc)
    first = set(np.flatnonzero(np.unpackbits(doc["ring"][0, 10 % 4].view(np.uint8), bitorder="little")))
    assert steps == 4 and set(keys[: len(first)]) == first


def test_cpu_layers_and_the_draft_head_are_left_out():
    doc = _doc(layers=5, cpu_layers=(0,), mtp_layer=4)
    est = se.estimate(doc)
    assert est["gpu_layers"] == 3 and est["capacity"] == 3 * 64
    keys, _, _ = se.accesses(doc)
    assert {k // 64 for k in keys} == {1, 2, 3}


def test_too_little_routing_gives_nothing():
    assert se.estimate(_doc(steps=3)) is None


def test_estimate_file_reads_the_npz_and_caches_by_mtime(tmp_path):
    p = tmp_path / "rank0.routes.npz"
    np.savez(p, **_doc())
    a = se.estimate_file(str(p))
    assert a is not None and se.estimate_file(str(p)) is a
    assert se.estimate_file(str(tmp_path / "missing.npz")) is None
