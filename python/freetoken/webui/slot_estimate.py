"""How the GPU expert cache would do at other sizes, from the routing it actually saw.

Each rank's cache records the last decode steps' sets of active experts per MoE layer
(``OffloadMoeCache.route_ring``, written out as ``rank<N>.routes.npz`` by scheduler/webstats.py).
Replaying them in decode order -- step by step, layer by layer, the way the cache is asked -- and
taking every access's LRU stack distance (how many other experts were touched since this one last
was) gives the hit rate of an LRU cache of every size at once: an access hits a cache of C slots
exactly when its distance is below C. The cache is one LRU over (layer, expert) shared by every
GPU layer of the rank, so the replay is too; CPU layers never take a slot and are left out.

What this is and is not:

* It assumes every miss is brought into the cache. Plain offload does that; the hybrid backend
  fetches only part of each step's misses (the rest run on the CPU), so its measured hit rate
  sits below this curve. The console shows both.
* The first quarter of the replayed steps only warms the stack (every expert is a miss the first
  time it is seen, at any size) and is not counted.
* A few hundred steps of one workload: other prompts route differently.

Torch-free (numpy only): it runs in the HTTP server, never in the scheduler.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

WARM_FRACTION = 0.25
MAX_ACCESSES = 400_000


def load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _aligned_layers(steps: np.ndarray, gpu_layers: np.ndarray) -> list[int]:
    """GPU layers stepped as often as most of them: the draft head's MoE layer (--spec-mtp) runs
    a different number of times per token and has no place in the per-step order."""
    counts = [int(steps[i]) for i in range(len(steps)) if gpu_layers[i] and steps[i] > 0]
    if not counts:
        return []
    mode = Counter(counts).most_common(1)[0][0]
    return [i for i in range(len(steps)) if gpu_layers[i] and int(steps[i]) == mode]


def accesses(doc: dict) -> tuple[list[int], int, int]:
    """The replayed access keys (layer * E + expert) in decode order, the number of steps, and
    the index of the first counted access (after the warm-up)."""
    ring, steps, gpu = doc["ring"], doc["steps"], doc["gpu_layers"]
    num_experts = int(doc["num_experts"])
    layers = _aligned_layers(steps, gpu)
    if not layers:
        return [], 0, 0
    ring_steps = ring.shape[1]
    total = int(steps[layers[0]])
    n = min(total, ring_steps)
    per_step = max(1, len(layers) * 8)
    n = min(n, max(1, MAX_ACCESSES // per_step))
    keys: list[int] = []
    warm_at = 0
    warm_steps = int(n * WARM_FRACTION)
    for j in range(n):
        if j == warm_steps:
            warm_at = len(keys)
        pos = (total - n + j) % ring_steps
        for layer in layers:
            words = ring[layer, pos].astype("<i4").view(np.uint8)
            bits = np.unpackbits(words, bitorder="little")[:num_experts]
            base = layer * num_experts
            keys.extend(base + int(e) for e in np.flatnonzero(bits))
    return keys, n, warm_at


def stack_distances(keys: list[int]) -> list[int]:
    """LRU stack distance of every access (-1 = first access): the number of distinct other keys
    touched since the previous access of the same key. Fenwick tree over access times."""
    n = len(keys)
    tree = [0] * (n + 1)

    def add(i: int, v: int) -> None:
        i += 1
        while i <= n:
            tree[i] += v
            i += i & -i

    def prefix(i: int) -> int:  # sum over [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    last: dict[int, int] = {}
    out = [-1] * n
    live = 0
    for t, k in enumerate(keys):
        p = last.get(k)
        if p is not None:
            out[t] = live - prefix(p + 1)  # markers after p = keys last touched since then
            add(p, -1)
            live -= 1
        add(t, 1)
        live += 1
        last[k] = t
    return out


def estimate(doc: dict, points: int = 24) -> dict | None:
    keys, steps, warm_at = accesses(doc)
    counted = len(keys) - warm_at
    if steps < 8 or counted <= 0:
        return None
    dist = stack_distances(keys)[warm_at:]
    num_experts = int(doc["num_experts"])
    layers = _aligned_layers(doc["steps"], doc["gpu_layers"])
    capacity = len(layers) * num_experts  # every expert of every GPU layer resident
    hist = Counter(d for d in dist if d >= 0)
    ordered = sorted(hist.items())

    def hit_rate(slots: int) -> float:
        return sum(c for d, c in ordered if d < slots) / counted

    current = int(doc["cache_size"])
    sizes = sorted({max(1, round(capacity * i / points)) for i in range(1, points + 1)} | {current})
    return {
        "steps": steps,
        "accesses": counted,
        "gpu_layers": len(layers),
        "num_experts": num_experts,
        "capacity": capacity,
        "cache_size": current,
        "decode_target": str(doc.get("decode_target", "")),
        "hit_at_current": hit_rate(current),
        "curve": [{"slots": s, "hit": hit_rate(s)} for s in sizes],
        "time": float(doc.get("time", 0.0)),
    }


_cache: dict[str, tuple[float, dict | None]] = {}


def estimate_file(path: str) -> dict | None:
    """``estimate`` of one rank's file, recomputed only when the file changed."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _cache.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        result = estimate(load(path))
    except (OSError, ValueError, KeyError):
        result = None
    _cache[path] = (mtime, result)
    return result
