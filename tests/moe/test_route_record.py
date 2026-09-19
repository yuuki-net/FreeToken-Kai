"""--moe-collect-stats routing record: decode_freq, route_steps and the routing ring.

``record_routes`` runs once per decode layer inside the captured graph. The CPU mirror is what
the tests below pin; the GPU kernel is checked against it (and, on a GPU, under a captured graph
replayed with new ids -- the property the old torch scatter was documented to lack).
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.moe.offload_kernels import record_routes


def _cache(num_layers=3, num_experts=40, ring_steps=8, device="cpu"):
    words = (num_experts + 31) // 32
    return SimpleNamespace(
        num_layers=num_layers, num_experts=num_experts, route_ring_steps=ring_steps, route_words=words,
        decode_freq=torch.zeros(num_layers, num_experts, dtype=torch.int64, device=device),
        route_steps=torch.zeros(num_layers, dtype=torch.int64, device=device),
        route_ring=torch.zeros(num_layers, ring_steps, words, dtype=torch.int32, device=device),
    )


def _unpack(row: torch.Tensor, num_experts: int) -> set[int]:
    bits = np.unpackbits(row.cpu().numpy().astype("<i4").view(np.uint8), bitorder="little")[:num_experts]
    return set(int(e) for e in np.flatnonzero(bits))


def test_counts_repeats_and_sets_one_bit_per_expert():
    c = _cache()
    ids = torch.tensor([[3, 39, 3], [0, 31, 32]], dtype=torch.int32)
    record_routes(c, 1, ids)
    assert c.decode_freq[1, 3] == 2 and c.decode_freq[1, 39] == 1 and c.decode_freq[1, 31] == 1
    assert int(c.decode_freq.sum()) == 6 and int(c.decode_freq[0].sum()) == 0
    assert c.route_steps.tolist() == [0, 1, 0]
    assert _unpack(c.route_ring[1, 0], 40) == {0, 3, 31, 32, 39}  # 31 is bit 31: a negative int32


def test_ring_wraps_at_route_steps_mod_ring():
    c = _cache(ring_steps=4)
    for step in range(6):
        record_routes(c, 0, torch.tensor([step], dtype=torch.int32))
    assert int(c.route_steps[0]) == 6
    # slots 0 and 1 were overwritten by steps 4 and 5
    assert [_unpack(c.route_ring[0, p], 40) for p in range(4)] == [{4}, {5}, {2}, {3}]


def test_negative_ids_are_skipped():
    c = _cache()
    record_routes(c, 2, torch.tensor([-1, 5, -1], dtype=torch.int32))
    assert int(c.decode_freq.sum()) == 1 and _unpack(c.route_ring[2, 0], 40) == {5}


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_cuda
@pytest.mark.parametrize("num_experts", [8, 64, 256, 513])
def test_kernel_matches_the_cpu_mirror(num_experts):
    rng = random.Random(num_experts)
    cpu, gpu = _cache(num_experts=num_experts), _cache(num_experts=num_experts, device="cuda")
    for step in range(20):
        layer = step % 3
        ids = torch.tensor([rng.randrange(num_experts) for _ in range(rng.choice([1, 8, 30]))], dtype=torch.int32)
        record_routes(cpu, layer, ids)
        record_routes(gpu, layer, ids.cuda())
    torch.cuda.synchronize()
    assert torch.equal(cpu.decode_freq, gpu.decode_freq.cpu())
    assert torch.equal(cpu.route_steps, gpu.route_steps.cpu())
    assert torch.equal(cpu.route_ring, gpu.route_ring.cpu())


@needs_cuda
def test_captured_graph_records_the_replayed_routing():
    c = _cache(device="cuda")
    ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    record_routes(c, 0, ids)  # warm-up (compiles the kernel outside the capture)
    torch.cuda.synchronize()
    c.decode_freq.zero_(); c.route_steps.zero_(); c.route_ring.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        record_routes(c, 0, ids)
    c.decode_freq.zero_(); c.route_steps.zero_(); c.route_ring.zero_()  # capture does not run it
    for step in range(3):
        ids.copy_(torch.tensor([step, step, 10 + step, 39], dtype=torch.int32))
        g.replay()
    torch.cuda.synchronize()
    freq = c.decode_freq[0].cpu()
    assert freq[0] == 2 and freq[1] == 2 and freq[2] == 2 and freq[39] == 3 and int(freq.sum()) == 12
    assert int(c.route_steps[0]) == 3
    assert [_unpack(c.route_ring[0, p], 40) for p in range(3)] == [{0, 10, 39}, {1, 11, 39}, {2, 12, 39}]
