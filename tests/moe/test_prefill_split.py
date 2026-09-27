"""Split prefill: the experts a prefill forward routes to least are computed by the CPU executor
while only the rest are moved to the GPU, at once (OffloadMoELayer._prefill_split)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

moe = pytest.importorskip("freetoken.layers.moe")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ---------------------------------------------------------------------------------- planner
def test_the_least_routed_experts_go_to_the_cpu():
    counts = [0, 40, 1, 1, 2, 30, 0, 1]
    cpu, gpu = moe.plan_prefill_split(counts, ratio=4.0)
    assert cpu == [2, 3, 4, 7] and gpu == [1, 5]
    assert set(cpu).isdisjoint(gpu) and set(cpu) | set(gpu) == {e for e, c in enumerate(counts) if c}


def test_the_cut_is_where_both_sides_finish_together():
    counts = [1] * 8 + [10] * 8
    for ratio in (0.5, 2.0, 4.0, 8.0, 50.0):
        cpu, gpu = moe.plan_prefill_split(counts, ratio)
        cost = max(sum(counts[e] for e in cpu), ratio * len(gpu))
        # no other cut along the sorted order does better
        order = sorted(range(16), key=lambda e: (counts[e], e))
        for k in range(17):
            other = max(sum(counts[e] for e in order[:k]), ratio * (16 - k))
            assert cost <= other + 1e-9


def test_extremes():
    assert moe.plan_prefill_split([0, 0, 0], 4.0) == ([], [])
    # one row per expert: the CPU is cheaper for every one of them
    cpu, gpu = moe.plan_prefill_split([1] * 16, 4.0)
    assert len(cpu) >= 12
    # moving an expert costs almost nothing: the GPU takes them all
    assert moe.plan_prefill_split([5, 5, 5], 1e-6) == ([], [0, 1, 2])


# ---------------------------------------------------------------------------------- dispatch
def _cache(unpinned=(), **kw):
    base = dict(prefill_overlap=False, cpu_executor=object(), prefix_pinned_rows=None, bank_reader=None,
                is_unpinned_layer=lambda layer_id: layer_id in unpinned)
    base.update(kw)
    return SimpleNamespace(**base)


def test_when_the_split_applies(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PREFILL_SPLIT", raising=False)
    monkeypatch.delenv("FREETOKEN_PREFILL_SPLIT_MAX_TOKENS", raising=False)
    monkeypatch.setenv("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256")
    assert moe.prefill_split_max_tokens(_cache()) == moe.PREFILL_SPLIT_MAX_TOKENS
    assert moe.prefill_split_max_tokens(_cache(prefill_overlap=True)) == moe.PREFILL_SPLIT_MAX_TOKENS
    assert moe.prefill_split_max_tokens(_cache(cpu_executor=None)) == 0
    assert moe.prefill_split_max_tokens(_cache(prefix_pinned_rows=100)) == 0   # --moe-bank-ram
    monkeypatch.setenv("FREETOKEN_PREFILL_SPLIT_MAX_TOKENS", "512")
    assert moe.prefill_split_max_tokens(_cache(prefill_overlap=True)) == 512
    monkeypatch.setenv("FREETOKEN_PREFILL_SPLIT", "0")
    assert moe.prefill_split_max_tokens(_cache()) == 0


def _fake_layer(monkeypatch, cache, weight_on_input=False):
    layer = moe.OffloadMoELayer.__new__(moe.OffloadMoELayer)
    layer.layer_id, layer.num_experts, layer.apply_router_weight_on_input = 0, 4, weight_on_input
    layer.offload_cache = cache
    monkeypatch.setattr(layer, "_prefill_split", lambda *a: "split")
    monkeypatch.setattr(layer, "_prefill_on_cpu", lambda *a: "cpu")
    return layer


def _route(layer, rows):
    return layer._prefill_routed(torch.zeros(rows, 4), torch.zeros(rows, 2), torch.zeros(rows, 2, dtype=torch.int32))


def test_which_forwards_split(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PREFILL_SPLIT", raising=False)
    monkeypatch.setenv("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256")
    layer = _fake_layer(monkeypatch, _cache())
    assert _route(layer, 100) == "split"
    # a few rows: the split's fixed cost outweighs it, the CPU takes them as before
    assert _route(layer, moe.PREFILL_SPLIT_MIN_TOKENS - 1) == "cpu"
    # an unpinned layer can only stage whole layers: the old paths
    assert _route(_fake_layer(monkeypatch, _cache(unpinned=(0,))), 100) == "cpu"
    # weight 0 on the input does not zero a biased expert's output, so the GPU's dummy routes
    # would not vanish: such layers never split
    assert _route(_fake_layer(monkeypatch, _cache(), weight_on_input=True), 100) == "cpu"


# ---------------------------------------------------------------------------------- on a GPU
L, E, H, I, TOPK = 2, 16, 512, 256, 4


def _rig():
    """An NVFP4 (Triton) offload layer, its cache and a CPU executor over the same host banks --
    the pieces the engine wires for --moe-strategy hybrid."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    layer = moe.OffloadMoELayer(0, E, TOPK, H, I, quant_config=quant, prefix="model.layers.0.mlp.experts")
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    n = L * E
    flat = {
        "gate_up": torch.randint(0, 256, (n, 2 * I, H // 2), dtype=torch.uint8, generator=g),
        "gate_up_scale": (torch.rand(n, 2 * I, H // 16, generator=g) + 0.5).to(torch.float8_e4m3fn),
        "gate_up_global": torch.full((n, 2 * I), 0.02, dtype=torch.float16),
        "down": torch.randint(0, 256, (n, H, I // 2), dtype=torch.uint8, generator=g),
        "down_scale": (torch.rand(n, H, I // 16, generator=g) + 0.5).to(torch.float8_e4m3fn),
        "down_global": torch.full((n, H), 0.02, dtype=torch.float16),
    }
    per_layer = {k: list(v.pin_memory().split(E)) for k, v in flat.items()}
    pieces = ((l, 0, E, {k: per_layer[k][l] for k in per_layer}) for l in range(L))
    banks = build_expert_banks(layer.quant_method, L, pieces, device=dev)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=E + 8, device=dev,
                            quant_format=banks.quant_format, layout=banks.layout,
                            max_slots=layer.quant_method.slot_limit(), decode_target="hybrid")
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.reset()
    ex = CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=moe.CPU_PREFILL_PIECE, device=dev,
                        fmt=layer.quant_method.cpu_format)
    cache.set_cpu_executor(ex)
    layer.offload_cache = cache
    return layer, cache


def _skewed_routing(rows, seed):
    """Two hot experts on every row, the other routes spread thin: a mix the planner splits."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        rest = torch.randperm(E - 2, generator=g)[: TOPK - 2] + 2
        ids[r] = torch.cat([torch.tensor([0, 1]), rest]).to(torch.int32)
    w = torch.rand(rows, TOPK, generator=g)
    return ids.cuda(), (w / w.sum(1, keepdim=True)).cuda()


def _run(layer, x, w, ids, monkeypatch, **env):
    for k in ("FREETOKEN_PREFILL_SPLIT", "FREETOKEN_PREFILL_SPLIT_RATIO", "FREETOKEN_CPU_PREFILL_MAX_TOKENS",
              "FREETOKEN_PREFILL_SPLIT_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = layer._prefill_routed(x.clone(), w.clone(), ids.clone())
    torch.cuda.synchronize()
    return out.float()


def _close(a, b):
    tol = 0.05 * float(b.abs().max())
    torch.testing.assert_close(a, b, rtol=5e-2, atol=tol)


@cuda
@pytest.mark.parametrize("rows", [64, 100, 200])
def test_split_matches_the_whole_layer(rows, monkeypatch):
    layer, cache = _rig()
    x = (torch.randn(rows, H, generator=torch.Generator().manual_seed(rows)) / 4).to("cuda", torch.float16)
    ids, w = _skewed_routing(rows, rows)
    whole = _run(layer, x, w, ids, monkeypatch, FREETOKEN_PREFILL_SPLIT="0", FREETOKEN_CPU_PREFILL_MAX_TOKENS="0")
    counts = torch.bincount(ids.flatten().long(), minlength=E).tolist()
    cpu, gpu = moe.plan_prefill_split(counts, moe.PREFILL_SPLIT_RATIO)
    assert cpu and gpu, "the routing should give each side some experts"
    split = _run(layer, x, w, ids, monkeypatch)
    _close(split, whole)
    all_cpu = _run(layer, x, w, ids, monkeypatch, FREETOKEN_PREFILL_SPLIT_RATIO="1e9")
    all_gpu = _run(layer, x, w, ids, monkeypatch, FREETOKEN_PREFILL_SPLIT_RATIO="1e-9")
    _close(all_cpu, whole)
    torch.testing.assert_close(all_gpu, whole)          # the same GPU kernel on the same bytes


@cuda
def test_after_a_split_decode_does_not_hit_what_was_not_moved(monkeypatch):
    layer, cache = _rig()
    rows = 70
    x = (torch.randn(rows, H) / 4).to("cuda", torch.float16)
    ids, w = _skewed_routing(rows, 3)
    _run(layer, x, w, ids, monkeypatch)
    counts = torch.bincount(ids.flatten().long(), minlength=E).tolist()
    cpu, gpu = moe.plan_prefill_split(counts, moe.PREFILL_SPLIT_RATIO)
    slots = cache.slot_for_id[0].tolist()
    assert all(slots[e] == e for e in gpu)
    assert all(slots[e] == -1 for e in range(E) if e not in gpu)
    # the moved experts' bytes are the layer's own
    src = cache.bank_sources[cache.bank_schema[0]][0]
    for e in gpu:
        assert torch.equal(cache.bank_caches[cache.bank_schema[0]][e].cpu(), src[e])


@cuda
def test_only_the_given_experts_are_moved():
    layer, cache = _rig()
    for _, c in cache.banks:
        c.zero_()
    moved = [1, 4, 5, 11]
    cache.materialize_experts(0, torch.tensor(moved, dtype=torch.int32, device="cuda"), moved)
    torch.cuda.synchronize()
    name = cache.bank_schema[0]
    src, dst = cache.bank_sources[name][0], cache.bank_caches[name]
    for e in range(E):
        if e in moved:
            assert torch.equal(dst[e].cpu(), src[e])
        else:
            assert not dst[e].any(), f"expert {e} was not asked for"
    assert cache.slot_for_id[0].tolist() == [e if e in moved else -1 for e in range(E)]
