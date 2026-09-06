"""Short prefill extends go to the CPU executor in fixed-size pieces (tail padded with skipped
routes) and reassemble in order; long extends and disabled mode keep the streaming path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

moe = pytest.importorskip("freetoken.layers.moe")


class _FakeExecutor:
    def __init__(self, piece):
        self.max_tokens = piece
        self.calls = []

    def decode(self, layer_id, x, w, ids):
        self.calls.append((layer_id, x.shape[0], ids.min().item()))
        # a recognisable function of the inputs: row sum of x times the first route weight
        return (x.float().sum(dim=1, keepdim=True) * w[:, :1]).expand(-1, x.shape[1]).to(x.dtype)


def _layer(executor, overlap=False):
    layer = moe.OffloadMoELayer.__new__(moe.OffloadMoELayer)
    layer.layer_id = 7
    layer.num_experts = 16
    layer.offload_cache = SimpleNamespace(prefill_overlap=overlap, cpu_executor=executor)
    return layer


def test_short_extend_is_computed_in_pieces(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256")
    ex = _FakeExecutor(piece=moe.CPU_PREFILL_PIECE)
    layer = _layer(ex)
    T, H, K = 150, 8, 2
    x = torch.randn(T, H, dtype=torch.bfloat16)
    w = torch.rand(T, K)
    ids = torch.randint(0, 16, (T, K), dtype=torch.int32)
    out = layer._prefill_on_cpu(layer.offload_cache, x, w, ids)
    ref = (x.float().sum(dim=1, keepdim=True) * w[:, :1]).expand(-1, H).to(x.dtype)
    assert out.shape == (T, H) and torch.equal(out, ref)
    # 64 + 64 + 22 rows: three calls of the one batch shape, the last padded with -1 routes
    assert [c[1] for c in ex.calls] == [64, 64, 64]
    assert ex.calls[-1][2] == -1 and ex.calls[0][2] >= 0


def test_dispatch_threshold_and_disable(monkeypatch):
    ex = _FakeExecutor(piece=64)
    layer = _layer(ex)
    monkeypatch.setattr(layer, "_prefill_on_cpu", lambda *a: "cpu")
    monkeypatch.setattr(layer, "_expert_gemm", lambda *a, **k: "gpu")
    layer.offload_cache.materialize_layer = lambda lid: None
    layer.offload_cache.copy_missing = lambda: None
    layer.offload_cache.bank_views = lambda n: ()
    layer.offload_cache.alphas_for_layer = lambda lid: None
    x = torch.zeros(40, 8)
    monkeypatch.setenv("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256")
    assert layer._prefill_routed(x, torch.zeros(40, 2), torch.zeros(40, 2, dtype=torch.int32)) == "cpu"
    assert layer._prefill_routed(torch.zeros(300, 8), torch.zeros(300, 2), torch.zeros(300, 2, dtype=torch.int32)) == "gpu"
    monkeypatch.setenv("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "0")
    assert layer._prefill_routed(x, torch.zeros(40, 2), torch.zeros(40, 2, dtype=torch.int32)) == "gpu"
