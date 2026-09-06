"""The pre-Ampere prefill MoE path (chunked dequant + per-expert matmul) reproduces the
torch reference of the fused kernel's semantics: gather, silu-gated MLP, router weight, route
sum. The dequant kernel is GPU-only, so it is replaced by a table of plain weights here."""

from __future__ import annotations

import pytest
import torch

fm = pytest.importorskip("freetoken.moe.fused_nvfp4")

E, H, I, T, K = 40, 96, 32, 50, 4  # 40 experts -> two dequant chunks of 32 and 8; H != 2I on purpose


def _fake_weights(seed):
    g = torch.Generator().manual_seed(seed)
    w_gu = torch.randn(E, 2 * I, H, generator=g) * 0.05
    w_d = torch.randn(E, H, I, generator=g) * 0.05
    return w_gu, w_d


def _reference(x, w_gu, w_d, w, ids, on_input):
    out = torch.zeros(T, H)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            xe = x[t].float() * (w[t, k] if on_input else 1.0)
            h = xe @ w_gu[e].t()
            a = torch.nn.functional.silu(h[:I]) * h[I:]
            y = a @ w_d[e].t()
            out[t] += y * (1.0 if on_input else w[t, k])
    return out


@pytest.mark.parametrize("on_input", [False, True])
def test_scratch_prefill_matches_reference(monkeypatch, on_input):
    torch.manual_seed(0)
    w_gu, w_d = _fake_weights(1)

    def fake_dequant(packed, scale, glob, slots, out=None, *, dtype=torch.bfloat16):
        table = w_gu if packed.shape[2] == H // 2 else w_d  # gate_up packs H, down packs I
        return table[slots.long()].to(dtype)

    monkeypatch.setattr("freetoken.kernel.triton.nvfp4_dequant.dequant_nvfp4", fake_dequant)
    x = torch.randn(T, H).to(torch.bfloat16)
    logits = torch.randn(T, E)
    w, ids = torch.topk(torch.softmax(logits, -1), K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float()
    ids = ids.to(torch.int32)
    gu_packed = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8)
    d_packed = torch.zeros(E, H, I // 2, dtype=torch.uint8)
    dummy = torch.zeros(1)
    out = fm._fused_experts_nvfp4_scratch(
        x, gu_packed, dummy, dummy, d_packed, dummy, dummy, w, ids, E, on_input)
    ref = _reference(x, w_gu, w_d, w, ids, on_input)
    assert out.dtype == torch.bfloat16 and out.shape == (T, H)
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


def test_scratch_dispatch_env_override(monkeypatch):
    fm._scratch_moe_preferred.cache_clear()
    monkeypatch.setenv("FREETOKEN_NVFP4_MOE_SCRATCH", "1")
    assert fm._scratch_moe_preferred() is True
    fm._scratch_moe_preferred.cache_clear()
    monkeypatch.setenv("FREETOKEN_NVFP4_MOE_SCRATCH", "0")
    assert fm._scratch_moe_preferred() is False
    fm._scratch_moe_preferred.cache_clear()
