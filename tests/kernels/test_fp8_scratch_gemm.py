"""The pre-Ampere fp8 W8A16 path: dequant into a scratch + plain matmul must match the
per-row-scaled reference, and the dispatch must honour the arch default and the override."""

from __future__ import annotations

import pytest
import torch

fp8mod = pytest.importorskip("freetoken.kernel.triton.fp8_pertensor_linear")


def _quant(w):
    s = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w.float() / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_scratch_matches_reference(dtype):
    torch.manual_seed(0)
    w = torch.randn(96, 128) * 0.02
    q, s = _quant(w)
    a = torch.randn(7, 128, dtype=dtype)
    out = fp8mod._gemm_scratch(a, q, s, dtype)
    ref = a.float() @ (q.float() * s[:, None]).t()
    assert out.dtype == dtype and out.shape == (7, 96)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_scratch_dispatch_follows_arch_and_override(monkeypatch):
    fp8mod._scratch_gemm_preferred.cache_clear()
    monkeypatch.setattr("freetoken.utils.is_pre_ampere", lambda: True)
    monkeypatch.delenv("FREETOKEN_FP8_SCRATCH_GEMM", raising=False)
    assert fp8mod._scratch_gemm_preferred() is True
    fp8mod._scratch_gemm_preferred.cache_clear()
    monkeypatch.setattr("freetoken.utils.is_pre_ampere", lambda: False)
    assert fp8mod._scratch_gemm_preferred() is False
    fp8mod._scratch_gemm_preferred.cache_clear()
    monkeypatch.setenv("FREETOKEN_FP8_SCRATCH_GEMM", "1")
    assert fp8mod._scratch_gemm_preferred() is True
    fp8mod._scratch_gemm_preferred.cache_clear()


def test_linear_routes_large_m_through_scratch_when_preferred(monkeypatch):
    """No GPU needed: with the scratch path preferred, fp8_pertensor_linear never touches the
    Triton kernels for M >= _SCRATCH_GEMM_MIN_M (the GEMV / GEMM launchers would fail on CPU
    tensors)."""
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.02
    q, s = _quant(w)
    a = torch.randn(fp8mod._SCRATCH_GEMM_MIN_M + 3, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(fp8mod, "_scratch_gemm_preferred", lambda: True)
    monkeypatch.setattr(fp8mod, "e4m3_native", lambda: False)
    monkeypatch.setattr(fp8mod, "_gemm", lambda *a, **k: pytest.fail("triton GEMM used"))
    out = fp8mod.fp8_pertensor_linear(a, q, s)
    ref = a.float() @ (q.float() * s[:, None]).t()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_small_m_threshold_routes_to_the_inline_kernel(monkeypatch):
    """Rows below _SCRATCH_GEMM_MIN_M take the inline kernel (the knob defaults to 2, so a
    4-row MTP window still uses the scratch: the inline kernel's e4m3 unpack is far slower
    on Turing)."""
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.02
    q, s = _quant(w)
    a = torch.randn(4, 128, dtype=torch.bfloat16)
    calls = []

    def fake_gemm(x, weight, scale, out_dtype):
        calls.append(x.shape)
        return (x.float() @ (q.float() * s[:, None]).t()).to(out_dtype)

    monkeypatch.setattr(fp8mod, "_scratch_gemm_preferred", lambda: True)
    monkeypatch.setattr(fp8mod, "e4m3_native", lambda: False)
    monkeypatch.setattr(fp8mod, "e4m3_kernel_view", lambda w: w)
    monkeypatch.setattr(fp8mod, "_gemm", fake_gemm)
    monkeypatch.setattr(fp8mod, "_SCRATCH_GEMM_MIN_M", 8)
    monkeypatch.setattr(fp8mod, "_gemm_scratch", lambda *a, **k: pytest.fail("scratch used below the threshold"))
    out = fp8mod.fp8_pertensor_linear(a, q, s)
    assert calls == [(4, 128)] and out.shape == (4, 64)
    assert fp8mod.__dict__["_SCRATCH_GEMM_MIN_M"] == 8
