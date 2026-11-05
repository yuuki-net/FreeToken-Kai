"""Round 2 of the window fixed cost (guides/71): the verify window's conv through the decode-style
update kernel equals the varlen prefill conv (output and the state it leaves); the Q4_0 quantizer
matches the vendored dequantizer; the packed draft head scores like the bf16 head within 4-bit
rounding and from every head storage (bf16, per-row fp8, Q8_0 rows); the 8-row dense MMQ tile
agrees with the 4-row one."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("seqlen", [1, 2, 4, 6, 8])
def test_window_conv_matches_the_varlen_conv(seqlen):
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen, causal_conv1d_window

    torch.manual_seed(seqlen)
    dev = torch.device("cuda")
    dim, width, slots = 256, 4, 3
    weight = torch.randn(dim, width, device=dev, dtype=torch.bfloat16) * 0.3
    state = torch.randn(slots, dim, width - 1, device=dev, dtype=torch.bfloat16)
    x = torch.randn(seqlen, dim, device=dev, dtype=torch.bfloat16)  # [total, conv_dim]
    x_in = x.clone()
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    # varlen (the old window path)
    st_a = state.clone()
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=dev)
    # each call gets its own copy: both write silu(conv) into their input, and for one token
    # x.t().contiguous() is x itself (a [dim, 1] view counts as contiguous), so sgl_kernel's varlen
    # would hand the window its output in place of x (申し送り 65: only [1], only on sgl_kernel)
    out_a = causal_conv1d_varlen(x.t().contiguous().clone(), weight, st_a, cu, idx,
                                 torch.ones(1, dtype=torch.bool, device=dev), max_seq_len=seqlen)
    # update kernel over the window
    st_b = state.clone()
    out_b = causal_conv1d_window(x.t().contiguous().clone().unsqueeze(0), st_b, weight, idx)[0]
    assert torch.allclose(out_a.float(), out_b.float(), atol=2e-2, rtol=2e-2), (out_a.float() - out_b.float()).abs().max()
    assert torch.allclose(st_a[1].float(), st_b[1].float(), atol=1e-2), (st_a[1].float() - st_b[1].float()).abs().max()
    assert torch.equal(st_a[0], st_b[0]) and torch.equal(st_a[2], st_b[2])  # other slots untouched
    assert torch.equal(x, x_in)  # the inputs were copies


def test_q4_0_quantizer_matches_the_vendored_dequantizer():
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequant_q4_0, quantize_q4_0

    torch.manual_seed(0)
    w = torch.randn(64, 320, device="cuda", dtype=torch.bfloat16) * 0.05
    q = quantize_q4_0(w)
    assert q.shape == (64, 320 // 32 * 18)
    back = ggml_dequantize(q, GGML_Q4_0, 64, 320, torch.float32)
    assert torch.allclose(back, dequant_q4_0(q, torch.float32).reshape(64, 320), rtol=2e-3, atol=1e-5)
    rel = (back - w.float()).abs().max() / w.float().abs().max()
    assert rel < 0.15, rel  # 4-bit with a per-32 scale


@pytest.mark.parametrize("storage", ["bf16", "fp8", "q8_0"])
@pytest.mark.parametrize("kind", ["q4_0", "q8_0"])
def test_draft_head_scores_like_the_head(storage, kind):
    from freetoken.engine.draft_head import draft_logits, pack_draft_head
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row
    from freetoken.models.gguf.dequant import quantize_q8_0

    torch.manual_seed(1)
    dev = torch.device("cuda")
    V, K = 4096, 512
    w = torch.randn(V, K, device=dev, dtype=torch.bfloat16) * 0.05
    if storage == "bf16":
        lm = SimpleNamespace(weight=w, out_features=V, in_features=K)
    elif storage == "fp8":
        q, s = quantize_fp8_per_row(w)
        lm = SimpleNamespace(weight=q, weight_scale=s, out_features=V, in_features=K)
    else:
        lm = SimpleNamespace(weight=quantize_q8_0(w), out_features=V, in_features=K)
    packed = pack_draft_head(lm, kind, chunk_rows=1000)
    h = torch.randn(3, K, device=dev, dtype=torch.bfloat16)
    logits = draft_logits(packed, kind, h, V).float()
    ref = h.float() @ w.float().t()
    rel = (logits - ref).abs().max() / ref.abs().max()
    assert rel < (0.12 if kind == "q4_0" else 0.03), rel
    # the greedy draft mostly agrees with the bf16 head (q4_0 flips a few close calls)
    agree = (logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= (0.6 if kind == "q4_0" else 0.99), agree


def test_a_head_the_checkpoint_quantized_itself_is_refused():
    from freetoken.engine.draft_head import UnsupportedHead, pack_draft_head

    nvfp4_like = SimpleNamespace(weight=torch.zeros(64, 256, dtype=torch.uint8, device="cuda"),
                                 weight_scale=torch.zeros(64, 32, device="cuda"), out_features=64, in_features=512)
    with pytest.raises(UnsupportedHead):
        pack_draft_head(nvfp4_like, "q4_0")


@pytest.mark.parametrize("rows", [5, 6, 8, 12])
def test_dense_q8_0_mmq_wide_tile_agrees_with_the_narrow_one(rows):
    """ncols_y > 4 takes the 8-token tile; the result equals the 4-token tile's (same q8_1
    activations, fp32 accumulation) within rounding."""
    from freetoken.kernel.gguf import ggml_mul_mat_a8
    from freetoken.models.gguf.dequant import GGML_Q8_0, quantize_q8_0

    torch.manual_seed(rows)
    dev = torch.device("cuda")
    N, K = 1280, 2560
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    q = quantize_q8_0(w)
    x = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
    wide = ggml_mul_mat_a8(q, x, GGML_Q8_0, N).float()
    narrow = torch.cat([ggml_mul_mat_a8(q, x[i:i + 4], GGML_Q8_0, N) for i in range(0, rows, 4)]).float()
    ref = x.float() @ w.float().t()
    assert (wide - narrow).abs().max() / ref.abs().max() < 2e-3
    assert (wide - ref).abs().max() / ref.abs().max() < 2e-2
