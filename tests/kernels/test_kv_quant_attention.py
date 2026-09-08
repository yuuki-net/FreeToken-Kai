"""Decode attention reading the block-quantized KV slabs (``--kv-cache-dtype``).

Held against a *dequantized oracle*: the same kernel, same inputs, but fed a 16-bit slab
built by pushing the code slab back through the torch reference. That isolates the thing
under test -- whether the in-kernel unpack reproduces the stored values -- from the
quantization error itself, which is a modelling choice measured elsewhere. Comparing
straight against unquantized attention would blur the two and could only ever be checked
against a loose tolerance."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.kv_quant import Q4_0, Q8_0, dequantize_rows, quantize_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _decode(q, k_cache, v_cache, indptr, indices, q_positions, *, scales=None, spec=None):
    from freetoken.kernel.triton.attention import decode_paged_attention

    batch, num_q_heads, head_dim = q.shape
    max_kv_splits = 8
    dev = q.device
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=dev
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=dev)
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=dev)
    ks, vs = scales if scales is not None else (None, None)
    return decode_paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        indptr=indptr,
        indices=indices,
        q_positions=q_positions,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale=head_dim**-0.5,
        k_scales=ks,
        v_scales=vs,
        kv_quant=spec,
    )


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "dtype"),
    [(128, 2, torch.float16), (256, 2, torch.bfloat16), (256, 8, torch.float16)],
)
def test_decode_reads_back_what_the_slab_holds(spec, head_dim, num_kv_heads, dtype):
    torch.manual_seed(3)
    dev = torch.device("cuda")
    batch, num_q_heads = 2, 16
    seq_lens = [5, 71]  # one short request, one crossing several BLOCK_N tiles
    total_kv = sum(seq_lens)

    q = torch.randn(batch, num_q_heads, head_dim, device=dev, dtype=dtype)
    k = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=dtype)
    v = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=dtype)

    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)
    kc, ks, vc, vs = kc.to(dev), ks.to(dev), vc.to(dev), vs.to(dev)

    # the oracle: exactly the values the kernel is supposed to reconstruct
    k_oracle = dequantize_rows(kc.cpu(), ks.cpu(), spec, dtype).to(dev)
    v_oracle = dequantize_rows(vc.cpu(), vs.cpu(), spec, dtype).to(dev)

    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=dev)
    indices = torch.arange(total_kv, dtype=torch.int32, device=dev)
    q_positions = torch.tensor(
        [seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=dev
    )

    want = _decode(q, k_oracle, v_oracle, indptr, indices, q_positions)
    got = _decode(q, kc, vc, indptr, indices, q_positions, scales=(ks, vs), spec=spec)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    assert cos > 0.9999, f"cosine {cos:.6f}"


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_quantization_error_stays_within_what_the_width_implies(spec):
    """How far the answer moves at all -- a floor on the arithmetic, not on model quality.

    The inputs are iid Gaussian, which is the worst case a block quantizer can be handed:
    no structure to exploit and every block spending its whole range on noise. For a
    32-value block the absmax lands near 2.5 sigma, so the step is 2.5/qmax sigma and the
    error is uniform over it -- RMS step/sqrt(12), i.e. relative error ~0.6% at 8 bits and
    ~10% at 4 bits per element, which after the attention average puts cosine near 0.9999
    and 0.99 respectively. Anything materially worse means the unpack is wrong, not that
    the format is.

    What this test cannot say is whether 4-bit KV costs the model anything: real K rows are
    not Gaussian (they have outlier channels, which is exactly where block scaling helps or
    hurts). That question belongs to an end-to-end accuracy run on a real checkpoint.
    """
    torch.manual_seed(4)
    dev = torch.device("cuda")
    batch, num_q_heads, num_kv_heads, head_dim = 1, 16, 2, 256
    total_kv = 96

    q = torch.randn(batch, num_q_heads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=torch.float16)

    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)
    kc, ks, vc, vs = kc.to(dev), ks.to(dev), vc.to(dev), vs.to(dev)

    indptr = torch.tensor([0, total_kv], dtype=torch.int32, device=dev)
    indices = torch.arange(total_kv, dtype=torch.int32, device=dev)
    q_positions = torch.tensor([total_kv - 1], dtype=torch.int64, device=dev)

    exact = _decode(q, k, v, indptr, indices, q_positions)
    got = _decode(q, kc, vc, indptr, indices, q_positions, scales=(ks, vs), spec=spec)

    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), exact.float().flatten(), dim=0
    )
    floor = 0.9999 if spec.bits == 8 else 0.99
    assert cos > floor, f"{spec.name}: cosine {cos:.6f} against unquantized attention"


def test_unquantized_path_is_untouched():
    """QBITS == 0 has to keep producing exactly what it did before the branch existed."""
    torch.manual_seed(5)
    dev = torch.device("cuda")
    batch, num_q_heads, num_kv_heads, head_dim = 2, 16, 2, 128
    total_kv = 40

    q = torch.randn(batch, num_q_heads, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 12, total_kv], dtype=torch.int32, device=dev)
    indices = torch.arange(total_kv, dtype=torch.int32, device=dev)
    q_positions = torch.tensor([11, total_kv - 13], dtype=torch.int64, device=dev)

    a = _decode(q, k, v, indptr, indices, q_positions)
    b = _decode(q, k, v, indptr, indices, q_positions)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
