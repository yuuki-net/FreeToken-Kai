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


def _decode(
    q,
    k_cache,
    v_cache,
    indptr,
    indices,
    q_positions,
    *,
    scales=None,
    spec=None,
    sliding_window=None,
    sinks=None,
):
    from freetoken.kernel.triton.attention import decode_paged_attention

    batch, num_q_heads, head_dim = q.shape
    max_kv_splits = 8
    dev = q.device
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=dev
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=dev)
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
        max_kv_splits=max_kv_splits,
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        sinks=sinks,
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


def _prefix_case(spec, dev, dtype, head_dim, num_kv_heads, prefix, new):
    """A prefill continuing a cached prefix: the prefix is quantized, this step's K/V is not."""
    total = prefix + new
    k = torch.randn(total, num_kv_heads, head_dim, device=dev, dtype=dtype)
    v = torch.randn(total, num_kv_heads, head_dim, device=dev, dtype=dtype)
    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)
    kc, ks, vc, vs = kc.to(dev), ks.to(dev), vc.to(dev), vs.to(dev)
    k_oracle = dequantize_rows(kc.cpu(), ks.cpu(), spec, dtype).to(dev)
    v_oracle = dequantize_rows(vc.cpu(), vs.cpu(), spec, dtype).to(dev)
    return (kc, ks, vc, vs), (k_oracle, v_oracle)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("split", [False, True], ids=["fused", "split-extend"])
def test_extend_reads_back_what_the_slab_holds(spec, split):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(6)
    dev = torch.device("cuda")
    dtype = torch.float16
    head_dim, num_kv_heads, num_q_heads = 256, 2, 16
    prefix, new = 48, 16
    total = prefix + new

    q = torch.randn(new, num_q_heads, head_dim, device=dev, dtype=dtype)
    (kc, ks, vc, vs), (k_oracle, v_oracle) = _prefix_case(
        spec, dev, dtype, head_dim, num_kv_heads, prefix, new
    )
    qo_indptr = torch.tensor([0, new], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, total], dtype=torch.int32, device=dev)
    kv_indices = torch.arange(total, dtype=torch.int32, device=dev)
    prefix_lens = torch.tensor([prefix], dtype=torch.int32, device=dev)

    extra = {}
    if split:
        # the split kernel takes this step's K/V unquantized alongside the cached prefix
        extra = dict(k_extend=k_oracle[prefix:].contiguous(), v_extend=v_oracle[prefix:].contiguous())

    common = dict(
        q=q,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        max_q_len=new,
        sm_scale=head_dim**-0.5,
    )
    want = extend_paged_attention(k_cache=k_oracle, v_cache=v_oracle, **common, **extra)
    got = extend_paged_attention(
        k_cache=kc, v_cache=vc, k_scales=ks, v_scales=vs, kv_quant=spec, **common, **extra
    )

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_the_paged_fallback_reads_back_what_the_slab_holds(spec):
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(7)
    dev = torch.device("cuda")
    dtype = torch.float16
    head_dim, num_kv_heads, num_q_heads = 128, 2, 8
    total = 33

    q = torch.randn(total, num_q_heads, head_dim, device=dev, dtype=dtype)
    (kc, ks, vc, vs), (k_oracle, v_oracle) = _prefix_case(
        spec, dev, dtype, head_dim, num_kv_heads, total, 0
    )
    indptr = torch.tensor([0, total], dtype=torch.int32, device=dev)
    indices = torch.arange(total, dtype=torch.int32, device=dev)
    q_to_req = torch.zeros(total, dtype=torch.int32, device=dev)
    q_positions = torch.arange(total, dtype=torch.int64, device=dev)

    common = dict(
        q=q,
        indptr=indptr,
        indices=indices,
        q_to_req=q_to_req,
        q_positions=q_positions,
        sm_scale=head_dim**-0.5,
    )
    want = paged_attention(k_cache=k_oracle, v_cache=v_oracle, **common)
    got = paged_attention(
        k_cache=kc, v_cache=vc, k_scales=ks, v_scales=vs, kv_quant=spec, **common
    )
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


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


# ---- gpt-oss: the sliding window and the attention sinks on top of the code slabs ----------
#
# gpt-oss is the one model family whose quantized KV goes through the SLIDING_WINDOW and
# HAS_SINKS branches of these kernels, at head_dim 64 (two blocks per row) and bf16. Every
# case above runs with neither, so these are the tests that reach those branches with QBITS
# set. The window layers read the window pool through the full -> window slot map, where
# every token that has left the window maps to slot 0: a zero row, masked by position. The
# decode case builds its slab the same way.

_OSS_HEAD_DIM, _OSS_KV_HEADS, _OSS_Q_HEADS = 64, 8, 16


def _sinks(dev):
    return torch.randn(_OSS_Q_HEADS, device=dev, dtype=torch.float32)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("window", [None, 16], ids=["full-layer", "window-layer"])
def test_decode_with_sinks_and_window_reads_back_what_the_slab_holds(spec, window):
    torch.manual_seed(21)
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    seq_lens = [5, 71]  # one shorter than the window, one far past it
    total_kv = sum(seq_lens)

    q = torch.randn(2, _OSS_Q_HEADS, _OSS_HEAD_DIM, device=dev, dtype=dtype)
    k = torch.randn(total_kv, _OSS_KV_HEADS, _OSS_HEAD_DIM, device=dev, dtype=dtype)
    v = torch.randn(total_kv, _OSS_KV_HEADS, _OSS_HEAD_DIM, device=dev, dtype=dtype)
    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)

    # row 0 is the all-zero sentinel; the tokens live in rows 1..total_kv
    def with_sentinel(t):
        return torch.cat([torch.zeros_like(t[:1]), t]).to(dev)

    kc, ks, vc, vs = (with_sentinel(t) for t in (kc, ks, vc, vs))
    k_oracle = dequantize_rows(kc.cpu(), ks.cpu(), spec, dtype).to(dev)
    v_oracle = dequantize_rows(vc.cpu(), vs.cpu(), spec, dtype).to(dev)

    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=dev)
    indices = torch.arange(1, total_kv + 1, dtype=torch.int32, device=dev)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=dev)
    if window is not None:
        # what translate_loc_from_full_to_swa hands the kernel once tokens leave the window
        start = seq_lens[0]
        gone = torch.arange(seq_lens[1], device=dev) < seq_lens[1] - window
        indices[start:][gone] = 0

    sinks = _sinks(dev)
    extra = dict(sliding_window=window, sinks=sinks)
    want = _decode(q, k_oracle, v_oracle, indptr, indices, q_positions, **extra)
    got = _decode(q, kc, vc, indptr, indices, q_positions, scales=(ks, vs), spec=spec, **extra)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
    # the sinks are not a no-op here, so the comparison above covered their branch
    plain = _decode(q, kc, vc, indptr, indices, q_positions, scales=(ks, vs), spec=spec,
                    sliding_window=window)
    assert not torch.allclose(plain, got, atol=1e-3)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("split", [False, True], ids=["fused", "split-extend"])
@pytest.mark.parametrize("window", [None, 24], ids=["full-layer", "window-layer"])
def test_extend_with_sinks_and_window_reads_back_what_the_slab_holds(spec, split, window):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(22)
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    prefix, new = 48, 16  # the window (24) ends inside the cached prefix
    total = prefix + new

    q = torch.randn(new, _OSS_Q_HEADS, _OSS_HEAD_DIM, device=dev, dtype=dtype)
    (kc, ks, vc, vs), (k_oracle, v_oracle) = _prefix_case(
        spec, dev, dtype, _OSS_HEAD_DIM, _OSS_KV_HEADS, prefix, new
    )
    extra = {}
    if split:
        extra = dict(k_extend=k_oracle[prefix:].contiguous(), v_extend=v_oracle[prefix:].contiguous())

    common = dict(
        q=q,
        qo_indptr=torch.tensor([0, new], dtype=torch.int32, device=dev),
        kv_indptr=torch.tensor([0, total], dtype=torch.int32, device=dev),
        kv_indices=torch.arange(total, dtype=torch.int32, device=dev),
        prefix_lens=torch.tensor([prefix], dtype=torch.int32, device=dev),
        max_q_len=new,
        sm_scale=_OSS_HEAD_DIM**-0.5,
        sliding_window=window,
        sinks=_sinks(dev),
    )
    want = extend_paged_attention(k_cache=k_oracle, v_cache=v_oracle, **common, **extra)
    got = extend_paged_attention(
        k_cache=kc, v_cache=vc, k_scales=ks, v_scales=vs, kv_quant=spec, **common, **extra
    )

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_the_paged_fallback_with_sinks_and_window_reads_back_what_the_slab_holds(spec):
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(23)
    dev = torch.device("cuda")
    # fp16, not gpt-oss's bf16: this kernel accumulates its output in the output dtype, and bf16
    # has a 0.0078 step at 1.0 -- the unquantized, window-less, sink-less call is already that
    # far from a float32 reference, so an oracle comparison at 2e-3 would fail on rounding
    # alone. (gpt-oss never takes this path: head_dim 64 always goes to the extend kernel.)
    dtype = torch.float16
    total = 33

    q = torch.randn(total, _OSS_Q_HEADS, _OSS_HEAD_DIM, device=dev, dtype=dtype)
    (kc, ks, vc, vs), (k_oracle, v_oracle) = _prefix_case(
        spec, dev, dtype, _OSS_HEAD_DIM, _OSS_KV_HEADS, total, 0
    )
    common = dict(
        q=q,
        indptr=torch.tensor([0, total], dtype=torch.int32, device=dev),
        indices=torch.arange(total, dtype=torch.int32, device=dev),
        q_to_req=torch.zeros(total, dtype=torch.int32, device=dev),
        q_positions=torch.arange(total, dtype=torch.int64, device=dev),
        sm_scale=_OSS_HEAD_DIM**-0.5,
        sliding_window=8,
        sinks=_sinks(dev),
    )
    want = paged_attention(k_cache=k_oracle, v_cache=v_oracle, **common)
    got = paged_attention(
        k_cache=kc, v_cache=vc, k_scales=ks, v_scales=vs, kv_quant=spec, **common
    )
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "head_dim", "cached_lens", "q_lens", "window", "use_sinks"),
    [
        (16, 2, 128, [0, 129, 1017], [4, 1, 8], None, False),
        (16, 2, 256, [300, 2000, 70], [4, 8, 2], 3, True),
        (24, 4, 256, [500, 33], [5, 2], 48, False),
        (8, 1, 512, [700, 64], [8, 8], None, True),
    ],
)
def test_multi_row_decode_reads_back_what_the_slab_holds(
    spec, num_q_heads, num_kv_heads, head_dim, cached_lens, q_lens, window, use_sinks
):
    """Up to 8 query rows per request take the decode kernel (a short extend: a verify window,
    a few-token prefill tail), so its q8_0 / q4_0 arm sees a rows x heads M axis: rows that see
    no key of a split, a window shorter than the rows, a non-power-of-two group, head_dim 512."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(11)
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    kv_lens = [c + r for c, r in zip(cached_lens, q_lens)]
    total_kv, num_rows = sum(kv_lens), sum(q_lens)
    q = torch.randn(num_rows, num_q_heads, head_dim, device=dev, dtype=dtype)
    k = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=dtype)
    v = torch.randn(total_kv, num_kv_heads, head_dim, device=dev, dtype=dtype)
    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)
    k_oracle = dequantize_rows(kc.cpu(), ks.cpu(), spec, dtype).to(dev)
    v_oracle = dequantize_rows(vc.cpu(), vs.cpu(), spec, dtype).to(dev)
    kc, ks, vc, vs = kc.to(dev), ks.to(dev), vc.to(dev), vs.to(dev)

    indptr = torch.tensor([0, *kv_lens], device=dev).cumsum(0).to(torch.int32)
    qo_indptr = torch.tensor([0, *q_lens], device=dev).cumsum(0).to(torch.int32)
    indices = torch.arange(total_kv, dtype=torch.int32, device=dev)
    q_positions = torch.cat([torch.arange(c, c + r, device=dev) for c, r in zip(cached_lens, q_lens)])
    sinks = torch.randn(num_q_heads, dtype=torch.float32, device=dev) if use_sinks else None

    def run(k_cache, v_cache, **quant):
        splits = 64
        attn_logits = torch.empty(num_rows, num_q_heads, splits, head_dim, dtype=torch.float32, device=dev)
        attn_lse = torch.empty(num_rows, num_q_heads, splits, dtype=torch.float32, device=dev)
        return decode_paged_attention(
            q, k_cache, v_cache, indptr, indices, q_positions, attn_logits, attn_lse, head_dim**-0.5,
            sliding_window=window, sinks=sinks, qo_indptr=qo_indptr, max_q_len=max(q_lens), **quant,
        )

    want = run(k_oracle, v_oracle)
    got = run(kc, vc, k_scales=ks, v_scales=vs, kv_quant=spec)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0)
    assert cos > 0.9999, f"cosine {cos:.6f}"
