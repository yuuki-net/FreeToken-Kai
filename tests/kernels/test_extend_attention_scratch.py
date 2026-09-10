"""``FREETOKEN_ATTN_SCRATCH``: prefill attention through cuBLAS instead of the fused kernel.

The two paths must answer the same thing, so every case here runs the same request twice --
once forced onto the scratch path, once forced onto the fused kernel -- and compares. What
the scratch path cannot express (a sliding window, sinks, a quantized cache, a short window)
must fall back rather than answer differently, which is checked by watching whether the fused
kernel gets called at all.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton import attention as attn


def _case(rows: int, prefix: int, *, q_heads: int = 16, kv_heads: int = 2, head_dim: int = 256,
          dtype=torch.float16, device="cuda", seed: int = 0):
    """One request: ``rows`` new tokens behind ``prefix`` tokens already in the paged cache.

    The prefix slots are shuffled, not consecutive: a real cache hands out whatever pages are
    free, and the gather has to follow kv_indices rather than a slice.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.2

    pages = prefix + 8
    slots = torch.randperm(pages, generator=gen, device=device)[:prefix].to(torch.int32)
    return dict(
        q=randn(rows, q_heads, head_dim),
        k_cache=randn(pages, kv_heads, head_dim),
        v_cache=randn(pages, kv_heads, head_dim),
        k_extend=randn(rows, kv_heads, head_dim),
        v_extend=randn(rows, kv_heads, head_dim),
        qo_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, prefix], dtype=torch.int32, device=device),
        kv_indices=slots,
        prefix_lens=torch.tensor([prefix], dtype=torch.int32, device=device),
        max_q_len=rows,
        sm_scale=head_dim**-0.5,
    )


def _run(case, monkeypatch, scratch: str, **extra):
    monkeypatch.setenv(attn._ATTN_SCRATCH_ENV, scratch)
    return attn.extend_paged_attention(**case, **extra)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="attention needs CUDA")


@cuda_only
@pytest.mark.parametrize(
    "rows, prefix",
    [
        (128, 0),      # a first chunk: no prefix at all, pure causal
        (256, 64),     # a short prefix, and rows not a multiple of the score tile
        (384, 1024),   # the ordinary case: a chunk behind a cached prefix
    ],
)
def test_the_two_paths_agree(rows, prefix, monkeypatch):
    case = _case(rows, prefix)
    scratch = _run(case, monkeypatch, "1")
    fused = _run(case, monkeypatch, "0")
    torch.testing.assert_close(scratch, fused, rtol=2e-3, atol=2e-3)


@cuda_only
def test_the_score_tile_does_not_change_the_answer(monkeypatch):
    """The query rows are walked in tiles sized from FREETOKEN_ATTN_SCRATCH_MB. The whole key
    range is scored in one go either way, so the tiling is bookkeeping, not arithmetic: a
    budget so small it forces the 16-row floor answers the same as one that fits every row.

    Not bit-identical, because the tile is the GEMM M and cuBLAS picks its kernel by shape;
    the difference is the last bit of fp16 (1.5e-5 here), not a different computation."""
    case = _case(256, 512)
    monkeypatch.setenv(attn._ATTN_SCRATCH_MB_ENV, "1")
    small = _run(case, monkeypatch, "1")
    monkeypatch.setenv(attn._ATTN_SCRATCH_MB_ENV, "512")
    large = _run(case, monkeypatch, "1")
    torch.testing.assert_close(small, large, rtol=1e-3, atol=1e-4)


@cuda_only
def test_grouped_query_heads_land_on_their_own_kv_head(monkeypatch):
    """A q head reads kv head ``q_head // group``. Getting that wrong still produces a
    plausible tensor, so it is checked against the fused kernel with a group of 8 and again
    with no grouping at all."""
    for q_heads, kv_heads in ((16, 2), (4, 4)):
        case = _case(192, 256, q_heads=q_heads, kv_heads=kv_heads, head_dim=128)
        scratch = _run(case, monkeypatch, "1")
        fused = _run(case, monkeypatch, "0")
        torch.testing.assert_close(scratch, fused, rtol=2e-3, atol=2e-3)


@cuda_only
def test_two_requests_in_one_batch(monkeypatch):
    """Each sequence has its own prefix and its own slice of q, and the scratch path walks
    them one at a time -- so a batch where the second request would be right only if the
    first one's offsets were used is the interesting one."""
    device, dtype = "cuda", torch.float16
    rows_a, rows_b, prefix_a, prefix_b = 130, 200, 64, 300
    gen = torch.Generator(device=device).manual_seed(7)
    randn = lambda *s: torch.randn(*s, generator=gen, device=device, dtype=dtype) * 0.2
    pages = prefix_a + prefix_b + 8
    perm = torch.randperm(pages, generator=gen, device=device).to(torch.int32)
    case = dict(
        q=randn(rows_a + rows_b, 16, 256),
        k_cache=randn(pages, 2, 256),
        v_cache=randn(pages, 2, 256),
        k_extend=randn(rows_a + rows_b, 2, 256),
        v_extend=randn(rows_a + rows_b, 2, 256),
        qo_indptr=torch.tensor([0, rows_a, rows_a + rows_b], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, prefix_a, prefix_a + prefix_b], dtype=torch.int32, device=device),
        kv_indices=perm[: prefix_a + prefix_b],
        prefix_lens=torch.tensor([prefix_a, prefix_b], dtype=torch.int32, device=device),
        max_q_len=max(rows_a, rows_b),
        sm_scale=256**-0.5,
    )
    scratch = _run(case, monkeypatch, "1")
    fused = _run(case, monkeypatch, "0")
    torch.testing.assert_close(scratch, fused, rtol=2e-3, atol=2e-3)


@cuda_only
def test_the_caller_buffer_is_the_one_written(monkeypatch):
    case = _case(192, 128)
    out = torch.empty_like(case["q"])
    got = _run(case, monkeypatch, "1", out=out)
    assert got is out
    assert torch.isfinite(out).all()


# --- what must fall back ------------------------------------------------------------------


def _watched(monkeypatch):
    """Count fused-kernel launches: a fallback is only a fallback if it reaches that kernel."""
    calls = []
    real = attn._extend_attention_split_kernel

    class _Watch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(grid)
                return real[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(attn, "_extend_attention_split_kernel", _Watch())
    return calls


@cuda_only
@pytest.mark.parametrize(
    "name, extra",
    [
        ("sliding window", {"sliding_window": 64}),
        ("attention sinks", {"sinks": torch.zeros(16, device="cuda")}),
    ],
)
def test_features_the_scratch_path_does_not_have_fall_back(name, extra, monkeypatch):
    case = _case(256, 128)
    calls = _watched(monkeypatch)
    _run(case, monkeypatch, "1", **extra)
    assert calls, f"{name} must reach the fused kernel"


@cuda_only
def test_a_short_extend_stays_on_the_fused_kernel(monkeypatch):
    """A verify window or a chat turn behind a cached prefix is a handful of rows: the GEMMs
    are thin and the gather would be paid for all of them."""
    case = _case(attn._ATTN_SCRATCH_MIN_ROWS - 1, 512)
    calls = _watched(monkeypatch)
    _run(case, monkeypatch, "1")
    assert calls


@cuda_only
def test_a_context_too_large_to_gather_falls_back(monkeypatch):
    """The gathered contiguous K/V is the whole context and is not bounded by the score
    budget, so past a multiple of it the fused kernel -- whose footprint is its tile -- takes
    the call back."""
    case = _case(256, 4096)
    monkeypatch.setenv(attn._ATTN_SCRATCH_MB_ENV, "1")  # 4 MiB of gather allowed
    calls = _watched(monkeypatch)
    _run(case, monkeypatch, "1")
    assert calls


def test_the_default_follows_the_shared_memory_the_card_has():
    """Ampere and later fit the fused (128, 64) tile and keep it; pre-Ampere does not."""
    attn._attn_scratch_default.cache_clear()
    try:
        from freetoken.utils import arch

        assert attn._attn_scratch_default() is arch.is_pre_ampere()
    finally:
        attn._attn_scratch_default.cache_clear()
