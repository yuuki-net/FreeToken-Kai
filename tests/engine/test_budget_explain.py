"""A budget that cannot hold the smallest plan says which flag closes the gap, and to what.

The message this replaces named three levers and no numbers ("raise memory_ratio, lower
kv_reserve_tokens, or free GPU memory"). On a 6 GB card that meant restarting a model several
times to find out which lever mattered and by how much. The guarantee worth pinning is not the
wording but that **every value the message suggests actually fits** -- a suggestion that still
fails is worse than none.
"""

from __future__ import annotations

import pytest

from freetoken.engine.cache_budget import (
    CacheBudgetTooSmall,
    describe_plan,
    resolve_moe_cache_auto,
    shortfall_fixes,
)

MiB = 1 << 20
GiB = 1 << 30

# Shaped like the 2060 failure that motivated this: ~5 GiB free, 256 experts, a KV reserve
# that is a little too generous for what is left after the weights and the GDN pool.
BASE = dict(
    baseline_free=int(4.97 * GiB),
    weights_bytes=int(2.3 * GiB),
    memory_ratio=0.82,
    cache_per_page=20_000,          # bytes per KV page (page_size 1 -> per token)
    fixed_cache_size=int(0.55 * GiB),
    per_expert_bytes=int(1.7 * MiB),
    num_experts=256,
    total_experts=256 * 40,
    prefill_overlap=False,
    kv_reserve_tokens=65536,
    page_size=1,
)


def _fail(**over):
    kw = {**BASE, **over}
    with pytest.raises(CacheBudgetTooSmall) as info:
        resolve_moe_cache_auto(**kw)
    return kw, info.value


def _fits(**kw):
    size, pages, _ = resolve_moe_cache_auto(**kw)
    return size, pages


def test_it_is_still_the_assertion_callers_catch():
    _, exc = _fail()
    assert isinstance(exc, AssertionError)
    assert "cache budget too small" in str(exc)


def test_the_message_says_where_the_budget_went_and_how_short_it_is():
    kw, exc = _fail(fixed_parts={"KV fixed": int(0.03 * GiB), "GDN state pool": int(0.52 * GiB)})
    msg = str(exc)
    assert "short" in msg
    assert "GDN state pool 0.52 GiB" in msg      # the part no KV flag touches, named
    assert "weights 2.30 GiB" in msg
    assert "256 expert slots" in msg and "KV reserve 65536 tokens" in msg


def test_the_suggested_kv_reserve_fits():
    kw, exc = _fail()
    fixes = shortfall_fixes(
        exc, baseline_free=kw["baseline_free"], memory_ratio=kw["memory_ratio"],
        weights_bytes=kw["weights_bytes"], fixed_cache_size=kw["fixed_cache_size"],
        page_size=kw["page_size"],
    )
    assert fixes["kv_reserve_tokens"] is not None
    assert f"--kv-reserve-tokens {fixes['kv_reserve_tokens']}" in str(exc)
    _fits(**{**kw, "kv_reserve_tokens": fixes["kv_reserve_tokens"]})


def test_the_suggested_memory_ratio_fits_and_is_the_smallest_to_a_hundredth():
    kw, exc = _fail()
    fixes = shortfall_fixes(
        exc, baseline_free=kw["baseline_free"], memory_ratio=kw["memory_ratio"],
        weights_bytes=kw["weights_bytes"], fixed_cache_size=kw["fixed_cache_size"],
        page_size=kw["page_size"],
    )
    r = fixes["memory_ratio"]
    assert r is not None and f"--memory-ratio {r:.2f}" in str(exc)
    _fits(**{**kw, "memory_ratio": r})
    with pytest.raises(CacheBudgetTooSmall):
        resolve_moe_cache_auto(**{**kw, "memory_ratio": round(r - 0.01, 2)})


def test_overlap_that_alone_blocks_the_plan_is_dropped_not_suggested():
    # With overlap on, the floor is 2 x 256 slots; the num_experts floor fits, so the plan
    # starts with overlap off instead of stopping with --disable-moe-prefill-overlap.
    kw = {**BASE, "prefill_overlap": True, "kv_reserve_tokens": 40_000}
    size, pages, overlap = resolve_moe_cache_auto(**kw)
    assert overlap is False and 256 <= size < 512 and pages >= 40_000
    # and when even the lower floor does not fit, the message has no overlap option
    _, exc = _fail(prefill_overlap=True)
    assert "--disable-moe-prefill-overlap" not in str(exc)
    assert exc.moe_slots == 256


def test_a_gap_no_single_flag_closes_says_so():
    # Weights alone exceed what memory_ratio could ever give: nothing to suggest.
    kw, exc = _fail(weights_bytes=int(4.9 * GiB))
    msg = str(exc)
    assert "no single flag closes it" in msg
    assert "--memory-ratio cannot" in msg
    assert "--kv-reserve-tokens cannot" in msg


def test_the_model_kv_floor_blocks_a_kv_suggestion_below_it():
    kw, exc = _fail()
    fixes = shortfall_fixes(
        exc, baseline_free=kw["baseline_free"], memory_ratio=kw["memory_ratio"],
        weights_bytes=kw["weights_bytes"], fixed_cache_size=kw["fixed_cache_size"],
        page_size=kw["page_size"], min_reserve_tokens=10**9,
    )
    assert fixes["kv_reserve_tokens"] is None


def test_a_plan_that_fits_is_described_in_one_line():
    line = describe_plan(
        moe_cache_size=925, num_pages=16447, per_expert_bytes=int(1.7 * MiB),
        cache_per_page=20_000, budget_bytes=int(2.0 * GiB), weights_bytes=int(2.3 * GiB),
        fixed_parts={"KV fixed": 0, "GDN state pool": int(0.52 * GiB)}, page_size=1,
    )
    assert "\n" not in line
    assert "experts 925 slots" in line and "KV 16447 tokens" in line
    assert "GDN state pool 0.52 GiB" in line and "KV fixed" not in line  # zero parts omitted
