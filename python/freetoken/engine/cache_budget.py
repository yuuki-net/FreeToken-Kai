"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


# Every KV pool allocates one page past the usable ones for padded / dummy rows to write into
# (create_kv_pool and rebuild_from_config: num_pages + 1). DSV4's solver already takes it off;
# the generic arithmetic below used to price only the usable pages.
_DUMMY_PAGES = 1


def pool_pages(num_pages: int) -> int:
    """Pages a pool of ``num_pages`` usable pages actually allocates."""
    return num_pages + _DUMMY_PAGES


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies: MoE slots, plus the
    ``num_pages`` usable KV pages and the pool's dummy page."""
    return moe_cache_size * per_expert_bytes + pool_pages(num_pages) * cache_per_page


class CacheBudgetTooSmall(AssertionError):
    """The smallest workable (experts, KV) plan does not fit the budget.

    Still an ``AssertionError`` -- what callers and tests have always caught -- but it carries
    the numbers, so the caller that knows where the budget came from (free VRAM, memory_ratio,
    weights, fixed pools) can say which flag closes the gap and to what value. The bare
    message used to name three levers with no numbers, which on a 6 GB card meant restarting
    the model a few times to find out which one mattered.
    """

    def __init__(
        self,
        message: str,
        *,
        budget_bytes: int,
        need_bytes: int,
        moe_slots: int,
        per_expert_bytes: int,
        kv_pages: int,
        cache_per_page: int,
        overlap_floor: bool,
        num_experts: int,
    ):
        super().__init__(message)
        self.budget_bytes = budget_bytes
        self.need_bytes = need_bytes
        self.moe_slots = moe_slots
        self.per_expert_bytes = per_expert_bytes
        self.kv_pages = kv_pages
        self.cache_per_page = cache_per_page
        self.overlap_floor = overlap_floor
        self.num_experts = num_experts

    def numbers(self) -> dict:
        return {
            "budget_bytes": self.budget_bytes,
            "need_bytes": self.need_bytes,
            "moe_slots": self.moe_slots,
            "per_expert_bytes": self.per_expert_bytes,
            "kv_pages": self.kv_pages,
            "cache_per_page": self.cache_per_page,
            "overlap_floor": self.overlap_floor,
            "num_experts": self.num_experts,
        }


# memory_ratio above this leaves no room for CUDA graphs and activations; suggesting it would
# trade a clean startup error for an OOM later.
_MAX_SUGGESTED_RATIO = 0.98


def shortfall_fixes(
    exc: CacheBudgetTooSmall,
    *,
    baseline_free: int,
    memory_ratio: float,
    weights_bytes: int,
    fixed_cache_size: int,
    page_size: int,
    min_reserve_tokens: int = 0,
) -> dict:
    """Each single flag change that would make the smallest plan fit, or None where that flag
    cannot. Pure arithmetic; every value returned is one ``resolve_moe_cache_auto`` accepts.

    - ``kv_reserve_tokens``: the most KV the budget leaves after the expert floor. With that
      reserve the greedy fill can only take experts out of what is left, so the plan fits.
    - ``memory_ratio``: the smallest ratio (to 0.01) whose budget covers the smallest plan,
      unless that is above ``_MAX_SUGGESTED_RATIO``.

    No ``--disable-moe-prefill-overlap``: the plan already drops overlap on its own whenever
    that alone makes it fit, so a plan that still fails has the ``num_experts`` floor.
    """
    expert_floor = exc.moe_slots * exc.per_expert_bytes

    kv_tokens = None
    left_for_kv = exc.budget_bytes - expert_floor
    if left_for_kv > 0:
        pages = left_for_kv // exc.cache_per_page - _DUMMY_PAGES
        tokens = pages * page_size
        if pages > 1 and tokens >= min_reserve_tokens:
            kv_tokens = int(tokens)

    ratio = None
    if baseline_free > 0:
        need = exc.need_bytes + weights_bytes + fixed_cache_size
        r = max(memory_ratio, int(need * 100 / baseline_free) / 100)
        while int(r * baseline_free) - weights_bytes - fixed_cache_size < exc.need_bytes:
            r = round(r + 0.01, 2)
            if r > _MAX_SUGGESTED_RATIO:
                break
        if r <= _MAX_SUGGESTED_RATIO:
            ratio = r

    return {"kv_reserve_tokens": kv_tokens, "memory_ratio": ratio}


def _mib(b: int) -> str:
    return f"{b / (1 << 20):.1f} MiB"


def _gib(b: int) -> str:
    return f"{b / (1 << 30):.2f} GiB"


def explain_shortfall(
    exc: CacheBudgetTooSmall,
    *,
    baseline_free: int,
    memory_ratio: float,
    weights_bytes: int,
    fixed_parts: dict[str, int],
    page_size: int,
    min_reserve_tokens: int = 0,
) -> str:
    """The startup error for a budget that cannot hold the smallest plan: where the budget went,
    what the smallest plan is made of, and the exact value of each flag that would close it."""
    fixed_total = sum(fixed_parts.values())
    fixes = shortfall_fixes(
        exc,
        baseline_free=baseline_free,
        memory_ratio=memory_ratio,
        weights_bytes=weights_bytes,
        fixed_cache_size=fixed_total,
        page_size=page_size,
        min_reserve_tokens=min_reserve_tokens,
    )
    parts = "  ".join(f"- {name} {_gib(b)}" for name, b in fixed_parts.items() if b) or "- fixed 0"
    expert_floor = exc.moe_slots * exc.per_expert_bytes
    kv_now = pool_pages(exc.kv_pages) * exc.cache_per_page
    short = exc.need_bytes - exc.budget_bytes
    lines = [
        f"cache budget too small: the smallest plan needs {_mib(exc.need_bytes)}, the budget is "
        f"{_mib(exc.budget_bytes)} ({_mib(short)} short).",
        f"  budget   = memory_ratio {memory_ratio:g} x {_gib(baseline_free)} free before the model"
        f"  - weights {_gib(weights_bytes)}  {parts}",
        f"  smallest = {exc.moe_slots} expert slots x {_mib(exc.per_expert_bytes)} = {_mib(expert_floor)}"
        f"  + KV reserve {exc.kv_pages * page_size} tokens = {_mib(kv_now)}",
    ]
    options = []
    if fixes["kv_reserve_tokens"] is not None:
        options.append(
            f"    --kv-reserve-tokens {fixes['kv_reserve_tokens']}"
            f"   (what the budget leaves for KV after the expert floor)"
        )
    if fixes["memory_ratio"] is not None:
        options.append(f"    --memory-ratio {fixes['memory_ratio']:.2f}")
    if options:
        lines.append("  any one of these fits:")
        lines.extend(options)
    else:
        lines.append(
            "  no single flag closes it: free GPU memory held by other processes, or load fewer "
            "weights onto this GPU"
        )
    if fixes["kv_reserve_tokens"] is None:
        why = (
            f"the expert floor alone is {_mib(expert_floor)} of a {_mib(exc.budget_bytes)} budget"
            if exc.budget_bytes - expert_floor <= exc.cache_per_page
            else f"the model needs at least {min_reserve_tokens} KV tokens"
        )
        lines.append(f"  (--kv-reserve-tokens cannot: {why})")
    if fixes["memory_ratio"] is None:
        lines.append(f"  (--memory-ratio cannot: it would have to exceed {_MAX_SUGGESTED_RATIO})")
    return "\n".join(lines)


def describe_plan(
    *,
    moe_cache_size: int,
    num_pages: int,
    per_expert_bytes: int,
    cache_per_page: int,
    budget_bytes: int,
    weights_bytes: int,
    fixed_parts: dict[str, int],
    page_size: int,
) -> str:
    """One line for a plan that fit: what the budget was spent on, so a later OOM or a slow
    decode can be read against it without re-deriving the arithmetic."""
    experts = moe_cache_size * per_expert_bytes
    kv = pool_pages(num_pages) * cache_per_page
    fixed = ", ".join(f"{name} {_gib(b)}" for name, b in fixed_parts.items() if b) or "none"
    return (
        f"cache plan: weights {_gib(weights_bytes)}; fixed {fixed}; "
        f"experts {moe_cache_size} slots {_gib(experts)}; KV {num_pages * page_size} tokens "
        f"{_gib(kv)}; unspent {_mib(budget_bytes - experts - kv)} of the {_gib(budget_bytes)} budget"
    )


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.

    Overlap is feasible only when its floor fits the budget next to the KV reserve. Where it
    does not but the ``num_experts`` floor does, the plan turns overlap off instead of
    failing: overlap only hides the prefill bank transfer, and one 12 GB card with a 512-expert
    model (Qwen3.8-Flash-Next) cannot hold the 1024-slot floor. The KV reserve is never cut
    for it -- that is a context length someone asked for.

    ``num_pages`` is the usable count. The pool allocates one dummy page on top of it
    (``create_kv_pool``: ``num_pages + 1``), so that page is charged to the budget here too.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    kv_reserve_bytes = pool_pages(kv_reserve_pages) * cache_per_page
    if overlap and 2 * num_experts * per_expert_bytes + kv_reserve_bytes > budget_bytes:
        overlap = False  # the overlap floor does not fit next to the KV reserve
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page - _DUMMY_PAGES, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = required_bytes(moe_cache_size, num_pages, per_expert_bytes, cache_per_page)
    if total > budget_bytes:
        raise CacheBudgetTooSmall(
            f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
            f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
            "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)",
            budget_bytes=budget_bytes,
            need_bytes=total,
            moe_slots=moe_cache_size,
            per_expert_bytes=per_expert_bytes,
            kv_pages=num_pages,
            cache_per_page=cache_per_page,
            overlap_floor=overlap,
            num_experts=num_experts,
        )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
    fixed_parts: dict[str, int] | None = None,
    min_reserve_tokens: int = 0,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``fixed_parts`` names what ``fixed_cache_size`` is made of and ``min_reserve_tokens`` is
    the model's own KV floor; both are only read to explain a budget that is too small.

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    try:
        return plan_cache_budget(
            budget_bytes=budget_bytes,
            per_expert_bytes=per_expert_bytes,
            cache_per_page=cache_per_page,
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=prefill_overlap,
            kv_reserve_pages=kv_reserve_pages,
            max_slots=max_slots,
        )
    except CacheBudgetTooSmall as exc:
        message = explain_shortfall(
            exc,
            baseline_free=baseline_free,
            memory_ratio=memory_ratio,
            weights_bytes=weights_bytes,
            fixed_parts=fixed_parts or {"fixed cache": fixed_cache_size},
            page_size=page_size,
            min_reserve_tokens=min_reserve_tokens,
        )
        raise CacheBudgetTooSmall(message, **exc.numbers()) from None
