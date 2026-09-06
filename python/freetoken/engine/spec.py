"""Host-side arithmetic of MTP speculative decoding (torch-light, unit-testable).

One iteration verifies a window ``[t_last, d_1, ..., d_k]`` (k drafts) in a single extend
forward, samples the target at every row, keeps the longest prefix of drafts the samples
confirm, and rolls the per-request state back to the last accepted row. The pieces here
are the pure functions that decide and describe that; the GPU work lives in the engine,
the GDN layer and the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class SpecResult:
    """What one forward produced for the (single) request: the committed tokens (>= 1; the
    last one is the target's own sample) and the MTP drafts for the next window."""

    accepted: list[int]
    drafts: list[int] = field(default_factory=list)


def accept_drafts(sampled: Sequence[int], drafts: Sequence[int]) -> list[int]:
    """Verification rule: row j of the window sampled ``sampled[j]``; draft ``drafts[j]`` (the
    input of row j+1) is accepted iff it equals ``sampled[j]``. Returns ``sampled[:m+1]`` for
    ``m`` accepted drafts -- the accepted drafts themselves plus the sample after them.

    With a deterministic (greedy) draft this is exact rejection sampling: accepting when the
    target's own sample equals the draft happens with probability p(d), and the sample that
    ends the run is drawn from the residual distribution."""
    m = 0
    while m < len(drafts) and m < len(sampled) - 1 and sampled[m] == drafts[m]:
        m += 1
    return [int(t) for t in sampled[: m + 1]]


def pack_spec_message(result: SpecResult, k: int) -> torch.Tensor:
    """Fixed-length int32 vector ``[a, tok_0..tok_k (pad -1), d_1..d_k (pad -1)]`` for the
    pipeline ranks (same size every step, so the receiver needs no header)."""
    a = len(result.accepted)
    assert 1 <= a <= k + 1, (a, k)
    toks = list(result.accepted) + [-1] * (k + 1 - a)
    drafts = list(result.drafts)[:k] + [-1] * (k - min(len(result.drafts), k))
    return torch.tensor([a, *toks, *drafts], dtype=torch.int32)


def unpack_spec_message(vec: torch.Tensor, k: int) -> SpecResult:
    v = vec.tolist()
    a = int(v[0])
    toks = [int(t) for t in v[1 : 1 + a]]
    drafts = [int(d) for d in v[2 + k : 2 + 2 * k] if d >= 0]
    return SpecResult(accepted=toks, drafts=drafts)


def spec_message_len(k: int) -> int:
    return 2 * k + 2


def pages_to_free(keep_len: int, alloc_len: int, page_size: int) -> tuple[int, int]:
    """Pages ``[first, last)`` that only hold positions ``>= keep_len`` once a request whose
    pages were allocated up to ``alloc_len`` rolls back; the page holding position
    ``keep_len - 1`` stays."""
    first = -(-keep_len // page_size)
    last = -(-alloc_len // page_size)
    return first, max(first, last)


def rebuild_conv_state(prev_state: torch.Tensor, conv_in: torch.Tensor, accepted: int) -> torch.Tensor:
    """Conv left-context after ``accepted`` of the window's tokens: ``prev_state [dim, K-1]``
    (state before the window) followed by the first ``accepted`` conv inputs ``conv_in [T, dim]``,
    keeping the last ``K-1`` columns."""
    width = prev_state.shape[-1]
    cat = torch.cat([prev_state, conv_in[:accepted].to(prev_state.dtype).transpose(0, 1)], dim=-1)
    return cat[..., -width:].contiguous()


def ngram_context_after(host_ids: Sequence[int], drafts: Sequence[int], accepted: int, ctx_len: int, boundary: int) -> list[int]:
    """PLE n-gram context (the last ``ctx_len`` processed ids) after ``accepted`` rows of the
    window ``[host_ids[-1], drafts...]`` were kept. ``host_ids`` is the request's committed
    id list before the window (its last id is the window's first input)."""
    seq = list(host_ids) + list(drafts)
    end = len(host_ids) - 1 + accepted  # rows 0..accepted-1 = seq[len-1 : len-1+accepted]
    window = seq[max(0, end - ctx_len) : end]
    return [boundary] * (ctx_len - len(window)) + [int(t) for t in window]


__all__ = [
    "SpecResult",
    "accept_drafts",
    "pack_spec_message",
    "unpack_spec_message",
    "spec_message_len",
    "pages_to_free",
    "rebuild_conv_state",
    "ngram_context_after",
]
