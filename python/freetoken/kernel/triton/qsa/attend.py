# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""Sparse paged GQA over the QSA selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.attention import _dequant_tile
from freetoken.utils import init_logger
from freetoken.utils.torch_utils import clear_cuda_error

logger = init_logger(__name__)

try:  # triton >= 3.0
    from triton.runtime.errors import OutOfResources as _OUT_OF_RESOURCES
except ImportError:  # pragma: no cover - older triton raises a plain error

    class _OUT_OF_RESOURCES(Exception):  # noqa: N801
        pass


# Shared memory per block, by what the profile asks for. Wider tiles and deeper pipelining
# read more of the KV slab at once, so both are ways down. Order matters: BLOCK_N is worth
# more than num_stages (it is what the profile was chosen for), so give up the pipelining
# first and only then halve the tile.
_LADDER_DESC = "BLOCK_N/num_stages: (n,2) -> (n,1) -> (n/2,2) -> (n/2,1) -> (16,2) -> (16,1)"

# key -> the profile that was accepted, so the exception is paid once per shape and not once
# per layer per token.
_ACCEPTED: dict[tuple, tuple[int, int, int, int]] = {}
_REFUSED: set[tuple] = set()


def _profile_ladder(key, block_n: int, target_splits: int, partial_warps: int):
    """The tuned profile first, then progressively cheaper ones."""
    if key in _ACCEPTED:
        yield _ACCEPTED[key]
        return
    seen = set()
    for width in (block_n, block_n // 2, 16):
        if width < 16 or width in seen:
            continue
        seen.add(width)
        # the narrow tile is the tuned decode profile, which pairs 16 with four warps
        warps = partial_warps if width == block_n else (4 if width == 16 else partial_warps)
        for stages in (2, 1):
            yield width, target_splits, warps, stages


def _note_refusal(key, block_n, target_splits, warps, stages, exc) -> None:
    seen = (key, block_n, stages)
    if seen in _REFUSED:
        return
    _REFUSED.add(seen)
    logger.warning_rank0(
        f"qsa_sparse: BLOCK_N={block_n} num_stages={stages} does not fit this device's "
        f"shared memory ({exc}); stepping the tile down"
    )


def _remember(key, block_n, target_splits, warps, stages) -> None:
    if key in _ACCEPTED:
        return
    _ACCEPTED[key] = (block_n, target_splits, warps, stages)
    tuned_block_n = key[-2]
    if (block_n, stages) != (tuned_block_n, 2):
        logger.info_rank0(
            f"qsa_sparse: BLOCK_N={block_n} num_stages={stages} num_warps={warps} for "
            f"{key[2]} heads x {key[3]} dim, top-k {key[5]}, QBITS={key[1]} "
            f"(tuned profile was BLOCK_N={tuned_block_n} num_stages=2)"
        )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ks_cache_ptr,
    vs_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    QBITS: tl.constexpr,
    QBLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    # row * stride can overflow int32 for large row counts.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request.to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if QBITS == 0:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        else:
            # Code bytes: 8-bit is one per element, 4-bit packs a pair low nibble first.
            byte_off = dim_offsets // 2 if QBITS == 4 else dim_offsets
            scale_off = dim_offsets // QBLOCK
            nib_hi = (dim_offsets % 2) == 1
            row_k = (
                safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
            )
            row_ks = (
                safe_page[None, :] * stride_ks_block
                + page_offset[None, :] * stride_ks_token
                + kv_head * stride_ks_head
            )
            keys = _dequant_tile(
                k_cache_ptr,
                row_k + byte_off[:, None],
                ks_cache_ptr,
                row_ks + scale_off[:, None],
                nib_hi[:, None],
                valid[None, :],
                QBITS,
            ).to(query.dtype)
            row_v = (
                safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
            )
            row_vs = (
                safe_page[:, None] * stride_vs_block
                + page_offset[:, None] * stride_vs_token
                + kv_head * stride_vs_head
            )
            values = _dequant_tile(
                v_cache_ptr,
                row_v + byte_off[None, :],
                vs_cache_ptr,
                row_vs + scale_off[None, :],
                nib_hi[None, :],
                valid[:, None],
                QBITS,
            ).to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    kv_quant=None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged K/V caches.

    With ``kv_quant`` (``--kv-cache-dtype``) the caches hold block-quantized codes and
    ``k_scales``/``v_scales`` their fp16 block scales; the tiles are dequantized inside the
    kernel. Only the paged K/V is quantized -- the compressed index slab, the pending ring
    and the scratch rows stay 16-bit, because they decide *which* blocks get read and an
    error there changes the selection rather than blurring a value.
    """

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    head_dim = q.shape[2]
    if kv_quant is None:
        if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
            raise ValueError("QSA sparse attention requires valid grouped-query heads")
        assert q.dtype == k_cache.dtype == v_cache.dtype
    else:
        if k_scales is None or v_scales is None:
            raise ValueError("QSA sparse attention got a quantized slab without scales")
        if k_cache.shape[3] != kv_quant.code_bytes_per_row(head_dim):
            raise ValueError("QSA code slab does not match the head_dim it claims")
        if q.shape[1] % k_cache.shape[2]:
            raise ValueError("QSA sparse attention requires valid grouped-query heads")
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.stride(2) == 1 and k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    # ...and GB300 has 227 KiB of shared memory per block. A 3060 has 99, and the prefill
    # profile above wants more than that for some shapes: Triton refuses the launch with
    # OutOfResources before it touches the GPU, which killed Qwen3.8-Flash-Next on two of
    # them on the first request (the boot itself was clean, and the decode profile -- the
    # narrow tile -- captures and replays fine, so nothing looks wrong until someone asks a
    # question). The tuned profile cannot be the only one on offer.
    #
    # So: try it, and on a refusal step down and try again. The refusal is raised while
    # loading the compiled module, so nothing has run and a retry is safe. Step-downs are
    # remembered per (device, shape, profile) -- a decode step calls this once per layer, and
    # paying an exception each time would cost more than the tile ever saved.
    key = (
        q.device.index,
        0 if kv_quant is None else kv_quant.bits,
        q.shape[1],
        q.shape[2],
        k_cache.shape[1],
        logical_indices.shape[1],
        block_m,
        block_n,
        target_splits,
    )
    for block_n, target_splits, partial_warps, num_stages in _profile_ladder(
        key, block_n, target_splits, partial_warps
    ):
        num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
        # Avoid empty splits when the selection width is smaller than the profile.
        max_useful_splits = 1 << (num_tiles.bit_length() - 1)
        num_splits = min(max_useful_splits, target_splits)

        # Split=1 writes output directly and compiles out all workspace accesses.
        if num_splits == 1:
            partial_output = out
            partial_lse = out
        else:
            # FP32 partials preserve accuracy when merging independently normalized
            # splits.
            partial_output = torch.empty(
                (num_splits, *q.shape), dtype=torch.float32, device=q.device
            )
            partial_lse = torch.empty(
                (num_splits, q.shape[0], q.shape[1]),
                dtype=torch.float32,
                device=q.device,
            )

        partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
        try:
            _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
                q,
                k_cache,
                v_cache,
                # unquantized: the slab stands in for the scale pointer (QBITS == 0 prunes every use)
                k_scales if k_scales is not None else k_cache,
                v_scales if v_scales is not None else v_cache,
                logical_indices,
                block_table,
                token_to_req,
                partial_output,
                partial_lse,
                out,
                q.stride(0),
                q.stride(1),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                k_scales.stride(0) if k_scales is not None else 0,
                k_scales.stride(1) if k_scales is not None else 0,
                k_scales.stride(2) if k_scales is not None else 0,
                v_scales.stride(0) if v_scales is not None else 0,
                v_scales.stride(1) if v_scales is not None else 0,
                v_scales.stride(2) if v_scales is not None else 0,
                logical_indices.stride(0),
                block_table.stride(0),
                out.stride(0),
                out.stride(1),
                q.shape[0],
                k_cache.shape[0],
                block_table.shape[0],
                QBITS=0 if kv_quant is None else kv_quant.bits,
                QBLOCK=1 if kv_quant is None else kv_quant.block,
                TOPK=logical_indices.shape[1],
                PAGE_SIZE=k_cache.shape[1],
                PAGE_TABLE_WIDTH=block_table.shape[1],
                GROUP_SIZE=group_size,
                HEAD_DIM=q.shape[2],
                NUM_QUERY_HEADS=q.shape[1],
                NUM_SPLITS=num_splits,
                NUM_TILES=num_tiles,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=partial_warps,
                num_stages=num_stages,
            )
        except _OUT_OF_RESOURCES as exc:
            # Nothing has run: Triton raises this from _init_handles, while it loads the
            # compiled module, so the next profile starts from the same state -- except for
            # the driver error the refused load leaves in this thread's slot. Left there it
            # surfaces later and elsewhere: a tensor destructor rethrowing it as
            # "terminate called after throwing c10::AcceleratorError" killed a whole pytest
            # process two files after the refusal.
            clear_cuda_error()
            _note_refusal(key, block_n, target_splits, partial_warps, num_stages, exc)
            continue
        _remember(key, block_n, target_splits, partial_warps, num_stages)
        break
    else:
        raise RuntimeError(
            "qsa_sparse: no tile profile fits this device's shared memory "
            f"(tried {_LADDER_DESC})"
        )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["qsa_sparse_paged_attention"]
