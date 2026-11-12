from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from freetoken.kernel.triton.kv_rows import kv_row_scale, kv_scale_args, kv_tile, kv_tile_staged
from freetoken.kvcache.format import NONE, KVFormat


_MIN_BLOCK_KV = 32
# (device, --kv-cache-dtype bits, head_dim, q dtype) whose wide decode tile the card refused; decode_paged_attention
_WIDE_TILE_REFUSED: set[tuple] = set()


# a long decode request takes splits of at least this many keys, so the partial result each one writes stays small next to the keys it reads
_MIN_SPLIT_LEN = 128
# a short request still takes up to this many splits of _MIN_BLOCK_KV keys, so tiny requests stay parallel
_MIN_SPLITS = 8
_MAX_KV_SPLITS = 64
# stage 2 merges at most this many fp32 partial elements per step
_STAGE2_TILE_ELEMS = 8192


def _kv_offset_dtype(k_cache: torch.Tensor, v_cache: torch.Tensor):
    # Bound physical offsets, not request lengths.
    max_offset = max(
        sum((size - 1) * stride for size, stride in zip(cache.shape, cache.stride()))
        for cache in (k_cache, v_cache)
    )
    return tl.int32 if max_offset < 1 << 31 else tl.int64


def _stage2_launch(max_kv_splits: int, block_dv: int) -> tuple[int, int]:
    """(splits merged per step, num_warps) for stage 2; the tile is at most the split cap rounded up to a power of two."""
    tile = min(triton.next_power_of_2(max_kv_splits), max(4, _STAGE2_TILE_ELEMS // block_dv), 64)
    return tile, max(1, min(8, tile * block_dv // 2048))


def _quantized_decode_launch(block_d: int, block_n: int, num_stages: int) -> tuple[int, int, int]:
    """(block_n, num_stages, num_warps) of stage 1 for quantized K/V, whose decode arms raise register use."""
    if block_d >= 512:
        return min(block_n, 32), num_stages, 8
    if block_d >= 256:
        # 3 stages also for a one-tile window: 2 stages spill in more configs
        return (block_n, 3, 8) if block_n == 64 else (block_n, 2, 4)
    if block_d >= 128 and block_n == 64:
        return block_n, num_stages, 8
    return block_n, num_stages, 4


def _launch_scales(kv_quant, kv_format: KVFormat, k_cache, v_cache, k_scales, v_scales, k_scale, v_scale, placeholder):
    """Scale arguments of one launch: ``(k ptr, v ptr, stride_kss, stride_ksh, stride_vss, stride_vsh, QBITS, QBLOCK)``.

    --kv-cache-dtype has two families that share the kernels' scale arguments, and at most one is on:
    Kai's q8_0 / q4_0 (``kv_quant``: uint8 code slabs, fp16 block scales indexed ``[slot, head, block]``)
    and upstream's per-token-head formats (``kv_format``: one scale per ``[slot, head]``; its kernel arms
    step heads by 1 and never read ``stride_ksh``).
    """
    if kv_quant is not None:
        assert not kv_format.quantized, "--kv-cache-dtype names one format"
        assert k_scales is not None and v_scales is not None, "quantized slab without scales"
        return (
            k_scales, v_scales,
            k_scales.stride(0), k_scales.stride(1), v_scales.stride(0), v_scales.stride(1),
            kv_quant.bits, kv_quant.block,
        )
    ks, vs, s_kss, s_vss = kv_scale_args(kv_format, k_cache, v_cache, k_scale, v_scale, placeholder)
    return ks, vs, s_kss, 1, s_vss, 1, 0, 1


def _max_request_splits(keys: int) -> int:
    return max(triton.cdiv(keys, _MIN_SPLIT_LEN), min(_MIN_SPLITS, triton.cdiv(keys, _MIN_BLOCK_KV)), 1)


@functools.lru_cache(maxsize=None)
def _optin_smem_bytes(device_index: int) -> int:
    """Per-block opt-in shared-memory budget for a CUDA device (0 if unavailable)."""
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "shared_memory_per_block_optin", 0))


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def decode_kv_split_cap(batch: int, num_q_heads: int, num_kv_heads: int, device_index: int) -> int:
    """Most KV splits per request for a decode batch: about four programs per SM, within [_MIN_SPLITS, _MAX_KV_SPLITS]."""
    head_blocks = triton.cdiv(num_q_heads, min(16, max(1, num_q_heads // num_kv_heads)))
    return max(_MIN_SPLITS, min(_MAX_KV_SPLITS, triton.cdiv(4 * _num_sms(device_index), batch * head_blocks)))


def _select_extend_tile(
    head_dim: int, block_d: int, smem_optin: int, split: bool = False
) -> tuple[int, int]:
    """Pick ``(BLOCK_M, BLOCK_N)`` for the extend/prefill kernel, shared-memory aware.

    Larger tiles run materially faster (~2x for head_dim 512 on H100) but their bf16
    q/k/v tiles need about ``(BLOCK_M + 2 * BLOCK_N) * BLOCK_D * 2`` bytes of shared
    memory, which overflows consumer GPUs (sm_89 ~99KB opt-in) once head_dim >= 256.
    Keep the fast tiles where the device's opt-in shared memory fits them (datacenter
    A100/H100); shrink only where it does not. ``smem_optin == 0`` (unknown) conservatively
    selects the small tiles, i.e. the prior consumer-safe behavior.

    ``split``: the extend kernel that also reads the cached prefix stages K/V tiles for
    both sources, i.e. ``(BLOCK_M + 4 * BLOCK_N)`` rows instead of ``(BLOCK_M + 2 * BLOCK_N)``.
    Whatever the preference above picked is finally halved until it fits the device's hard
    opt-in limit (Turing: 64 KB), since Triton refuses to launch a kernel whose static shared
    memory exceeds it.
    """
    budget = smem_optin * 0.8  # headroom for scores/acc/alignment/triton scratch

    def fits(block_m: int, block_n: int) -> bool:
        return (block_m + 2 * block_n) * block_d * 2 <= budget

    if head_dim <= 128:
        block_m, block_n = 128, 64
    elif head_dim <= 256:
        block_m, block_n = (128, 64) if fits(128, 64) else (64, 32)
    elif head_dim <= 384:
        block_m, block_n = (32, 64) if fits(32, 64) else (32, 32)
    else:
        block_m, block_n = (32, 64) if fits(32, 64) else (16, 16)
    if smem_optin > 0:
        kv_tiles = 4 if split else 2
        while (block_m + kv_tiles * block_n) * block_d * 2 > smem_optin and (
            block_m > 16 or block_n > 16
        ):
            block_m, block_n = max(16, block_m // 2), max(16, block_n // 2)
    return block_m, block_n


def _quantized_extend_tile(head_dim: int, block_d: int, has_blocks: bool) -> tuple[int, int]:
    """``(BLOCK_M, BLOCK_N)`` for quantized K/V: the small-SMEM tiles on every GPU; multimodal spans at head_dim 256 need a smaller one."""
    if has_blocks and 128 < head_dim <= 256:
        return 32, 16
    return _select_extend_tile(head_dim, block_d, 0)


@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    # Kai's block scales (QBITS > 0) or upstream's row scales (KV_FMT != "none"); never both
    k_scale_ptr,
    v_scale_ptr,
    o_ptr,
    indptr_ptr,
    indices_ptr,
    q_to_req_ptr,
    q_pos_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    QBITS: tl.constexpr,
    QBLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    KV_OFFSET_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    q_tok = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP

    req = tl.load(q_to_req_ptr + q_tok)
    kv_start = tl.load(indptr_ptr + req)
    kv_end = tl.load(indptr_ptr + req + 1)
    kv_len = kv_end - kv_start
    q_pos = tl.load(q_pos_ptr + q_tok)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q = tl.load(
        q_ptr + q_tok * stride_qt + q_head * stride_qh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in range(0, kv_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        k_pos = offs_n
        causal_mask = k_pos <= q_pos
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((k_pos + SLIDING_WINDOW) > q_pos)
        mask_n = mask_n & causal_mask

        skip_tile = tl.max(mask_n.to(tl.int32), axis=0) == 0
        if not skip_tile:
            slots = tl.load(indices_ptr + kv_start + offs_n, mask=offs_n < kv_len, other=0).to(KV_OFFSET_DTYPE)
            if QBITS == 0:
                k = kv_tile(
                    k_ptr
                    + slots[:, None] * stride_ks
                    + kv_head * stride_kh
                    + offs_d[None, :],
                    (offs_n[:, None] < kv_len) & mask_d[None, :],
                    KV_FMT,
                ).to(tl.float32)
                scale_k = kv_row_scale(k_scale_ptr + slots * stride_kss + kv_head, offs_n < kv_len, KV_FMT)
                scores = tl.sum(q[None, :] * k, axis=1) * sm_scale * scale_k
            else:
                byte_d = offs_d // 2 if QBITS == 4 else offs_d
                k = _dequant_tile(
                    k_ptr,
                    slots[:, None] * stride_ks + kv_head * stride_kh + byte_d[None, :],
                    k_scale_ptr,
                    slots[:, None] * stride_kss
                    + kv_head * stride_ksh
                    + (offs_d // QBLOCK)[None, :],
                    ((offs_d % 2) == 1)[None, :],
                    (offs_n[:, None] < kv_len) & mask_d[None, :],
                    QBITS,
                )
                scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
            scores = tl.where(mask_n, scores, -float("inf"))

            row_max = tl.max(scores, axis=0)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            if QBITS == 0:
                v = kv_tile(
                    v_ptr
                    + slots[:, None] * stride_vs
                    + kv_head * stride_vh
                    + offs_d[None, :],
                    (offs_n[:, None] < kv_len) & mask_d[None, :],
                    KV_FMT,
                ).to(tl.float32)
                scale_v = kv_row_scale(v_scale_ptr + slots * stride_vss + kv_head, offs_n < kv_len, KV_FMT)
                acc = acc * alpha + tl.sum((p * scale_v)[:, None] * v, axis=0)
            else:
                byte_d = offs_d // 2 if QBITS == 4 else offs_d
                v = _dequant_tile(
                    v_ptr,
                    slots[:, None] * stride_vs + kv_head * stride_vh + byte_d[None, :],
                    v_scale_ptr,
                    slots[:, None] * stride_vss
                    + kv_head * stride_vsh
                    + (offs_d // QBLOCK)[None, :],
                    ((offs_d % 2) == 1)[None, :],
                    (offs_n[:, None] < kv_len) & mask_d[None, :],
                    QBITS,
                )
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + q_tok * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


@triton.jit
def _dequant_tile(
    code_ptr,
    code_off,
    scale_ptr,
    scale_off,
    nib_hi,
    mask,
    QBITS: tl.constexpr,
):
    """One K or V tile read back out of the block-quantized slabs (``--kv-cache-dtype``).

    The caller owns the offsets because K arrives transposed ([D, N]) and V does not
    ([N, DV]); everything that differs between the two is in how those are broadcast, and
    everything that differs between 4- and 8-bit is here. ``other=0`` on both loads makes a
    masked lane dequantize to a clean 0 rather than to garbage times a stale scale.
    """
    b = tl.load(code_ptr + code_off, mask=mask, other=0).to(tl.int32)
    if QBITS == 8:
        code = tl.where(b > 127, b - 256, b)
    else:
        nib = tl.where(nib_hi, (b >> 4) & 0xF, b & 0xF)
        code = tl.where(nib > 7, nib - 16, nib)
    s = tl.load(scale_ptr + scale_off, mask=mask, other=0.0).to(tl.float32)
    return code.to(tl.float32) * s


@triton.jit
def _decode_num_splits(effective_len, max_kv_splits, MIN_SPLIT_LEN: tl.constexpr, MIN_SPLITS: tl.constexpr, MIN_BLOCK_KV: tl.constexpr):
    # same rule as _max_request_splits: long requests split by length, short ones still take up to MIN_SPLITS small splits
    by_len = tl.maximum(tl.cdiv(effective_len, MIN_SPLIT_LEN), tl.minimum(MIN_SPLITS, tl.cdiv(effective_len, MIN_BLOCK_KV)))
    return tl.minimum(tl.maximum(by_len, 1), max_kv_splits)


@triton.jit(do_not_specialize=["max_kv_splits"])
def _decode_grouped_stage1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_scale_ptr,
    v_scale_ptr,
    sm_scale,
    indptr_ptr,
    indices_ptr,
    q_pos_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    max_kv_splits,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    QBITS: tl.constexpr,
    QBLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    KV_OFFSET_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    split_id = tl.program_id(2)

    # VALID_BLOCK_H == min(cap, GROUP) is the number of query heads actually handled per program;
    # BLOCK_H is the power-of-two tile size for the head axis (tl.arange requires a power of two),
    # so a non-power-of-two GQA group (e.g. 24/4 == 6) rounds the tile up and masks the extra
    # lanes. Each kv head spans cdiv(GROUP, VALID_BLOCK_H) head blocks.
    kv_head = head_block_id // tl.cdiv(GROUP, VALID_BLOCK_H)
    q_heads = head_block_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = q_heads < (head_block_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (q_heads < NUM_Q_HEADS)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < D
    mask_dv = offs_dv < DV

    kv_start = tl.load(indptr_ptr + batch_id)
    kv_len = tl.load(indptr_ptr + batch_id + 1) - kv_start
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_splits = _decode_num_splits(effective_len, max_kv_splits, MIN_SPLIT_LEN, MIN_SPLITS, MIN_BLOCK_KV)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_start = kv_len_per_split * split_id
    split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

    m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    q_offsets = batch_id * stride_qt + q_heads[:, None] * stride_qh + offs_d[None, :]
    if QBITS == 0:
        k_base_offsets = kv_head * stride_kh + offs_d[:, None]
        v_base_offsets = kv_head * stride_vh + offs_dv[None, :]
    else:
        # code byte per element: 8-bit is one byte each, 4-bit packs a pair low nibble first
        byte_d = offs_d // 2 if QBITS == 4 else offs_d
        byte_dv = offs_dv // 2 if QBITS == 4 else offs_dv
        k_base_offsets = kv_head * stride_kh + byte_d[:, None]
        v_base_offsets = kv_head * stride_vh + byte_dv[None, :]
        k_scale_base = kv_head * stride_ksh + (offs_d // QBLOCK)[:, None]
        v_scale_base = kv_head * stride_vsh + (offs_dv // QBLOCK)[None, :]
        k_nib_hi = ((offs_d % 2) == 1)[:, None]
        v_nib_hi = ((offs_dv % 2) == 1)[None, :]

    if split_end > split_start:
        q = tl.load(q_ptr + q_offsets, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        if QBITS == 0:
            if KV_FMT == "none":
                q = q.to(k_ptr.dtype.element_ty)

        for rel_start in tl.range(split_start, split_end, BLOCK_N):
            rel_offs = rel_start + tl.arange(0, BLOCK_N)
            mask_n = rel_offs < split_end
            logical_offs = effective_start + rel_offs
            slots = tl.load(indices_ptr + kv_start + logical_offs, mask=mask_n, other=0).to(KV_OFFSET_DTYPE)

            if QBITS == 0:
                k = kv_tile(k_ptr + slots[None, :] * stride_ks + k_base_offsets, mask_n[None, :] & mask_d[:, None], KV_FMT)
                scale_k = kv_row_scale(k_scale_ptr + slots * stride_kss + kv_head, mask_n, KV_FMT)
                scores = tl.dot(q, k) * sm_scale * scale_k[None, :]
            else:
                k = _dequant_tile(
                    k_ptr,
                    slots[None, :] * stride_ks + k_base_offsets,
                    k_scale_ptr,
                    slots[None, :] * stride_kss + k_scale_base,
                    k_nib_hi,
                    mask_n[None, :] & mask_d[:, None],
                    QBITS,
                ).to(q.dtype)
                scores = tl.dot(q, k) * sm_scale
            scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))

            if QBITS == 0:
                v = kv_tile(v_ptr + slots[:, None] * stride_vs + v_base_offsets, mask_n[:, None] & mask_dv[None, :], KV_FMT)
                scale_v = kv_row_scale(v_scale_ptr + slots * stride_vss + kv_head, mask_n, KV_FMT)
            else:
                v = _dequant_tile(
                    v_ptr,
                    slots[:, None] * stride_vs + v_base_offsets,
                    v_scale_ptr,
                    slots[:, None] * stride_vss + v_scale_base,
                    v_nib_hi,
                    mask_n[:, None] & mask_dv[None, :],
                    QBITS,
                ).to(q.dtype)

            m_new = tl.maximum(tl.max(scores, axis=1), m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            if QBITS == 0:
                acc = acc * alpha[:, None] + tl.dot((p * scale_v[None, :]).to(v.dtype), v)
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        mid_offsets = (
            batch_id * stride_mid_ob
            + q_heads[:, None] * stride_mid_oh
            + split_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(mid_o_ptr + mid_offsets, out, mask=mask_h[:, None] & mask_dv[None, :])

        lse_offsets = (
            batch_id * stride_lse_b
            + q_heads * stride_lse_h
            + split_id * stride_lse_s
        )
        tl.store(mid_lse_ptr + lse_offsets, m_i + tl.log(l_i), mask=mask_h)


@triton.jit(do_not_specialize=["max_kv_splits"])
def _decode_stage2_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    o_ptr,
    indptr_ptr,
    q_pos_ptr,
    max_kv_splits,
    sinks_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ot,
    stride_oh,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head = tl.program_id(1)

    kv_len = tl.load(indptr_ptr + batch_id + 1) - tl.load(indptr_ptr + batch_id)
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_splits = _decode_num_splits(effective_len, max_kv_splits, MIN_SPLIT_LEN, MIN_SPLITS, MIN_BLOCK_KV)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < DV
    offs_s = tl.arange(0, SPLIT_TILE)
    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    mid_base = batch_id * stride_mid_ob + q_head * stride_mid_oh
    lse_base = batch_id * stride_lse_b + q_head * stride_lse_h

    # one load round trip per SPLIT_TILE splits
    for s0 in range(0, kv_splits, SPLIT_TILE):
        split_id = s0 + offs_s
        valid = (split_id < kv_splits) & (kv_len_per_split * split_id < effective_len)
        partial_lse = tl.load(mid_lse_ptr + lse_base + split_id * stride_lse_s, mask=valid, other=-float("inf"))
        partial = tl.load(
            mid_o_ptr + mid_base + split_id[:, None] * stride_mid_os + offs_d[None, :],
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        m_new = tl.maximum(m_i, tl.max(partial_lse, axis=0))
        # a tile of empty splits with no sink leaves m at -inf, where exp(-inf - -inf) is nan
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        beta = tl.exp(partial_lse - m_safe)
        acc = acc * alpha + tl.sum(partial * beta[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(beta, axis=0)
        m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + batch_id * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


def decode_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_positions: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    kv_quant=None,
    max_kv_splits: int | None = None,
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """SGLang-style split-k grouped decode attention for one query per request.

    Each request takes ``max(ceil(keys / 128), min(8, ceil(keys / 32)))`` splits, capped by ``max_kv_splits`` (default :func:`decode_kv_split_cap`) and by the split dim of ``attn_logits`` / ``attn_lse``.
    The kernels compute the split count on the device, so a captured graph serves any context length.

    With ``kv_quant`` set (``--kv-cache-dtype``), ``k_cache``/``v_cache`` are uint8 code
    slabs and ``k_scales``/``v_scales`` their fp16 block scales; the tiles are dequantized
    inside the kernel, so nothing downstream of this call changes."""

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert batch == indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    if kv_quant is None:
        assert k_cache.shape[-1] == head_dim and v_cache.shape[-1] == head_dim
    else:
        assert k_scales is not None and v_scales is not None, "quantized slab without scales"
        want = kv_quant.code_bytes_per_row(head_dim)
        assert k_cache.shape[-1] == want and v_cache.shape[-1] == want
        assert k_scales.shape[-1] == kv_quant.blocks_per_row(head_dim)
    assert num_q_heads % num_kv_heads == 0
    assert attn_logits.shape[0] >= batch
    assert attn_logits.shape[1] >= num_q_heads
    assert attn_logits.shape[3] >= head_dim
    assert attn_lse.shape[0] >= batch
    assert attn_lse.shape[1] >= num_q_heads
    if sliding_window is not None and sliding_window <= 0:
        # the kernels treat a window <= 0 as no window
        sliding_window = None
    if max_kv_splits is None:
        max_kv_splits = decode_kv_split_cap(batch, num_q_heads, num_kv_heads, q.device.index)
    if sliding_window:
        # no request attends more than the window, so programs past its splits would only exit
        max_kv_splits = min(max_kv_splits, _max_request_splits(sliding_window))
    max_kv_splits = min(max_kv_splits, attn_logits.shape[2], attn_lse.shape[2])
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    group = num_q_heads // num_kv_heads
    # valid_block_h = heads computed per program (drives the grid + head indexing); block_h =
    # power-of-two tile size for tl.arange. They differ only for non-power-of-two GQA groups
    # (e.g. 6), where block_h rounds up and the kernel masks the extra lanes.
    valid_block_h = min(16, group)
    block_h = triton.next_power_of_2(valid_block_h)
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)
    grid = (batch, triton.cdiv(num_q_heads, valid_block_h), max_kv_splits)
    # a grid that fits in one wave runs about one program per SM, which then needs wider tiles to keep enough loads in flight
    block_n = 64 if grid[0] * grid[1] * grid[2] <= _num_sms(q.device.index) else _MIN_BLOCK_KV
    num_stages = 3
    # stage 1 keeps the Q tile and one K and one V tile in shared memory, in the dot's dtype: a
    # quantized slab (--kv-cache-dtype, 1-byte codes) is dequantized to q's dtype before the dot,
    # so its element size says nothing about the tiles
    smem = _optin_smem_bytes(q.device.index)
    elem = q.element_size()
    tile_key = (q.device.index, 0 if kv_quant is None else kv_quant.bits, kv_format.name, head_dim, q.dtype)
    if 2 * block_n * block_d * elem > 0.8 * smem or tile_key in _WIDE_TILE_REFUSED:
        block_n = _MIN_BLOCK_KV
    # 64 KiB parts (sm_75) cannot fit a head_dim 512 Q tile next to 32-row K/V tiles:
    # halve the KV tile down to tl.dot's 16-row minimum until the three tiles fit.
    while smem and block_n > 16 and (block_h * block_d + 2 * block_n * block_d) * elem > smem:
        block_n //= 2
    if sliding_window:
        split_len = triton.cdiv(triton.cdiv(sliding_window, max_kv_splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
        if split_len <= block_n:
            # a full window's split fits in one tile, where a deeper pipeline only adds overhead
            block_n, num_stages = split_len, 2
    num_warps = 4
    if kv_format.quantized:
        block_n, num_stages, num_warps = _quantized_decode_launch(block_d, block_n, num_stages)

    ks_arg, vs_arg, s_kss, s_ksh, s_vss, s_vsh, qbits, qblock = _launch_scales(
        kv_quant, kv_format, k_cache, v_cache, k_scales, v_scales, k_scale, v_scale, q
    )

    def stage1(block_n: int, num_stages: int, num_warps: int) -> None:
        _decode_grouped_stage1_kernel[grid](
            q,
            k_cache,
            v_cache,
            ks_arg,
            vs_arg,
            sm_scale,
            indptr,
            indices,
            q_positions,
            attn_logits,
            attn_lse,
            max_kv_splits,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            v_cache.stride(0),
            v_cache.stride(1),
            s_kss,
            s_ksh,
            s_vss,
            s_vsh,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            attn_lse.stride(0),
            attn_lse.stride(1),
            attn_lse.stride(2),
            QBITS=qbits,
            QBLOCK=qblock,
            KV_OFFSET_DTYPE=_kv_offset_dtype(k_cache, v_cache),
            KV_FMT=kv_format.name,
            GROUP=group,
            NUM_Q_HEADS=num_q_heads,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            VALID_BLOCK_H=valid_block_h,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MIN_SPLIT_LEN=_MIN_SPLIT_LEN,
            MIN_SPLITS=_MIN_SPLITS,
            D=head_dim,
            DV=head_dim,
            SLIDING_WINDOW=sliding_window or 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    try:
        stage1(block_n, num_stages, num_warps)
    except OutOfResources:
        # the estimate above is of the tiles alone; the dequantizing kernel keeps more (int32 codes,
        # fp32 products), and Triton refuses the launch while loading the module, so nothing has run.
        # Step down to the narrow tile and remember it for this shape (one exception per shape).
        if block_n <= _MIN_BLOCK_KV:
            raise
        _WIDE_TILE_REFUSED.add(tile_key)
        stage1(_MIN_BLOCK_KV, 3, num_warps)
    split_tile, stage2_warps = _stage2_launch(max_kv_splits, block_dv)
    _decode_stage2_kernel[(batch, num_q_heads)](
        attn_logits,
        attn_lse,
        o,
        indptr,
        q_positions,
        max_kv_splits,
        sinks_arg,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        o.stride(0),
        o.stride(1),
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        MIN_SPLIT_LEN=_MIN_SPLIT_LEN,
        MIN_SPLITS=_MIN_SPLITS,
        BLOCK_DV=block_dv,
        DV=head_dim,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        SPLIT_TILE=split_tile,
        num_warps=stage2_warps,
    )
    return o


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_scale_ptr,
    v_scale_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    block_ends_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    QBITS: tl.constexpr,
    QBLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    HAS_BLOCKS: tl.constexpr,
    KV_OFFSET_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_end = tl.load(qo_indptr_ptr + seq_id + 1)
    q_len = q_end - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    kv_len = tl.load(kv_indptr_ptr + seq_id + 1) - kv_start
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    q_abs_pos = prefix_len + offs_m
    block_q_end = tl.minimum(q_len, (block_m_id + 1) * BLOCK_M)
    kv_loop_end = tl.minimum(kv_len, prefix_len + block_q_end)
    if HAS_BLOCKS:
        # rows inside a multimodal span also attend forward to the span's later keys: the tile loop must reach them
        block_end = tl.load(block_ends_ptr + q_start + offs_m, mask=mask_m, other=0)
        kv_loop_end = tl.minimum(kv_len, tl.maximum(kv_loop_end, tl.max(block_end, axis=0)))
    else:
        block_end = tl.zeros((BLOCK_M,), dtype=tl.int32)

    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_qt + q_head * stride_qh + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32)
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    for start_n in tl.range(0, kv_loop_end, BLOCK_N):
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < kv_len
        key_pos = kv_offsets
        causal_mask = key_pos[None, :] <= q_abs_pos[:, None]
        if HAS_BLOCKS:
            causal_mask = causal_mask | (key_pos[None, :] < block_end[:, None])
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((key_pos[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None])
        final_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not skip_tile:
            slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0).to(KV_OFFSET_DTYPE)
            if QBITS == 0:
                k = kv_tile(
                    k_ptr
                    + slots[None, :] * stride_ks
                    + kv_head * stride_kh
                    + offs_d[:, None],
                    mask_n[None, :] & mask_d[:, None],
                    KV_FMT,
                )
                scale_k = kv_row_scale(k_scale_ptr + slots * stride_kss + kv_head, mask_n, KV_FMT)
                scores = tl.dot(q.to(k.dtype), k) * sm_scale * scale_k[None, :]
            else:
                byte_d = offs_d // 2 if QBITS == 4 else offs_d
                k = _dequant_tile(
                    k_ptr,
                    slots[None, :] * stride_ks + kv_head * stride_kh + byte_d[:, None],
                    k_scale_ptr,
                    slots[None, :] * stride_kss
                    + kv_head * stride_ksh
                    + (offs_d // QBLOCK)[:, None],
                    ((offs_d % 2) == 1)[:, None],
                    mask_n[None, :] & mask_d[:, None],
                    QBITS,
                ).to(q.dtype)
                scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            if QBITS == 0:
                v = kv_tile(
                    v_ptr
                    + slots[:, None] * stride_vs
                    + kv_head * stride_vh
                    + offs_dv[None, :],
                    mask_n[:, None] & mask_dv[None, :],
                    KV_FMT,
                )
                scale_v = kv_row_scale(v_scale_ptr + slots * stride_vss + kv_head, mask_n, KV_FMT)
                acc = acc * alpha[:, None] + tl.dot((p * scale_v[None, :]).to(v.dtype), v)
            else:
                byte_dv = offs_dv // 2 if QBITS == 4 else offs_dv
                v = _dequant_tile(
                    v_ptr,
                    slots[:, None] * stride_vs + kv_head * stride_vh + byte_dv[None, :],
                    v_scale_ptr,
                    slots[:, None] * stride_vss
                    + kv_head * stride_vsh
                    + (offs_dv // QBLOCK)[None, :],
                    ((offs_dv % 2) == 1)[None, :],
                    mask_n[:, None] & mask_dv[None, :],
                    QBITS,
                ).to(q.dtype)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


# ---------------------------------------------------------------------------------------
# Prefill attention through cuBLAS, for cards the fused kernel does not fit
# ---------------------------------------------------------------------------------------
# The fused kernel stages q and four k/v tiles in shared memory, so its tile is bounded by
# what the device has: at head_dim 256 a Turing card (64 KB) is cut to (32, 16) and runs the
# prefill at 0.47 TFLOPS, while the same card does 19 TFLOPS of fp16 cuBLAS. This path spends
# memory to get that back -- materialize the scores, softmax them, matmul again -- which is
# what --dense-quant fp8 and the NVFP4 MoE already do on this hardware.
#
# Measured on an RTX 2060 (Ornith: 16 q heads, 2 kv heads, head_dim 256), one layer, 2048 new
# tokens behind an 8192-token prefix: 658 ms fused -> 51 ms here, output bit-identical.
_ATTN_SCRATCH_ENV = "FREETOKEN_ATTN_SCRATCH"
_ATTN_SCRATCH_MB_ENV = "FREETOKEN_ATTN_SCRATCH_MB"
# The score tile is the knob. Everything else this path allocates is fixed by the request.
_ATTN_SCRATCH_MB = 48
# Fewest rows this path takes. It was 128 on the reading that thin GEMMs lose to the fused
# kernel for a verify window or a chat turn behind a cached prefix; measured on the RTX 2060
# (Ornith's shape, fp16) this path is faster at every row count from 1 up, and more so the
# longer the prefix: 1 row behind 16k 1.4 ms against 26.7 ms, 64 rows 3.9 against 53.9 ms
# (guides/26 §10). FREETOKEN_ATTN_SCRATCH_MIN_ROWS overrides it.
_ATTN_SCRATCH_MIN_ROWS = 1
# The gathered contiguous K/V is the one allocation not bounded by the score budget -- it is
# the whole context. Past this multiple of the budget, leave it to the fused kernel, whose
# footprint is its tile.
_ATTN_SCRATCH_GATHER_LIMIT = 4


def _attn_scratch_bytes() -> int:
    return max(1, int(os.getenv(_ATTN_SCRATCH_MB_ENV) or _ATTN_SCRATCH_MB)) << 20


@functools.lru_cache(maxsize=1)
def _attn_scratch_default() -> bool:
    """On by default where the fused tile had to be cut: pre-Ampere (64 KB of shared memory).

    Ampere and later fit (128, 64) and run the fused kernel at rates this path cannot reach,
    so they keep it. ``FREETOKEN_ATTN_SCRATCH=1`` forces this path on for a card that was not
    measured, ``=0`` forces it off.
    """
    from freetoken.utils.arch import is_pre_ampere

    return is_pre_ampere()


def _scratch_attention_applies(
    q: torch.Tensor,
    k_extend: torch.Tensor | None,
    v_extend: torch.Tensor | None,
    sliding_window: int | None,
    sinks: torch.Tensor | None,
    kv_quant,
    num_rows: int,
    kv_len: int,
    head_dim: int,
    num_kv_heads: int,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
) -> bool:
    """Whether the cuBLAS path can serve this call.

    Everything it cannot express falls through to the fused kernel rather than growing a
    second implementation of it: a sliding window, attention sinks, and the non-split call
    where this forward's own K/V are already in the cache. A quantized cache
    (``--kv-cache-dtype``) IS served here -- the prefix is decoded as part of the gather.
    """
    forced = os.getenv(_ATTN_SCRATCH_ENV)
    if forced is not None:
        if forced.strip() not in ("1", "true", "yes", "on"):
            return False
    elif not _attn_scratch_default():
        return False
    if k_extend is None or v_extend is None:
        return False
    if kv_quant is not None and (k_scales is None or v_scales is None):
        return False        # a quantized slab without its scales cannot be decoded
    if sliding_window or sinks is not None:
        return False
    if num_rows < int(os.getenv("FREETOKEN_ATTN_SCRATCH_MIN_ROWS") or _ATTN_SCRATCH_MIN_ROWS):
        return False
    gather = 2 * num_kv_heads * kv_len * head_dim * q.element_size()
    return gather <= _ATTN_SCRATCH_GATHER_LIMIT * _attn_scratch_bytes()


def _extend_attention_scratch(
    q: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    sm_scale: float,
    kv_quant=None,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal extend attention as three cuBLAS calls per tile, over a materialized score tile.

    Per sequence and per KV head: gather the prefix rows out of the paged cache, append this
    forward K/V so the keys are one contiguous matrix, then walk the query rows in tiles whose
    score block fits ``FREETOKEN_ATTN_SCRATCH_MB``. The whole key range is scored at once, so
    the softmax is the plain one -- no running maximum, nothing to rescale.

    The mask is only ever needed on the last ``rows`` columns: every prefix key precedes every
    query row in this forward, and within the forward row i takes keys 0..i.
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    budget = _attn_scratch_bytes()
    # one .tolist() per call, not one .item() per index: these are tiny host-side loop bounds
    starts = qo_indptr.tolist()
    kv_starts = kv_indptr.tolist()
    prefixes = prefix_lens.tolist()

    for seq, prefix in enumerate(prefixes):
        q0, q1 = starts[seq], starts[seq + 1]
        rows = q1 - q0
        if rows <= 0:
            continue
        kv_len = prefix + rows
        # bytes one query row costs in the score block: the fp16 scores, the fp32 softmax
        # torch materializes, and the cast back
        per_row = group * kv_len * (2 * q.element_size() + 4)
        block = max(16, min(rows, int(budget // max(1, per_row))))
        cols = torch.arange(rows, device=q.device)
        slots = kv_indices[kv_starts[seq] : kv_starts[seq] + prefix].long()
        for head in range(num_kv_heads):
            prefix_k, prefix_v = k_cache[slots, head], v_cache[slots, head]
            if kv_quant is not None:
                # The slabs hold packed codes; decode the rows this sequence actually reads,
                # once, into the gather that was going to be built anyway. The result is the
                # same fp16 matrix the unquantized path produces, so nothing downstream
                # changes -- and the transient does not grow either.
                from freetoken.kvcache.kv_quant import dequantize_rows

                prefix_k = dequantize_rows(prefix_k, k_scales[slots, head], kv_quant, q.dtype)
                prefix_v = dequantize_rows(prefix_v, v_scales[slots, head], kv_quant, q.dtype)
            keys = torch.cat([prefix_k, k_extend[q0:q1, head]]).transpose(0, 1)
            values = torch.cat([prefix_v, v_extend[q0:q1, head]])
            lo, hi = head * group, (head + 1) * group
            for start in range(0, rows, block):
                end = min(start + block, rows)
                # [group, tile, head_dim] @ [head_dim, kv_len] -> [group, tile, kv_len]
                scores = torch.matmul(q[q0 + start : q0 + end, lo:hi].permute(1, 0, 2), keys)
                scores.mul_(sm_scale)
                future = cols[None, :] > cols[start:end, None]
                scores[:, :, prefix:].masked_fill_(future, -float("inf"))
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                o[q0 + start : q0 + end, lo:hi] = torch.matmul(probs, values).permute(1, 0, 2)
    return o


@triton.jit
def _extend_attend_tile(
    q, m_i, l_i, acc, qk_scale, k_base, v_base, k_rows, v_rows, k_scales, v_scales, offs_d, offs_dv, mask_d, mask_dv, mask_n, mask,
    MASKED: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    if MASKED:
        k_mask = mask_n[None, :] & mask_d[:, None]
        v_mask = mask_n[:, None] & mask_dv[None, :]
        scale_mask = mask_n
    else:
        k_mask = mask_d[:, None]
        v_mask = mask_dv[None, :]
        scale_mask = tl.full(k_rows.shape, 1, tl.int1)
    # scales first: loading them after the staged decode ran slower at head_dim 128
    scale_k = kv_row_scale(k_scales, scale_mask, KV_FMT)
    k = kv_tile_staged(k_base + k_rows[None, :] + offs_d[:, None], k_mask, KV_FMT)
    scores = tl.dot(q.to(k.dtype), k) * scale_k[None, :]
    if MASKED:
        scores = tl.where(mask, scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1) * qk_scale)
    if MASKED:
        m_new = tl.where(m_new == -float("inf"), -1e20, m_new)
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(scores * qk_scale - m_new[:, None])
    scale_v = kv_row_scale(v_scales, scale_mask, KV_FMT)
    v = kv_tile_staged(v_base + v_rows[:, None] + offs_dv[None, :], v_mask, KV_FMT)
    acc = acc * alpha[:, None] + tl.dot((p * scale_v[None, :]).to(v.dtype), v)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return m_new, l_i, acc


@triton.jit
def _extend_attend_tile_q(
    q, m_i, l_i, acc, qk_scale, k_cache_ptr, v_cache_ptr, ks_ptr, vs_ptr, slots, kv_head,
    stride_kcs, stride_kch, stride_vcs, stride_vch, stride_kss, stride_ksh, stride_vss, stride_vsh,
    offs_d, offs_dv, mask_d, mask_dv, mask_n, mask,
    QBITS: tl.constexpr, QBLOCK: tl.constexpr, MASKED: tl.constexpr,
):
    """``_extend_attend_tile`` for a block-quantized prefix (``--kv-cache-dtype``): the K/V
    tiles are read back out of the uint8 slabs through ``_dequant_tile``; the online-softmax
    step is the same. Only the cached prefix is quantized, so the extend-side tiles keep the
    plain helper."""
    if MASKED:
        k_mask = mask_n[None, :] & mask_d[:, None]
        v_mask = mask_n[:, None] & mask_dv[None, :]
    else:
        k_mask = mask_d[:, None]
        v_mask = mask_dv[None, :]
    byte_d = offs_d // 2 if QBITS == 4 else offs_d
    k = _dequant_tile(
        k_cache_ptr,
        slots[None, :] * stride_kcs + kv_head * stride_kch + byte_d[:, None],
        ks_ptr,
        slots[None, :] * stride_kss + kv_head * stride_ksh + (offs_d // QBLOCK)[:, None],
        ((offs_d % 2) == 1)[:, None],
        k_mask,
        QBITS,
    ).to(q.dtype)
    scores = tl.dot(q, k)
    if MASKED:
        scores = tl.where(mask, scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1) * qk_scale)
    if MASKED:
        m_new = tl.where(m_new == -float("inf"), -1e20, m_new)
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(scores * qk_scale - m_new[:, None])
    byte_dv = offs_dv // 2 if QBITS == 4 else offs_dv
    v = _dequant_tile(
        v_cache_ptr,
        slots[:, None] * stride_vcs + kv_head * stride_vch + byte_dv[None, :],
        vs_ptr,
        slots[:, None] * stride_vss + kv_head * stride_vsh + (offs_dv // QBLOCK)[None, :],
        ((offs_dv % 2) == 1)[None, :],
        v_mask,
        QBITS,
    ).to(q.dtype)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return m_new, l_i, acc


@triton.jit
def _extend_tile_ranges(
    first_row, diag, end, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr
):
    """Split keys [0, end) seen by rows first_row + [0, BLOCK_M) into skipped [0, lo), masked [lo, u0), unmasked [u0, u1) and masked [u1, end); keys below diag are causal for every row."""
    if SLIDING_WINDOW > 0:
        lo = tl.minimum(tl.maximum(first_row - SLIDING_WINDOW + 1, 0) // BLOCK_N * BLOCK_N, end)
        full = tl.cdiv(tl.maximum(first_row + BLOCK_M - SLIDING_WINDOW, 0), BLOCK_N) * BLOCK_N
    else:
        lo = 0
        full = 0
    u1 = tl.maximum(diag, lo)
    u0 = tl.minimum(tl.maximum(full, lo), u1)
    return lo, u0, u1


@triton.jit
def _extend_attention_split_kernel(
    q_ptr,
    k_extend_ptr,
    v_extend_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    block_ends_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ket,
    stride_keh,
    stride_vet,
    stride_veh,
    stride_kcs,
    stride_kch,
    stride_vcs,
    stride_vch,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    QBITS: tl.constexpr,
    QBLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    HAS_BLOCKS: tl.constexpr,
    KV_OFFSET_DTYPE: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_len = tl.load(qo_indptr_ptr + seq_id + 1) - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    block_start = block_m_id * BLOCK_M
    offs_m = block_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    q_abs_pos = prefix_len + offs_m
    # Scores live in the log2 domain: exp2 lowers to ex2.approx.ftz, while tl.exp adds a denormal fixup per element.
    qk_scale = sm_scale * 1.4426950408889634

    q = tl.load(
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + q_head * stride_qh
        + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32) * 1.4426950408889634
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    kc_base = k_cache_ptr + kv_head * stride_kch
    vc_base = v_cache_ptr + kv_head * stride_vch
    ks_base = k_scale_ptr + kv_head
    vs_base = v_scale_ptr + kv_head
    prefix_end = tl.where(block_start < q_len, prefix_len, 0)
    lo, u0, u1 = _extend_tile_ranges(
        prefix_len + block_start, prefix_end // BLOCK_N * BLOCK_N, prefix_end, BLOCK_M, BLOCK_N, SLIDING_WINDOW
    )
    for start_n in tl.range(u0, u1, BLOCK_N):
        slots = tl.load(kv_indices_ptr + kv_start + start_n + offs_n).to(KV_OFFSET_DTYPE)
        if QBITS == 0:
            m_i, l_i, acc = _extend_attend_tile(
                q, m_i, l_i, acc, qk_scale, kc_base, vc_base, slots * stride_kcs, slots * stride_vcs,
                ks_base + slots * stride_kss, vs_base + slots * stride_vss,
                offs_d, offs_dv, mask_d, mask_dv, None, None, False, KV_FMT,
            )
        else:
            m_i, l_i, acc = _extend_attend_tile_q(
                q, m_i, l_i, acc, qk_scale, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, slots, kv_head,
                stride_kcs, stride_kch, stride_vcs, stride_vch, stride_kss, stride_ksh, stride_vss, stride_vsh,
                offs_d, offs_dv, mask_d, mask_dv, None, None, QBITS, QBLOCK, False,
            )
    n_left = (u0 - lo) // BLOCK_N
    for i in tl.range(0, n_left + tl.cdiv(prefix_end - u1, BLOCK_N)):
        kv_offsets = tl.where(i < n_left, lo + i * BLOCK_N, u1 + (i - n_left) * BLOCK_N) + offs_n
        mask_n = kv_offsets < prefix_end
        final_mask = mask_n[None, :]
        if SLIDING_WINDOW > 0:
            final_mask = final_mask & ((kv_offsets[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None])
        slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0).to(KV_OFFSET_DTYPE)
        if QBITS == 0:
            m_i, l_i, acc = _extend_attend_tile(
                q, m_i, l_i, acc, qk_scale, kc_base, vc_base, slots * stride_kcs, slots * stride_vcs,
                ks_base + slots * stride_kss, vs_base + slots * stride_vss,
                offs_d, offs_dv, mask_d, mask_dv, mask_n, final_mask, True, KV_FMT,
            )
        else:
            m_i, l_i, acc = _extend_attend_tile_q(
                q, m_i, l_i, acc, qk_scale, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, slots, kv_head,
                stride_kcs, stride_kch, stride_vcs, stride_vch, stride_kss, stride_ksh, stride_vss, stride_vsh,
                offs_d, offs_dv, mask_d, mask_dv, mask_n, final_mask, QBITS, QBLOCK, True,
            )

    # this chunk's K and V are raw compute-dtype tensors, not pool rows
    ke_base = k_extend_ptr + kv_head * stride_keh
    ve_base = v_extend_ptr + kv_head * stride_veh
    current_end = tl.minimum(q_len, block_start + BLOCK_M)
    if HAS_BLOCKS:
        # rows inside a multimodal span also attend forward to the span's later keys: the tile loop must reach them
        block_end = tl.load(block_ends_ptr + q_start + offs_m, mask=mask_m, other=0) - prefix_len
        current_end = tl.minimum(q_len, tl.maximum(current_end, tl.max(block_end, axis=0)))
    else:
        block_end = tl.zeros((BLOCK_M,), dtype=tl.int32)
    current_end = tl.where(block_start < q_len, current_end, 0)
    lo, u0, u1 = _extend_tile_ranges(
        block_start, tl.minimum(block_start // BLOCK_N * BLOCK_N, current_end), current_end, BLOCK_M, BLOCK_N,
        SLIDING_WINDOW,
    )
    for start_n in tl.range(u0, u1, BLOCK_N):
        rows = q_start + start_n + offs_n
        m_i, l_i, acc = _extend_attend_tile(
            q, m_i, l_i, acc, qk_scale, ke_base, ve_base, rows * stride_ket, rows * stride_vet, None, None,
            offs_d, offs_dv, mask_d, mask_dv, None, None, False, "none",
        )
    n_left = (u0 - lo) // BLOCK_N
    for i in tl.range(0, n_left + tl.cdiv(current_end - u1, BLOCK_N)):
        local_kv_offsets = tl.where(i < n_left, lo + i * BLOCK_N, u1 + (i - n_left) * BLOCK_N) + offs_n
        mask_n = local_kv_offsets < current_end
        local_q_pos = offs_m
        causal_mask = local_kv_offsets[None, :] <= local_q_pos[:, None]
        if HAS_BLOCKS:
            causal_mask = causal_mask | (local_kv_offsets[None, :] < block_end[:, None])
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & (
                (local_kv_offsets[None, :] + SLIDING_WINDOW) > local_q_pos[:, None]
            )
        final_mask = mask_n[None, :] & causal_mask
        rows = q_start + local_kv_offsets
        m_i, l_i, acc = _extend_attend_tile(
            q, m_i, l_i, acc, qk_scale, ke_base, ve_base, rows * stride_ket, rows * stride_vet, None, None,
            offs_d, offs_dv, mask_d, mask_dv, mask_n, final_mask, True, "none",
        )

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_extend: torch.Tensor | None = None,
    v_extend: torch.Tensor | None = None,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    kv_quant=None,
    block_ends: torch.Tensor | None = None,
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-tiled causal prefill/extend attention over paged KV cache; block_ends holds per query
    token the end of the multimodal span it sits in (0 for none), whose later keys the row also attends.

    ``kv_quant`` (``--kv-cache-dtype``) makes ``k_cache``/``v_cache`` uint8 code slabs read
    through ``k_scales``/``v_scales``. Only the PREFIX comes from the cache: this forward's
    own K/V arrive unquantized in ``k_extend``/``v_extend`` and stay that way, so the tokens
    being written this step are attended at full precision.
    """

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_q_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert qo_indptr.numel() == kv_indptr.numel()
    assert prefix_lens.numel() == qo_indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    if kv_quant is None:
        assert k_cache.shape[-1] == head_dim and v_cache.shape[-1] == head_dim
    else:
        assert k_scales is not None and v_scales is not None, "quantized slab without scales"
        want = kv_quant.code_bytes_per_row(head_dim)
        assert k_cache.shape[-1] == want and v_cache.shape[-1] == want
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    if block_ends is not None:
        assert block_ends.is_cuda and block_ends.dtype == torch.int32 and block_ends.numel() == num_q_tokens
    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    # Not while a CUDA graph is being captured: the gather is sized by the context (so a
    # replay could not reuse it), and reading that size is a device-to-host sync, which
    # invalidates the capture. The graph keeps the fused kernel.
    # the gather decodes Kai's block formats only; upstream's per-token-head rows stay on the fused kernel
    if block_ends is None and not kv_format.quantized and not torch.cuda.is_current_stream_capturing() and _scratch_attention_applies(
        q,
        k_extend,
        v_extend,
        sliding_window,
        sinks,
        kv_quant,
        num_rows=num_q_tokens // max(1, qo_indptr.numel() - 1),
        kv_len=int(prefix_lens.max()) + num_q_tokens,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        k_scales=k_scales,
        v_scales=v_scales,
    ):
        return _extend_attention_scratch(
            q, k_extend, v_extend, k_cache, v_cache, o,
            qo_indptr, kv_indptr, kv_indices, prefix_lens, sm_scale,
            kv_quant=kv_quant, k_scales=k_scales, v_scales=v_scales,
        )
    block_ends_arg = block_ends if block_ends is not None else qo_indptr
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)
    # Tile size is shared-memory bound: keep the fast (large) tiles on GPUs whose opt-in
    # shared memory fits them, shrink on consumer GPUs (sm_89 ~99KB) where the default
    # 128x64 overflows once head_dim >= 256 (e.g. gemma4: SWA 256, full-attention 512).
    if kv_format.quantized:
        block_m, block_n = _quantized_extend_tile(head_dim, block_d, block_ends is not None)
    else:
        block_m, block_n = _select_extend_tile(
            head_dim, block_d, _optin_smem_bytes(q.device.index), split=k_extend is not None
        )
    grid = (qo_indptr.numel() - 1, num_q_heads, triton.cdiv(max_q_len, block_m))
    ks_arg, vs_arg, s_kss, s_ksh, s_vss, s_vsh, qbits, qblock = _launch_scales(
        kv_quant, kv_format, k_cache, v_cache, k_scales, v_scales, k_scale, v_scale, q
    )
    if k_extend is not None or v_extend is not None:
        assert k_extend is not None and v_extend is not None
        assert k_extend.is_cuda and v_extend.is_cuda
        assert k_extend.dim() == 3 and v_extend.dim() == 3
        assert k_extend.shape[0] == num_q_tokens and v_extend.shape[0] == num_q_tokens
        assert k_extend.shape[1] == num_kv_heads and v_extend.shape[1] == num_kv_heads
        assert k_extend.shape[-1] == head_dim and v_extend.shape[-1] == head_dim
        _extend_attention_split_kernel[grid](
            q,
            k_extend,
            v_extend,
            k_cache,
            v_cache,
            ks_arg,
            vs_arg,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            block_ends_arg,
            sm_scale,
            sinks_arg,
            q.stride(0),
            q.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            v_cache.stride(0),
            v_cache.stride(1),
            s_kss,
            s_ksh,
            s_vss,
            s_vsh,
            o.stride(0),
            o.stride(1),
            QBITS=qbits,
            QBLOCK=qblock,
            KV_OFFSET_DTYPE=_kv_offset_dtype(k_cache, v_cache),
            KV_FMT=kv_format.name,
            GROUP=num_q_heads // num_kv_heads,
            D=head_dim,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            SLIDING_WINDOW=sliding_window or 0,
            HAS_SINKS=sinks is not None,
            HAS_BLOCKS=block_ends is not None,
            num_warps=8,
            num_stages=1,
        )
        return o

    _extend_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks_arg,
        vs_arg,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        block_ends_arg,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        QBITS=qbits,
        QBLOCK=qblock,
        KV_OFFSET_DTYPE=_kv_offset_dtype(k_cache, v_cache),
        KV_FMT=kv_format.name,
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        HAS_BLOCKS=block_ends is not None,
        num_warps=8,
        num_stages=1,
    )
    return o


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_n: int = 32,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    kv_quant=None,
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged causal attention for one layer.

    ``q`` is ``[num_query_tokens, num_q_heads, head_dim]``. KV cache tensors are
    flattened to ``[num_slots, num_kv_heads, head_dim]``. ``indptr`` and
    ``indices`` describe each request's logical KV slots in order.

    ``kv_quant`` (``--kv-cache-dtype``): code slabs read through ``k_scales``/``v_scales``.
    This is the fallback path -- the backend only reaches it for a head_dim the tiled extend
    kernel will not take -- but it carries the quantized read too, because a path that
    silently interpreted codes as 16-bit floats would produce plausible garbage rather than
    an error.
    """

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert v_cache.shape[1] == num_kv_heads
    if kv_quant is None:
        assert k_cache.shape[-1] == head_dim and v_cache.shape[-1] == head_dim
    else:
        assert k_scales is not None and v_scales is not None, "quantized slab without scales"
        want = kv_quant.code_bytes_per_row(head_dim)
        assert k_cache.shape[-1] == want and v_cache.shape[-1] == want
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    block_d = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_q_heads)
    ks_arg, vs_arg, s_kss, s_ksh, s_vss, s_vsh, qbits, qblock = _launch_scales(
        kv_quant, kv_format, k_cache, v_cache, k_scales, v_scales, k_scale, v_scale, q
    )
    _paged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks_arg,
        vs_arg,
        o,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        QBITS=qbits,
        QBLOCK=qblock,
        KV_OFFSET_DTYPE=_kv_offset_dtype(k_cache, v_cache),
        KV_FMT=kv_format.name,
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        num_warps=8 if head_dim >= 256 else 4,
        num_stages=2,
    )
    return o


@triton.jit
def _gather_kv_indices_kernel(
    page_table,
    rows,
    indptr,
    out,
    pt_stride,
    indptr_stride,
    out_stride,
    BLOCK: tl.constexpr,
):
    batch_id = tl.program_id(1)
    indptr_row = indptr + tl.program_id(0) * indptr_stride
    start = tl.load(indptr_row + batch_id)
    n = tl.load(indptr_row + batch_id + 1) - start
    src = page_table + tl.load(rows + batch_id).to(tl.int64) * pt_stride
    dst = out + tl.program_id(0).to(tl.int64) * out_stride + start
    for off in range(tl.program_id(2) * BLOCK, n, tl.num_programs(2) * BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n
        tl.store(dst + cols, tl.load(src + cols, mask=mask), mask=mask)


def gather_kv_indices(page_table: torch.Tensor, rows: torch.Tensor, indptr: torch.Tensor, out: torch.Tensor) -> None:
    """Flat KV slot lists of several decode batches, built on the device: ``out[s, indptr[s, i] + j] = page_table[rows[i], j]`` for ``j < indptr[s, i + 1] - indptr[s, i]``."""
    steps, batch = indptr.shape[0], indptr.shape[1] - 1
    # a fixed grid keeps the launch graph-capturable; long rows loop over it
    chunks = max(1, triton.cdiv(2 * _num_sms(page_table.device.index), steps * batch))
    _gather_kv_indices_kernel[(steps, batch, chunks)](
        page_table,
        rows,
        indptr,
        out,
        page_table.stride(0),
        indptr.stride(0),
        out.stride(0),
        BLOCK=1024,
    )
