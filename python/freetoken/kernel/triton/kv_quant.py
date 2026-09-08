"""Fused quantize + scatter into the paged K/V code slabs (``--kv-cache-dtype``).

One launch writes both K and V for a whole forward: the unquantized path's ``store_cache``
is a byte copy, and this replaces it rather than sitting after it, so a decode step still
costs exactly one KV store. That matters beyond speed -- the store is captured into the
decode CUDA graph, where the destination slots arrive as a device tensor and no host-side
branch is allowed.

Layout is ``freetoken.kvcache.kv_quant``'s: per (token, kv head) row, ``head_dim // BLOCK``
blocks of ``BLOCK`` contiguous values, each with one fp16 scale.

The kernel reads each row as two half-width lanes (the even and the odd positions of every
pair) instead of a ``[nblocks, BLOCK]`` tile. That looks roundabout for 8-bit, but it is what
lets the 4-bit path pack a pair into one byte without a reshape or a split of a register
tile: the two nibbles of a byte are exactly one even lane and one odd lane, a pair never
straddles a block boundary (BLOCK is even), and the block-wide absmax is just the max of the
two lanes' maxima. Both widths then run the same code with one constexpr branch at the store.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _quantize_store_kernel(
    k_ptr,
    v_ptr,
    kc_ptr,
    ks_ptr,
    vc_ptr,
    vs_ptr,
    slot_ptr,
    num_tokens,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    HALF: tl.constexpr,
    BITS: tl.constexpr,
    QMAX: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    if token >= num_tokens:
        return
    slot = tl.load(slot_ptr + token).to(tl.int64)

    offs_nb = tl.arange(0, NB)
    offs_j = tl.arange(0, HALF)
    # even/odd positions of each pair, laid out [nblocks, BLOCK//2]
    pair = offs_nb[:, None] * BLOCK + 2 * offs_j[None, :]

    for which in tl.static_range(2):
        src = k_ptr + token * stride_kt + head * stride_kh
        code = kc_ptr + slot * stride_ct + head * stride_ch
        scale_p = ks_ptr + slot * stride_st + head * stride_sh
        if which == 1:
            src = v_ptr + token * stride_vt + head * stride_vh
            code = vc_ptr + slot * stride_ct + head * stride_ch
            scale_p = vs_ptr + slot * stride_st + head * stride_sh

        x_e = tl.load(src + pair).to(tl.float32)
        x_o = tl.load(src + pair + 1).to(tl.float32)

        absmax = tl.maximum(
            tl.max(tl.abs(x_e), axis=1),
            tl.max(tl.abs(x_o), axis=1),
        )
        # Round the scale to fp16 BEFORE quantizing against it: the reader only ever has the
        # stored value, so anything finer just shifts error onto the codes.
        scale = (absmax / QMAX).to(tl.float16).to(tl.float32)
        tl.store(scale_p + offs_nb, scale.to(tl.float16))

        inv = tl.where(scale > 0, 1.0 / tl.where(scale > 0, scale, 1.0), 0.0)
        q_e = x_e * inv[:, None]
        q_o = x_o * inv[:, None]
        q_e = tl.where(q_e >= 0, tl.floor(q_e + 0.5), tl.ceil(q_e - 0.5))
        q_o = tl.where(q_o >= 0, tl.floor(q_o + 0.5), tl.ceil(q_o - 0.5))
        c_e = tl.minimum(tl.maximum(q_e, -QMAX), QMAX).to(tl.int32)
        c_o = tl.minimum(tl.maximum(q_o, -QMAX), QMAX).to(tl.int32)

        if BITS == 8:
            tl.store(code + pair, (c_e & 0xFF).to(tl.uint8))
            tl.store(code + pair + 1, (c_o & 0xFF).to(tl.uint8))
        else:
            byte = (c_e & 0xF) | ((c_o & 0xF) << 4)
            packed = offs_nb[:, None] * HALF + offs_j[None, :]
            tl.store(code + packed, byte.to(tl.uint8))


def quantize_store_kv(
    k,
    v,
    out_loc,
    k_codes,
    k_scales,
    v_codes,
    v_scales,
    spec,
) -> None:
    """Quantize ``k``/``v`` ``[T, H, D]`` into the code/scale slabs at ``out_loc``.

    ``k_codes``/``v_codes`` are ``[S, H, D * bits // 8]`` uint8 and the scale slabs
    ``[S, H, D // BLOCK]`` fp16, both row-flat over the pooled slots.
    """
    num_tokens, heads, head_dim = k.shape
    nb = spec.blocks_per_row(head_dim)
    if num_tokens == 0:
        return
    _quantize_store_kernel[(num_tokens, heads)](
        k,
        v,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        out_loc,
        num_tokens,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_codes.stride(0),
        k_codes.stride(1),
        k_scales.stride(0),
        k_scales.stride(1),
        BLOCK=spec.block,
        NB=nb,
        HALF=spec.block // 2,
        BITS=spec.bits,
        QMAX=spec.qmax,
        num_warps=4,
    )


__all__ = ["quantize_store_kv"]
