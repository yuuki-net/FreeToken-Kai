# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from freetoken.kernel.fla.index import prepare_chunk_indices
from freetoken.kernel.fla.op import exp, safe_exp
from freetoken.kernel.fla.utils import check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@lru_cache(maxsize=1)
def chunk_o_config() -> tuple[int, int, int, int]:
    """``(BK, BV, num_warps, num_stages)`` for ``chunk_fwd_kernel_o``.

    Upstream's autotune above is commented out, so one hardcoded config serves every card.
    That config -- (128, 64), 4 warps -- is an Ampere+ choice, and on a Turing card it is
    the wrong one: with 64 KB of shared memory the (128, 64) tiles leave one block per SM
    and four warps to hide it. Measured on an RTX 2060 at Ornith's GDN shapes (T=2048,
    Hg=16, H=32, K=V=128, BT=64), one layer of the 30:

        (128, 64) 4 warps   63.6 ms   0.07 TFLOPS   <- upstream
        (128, 64) 8 warps   38.6 ms   0.11
        ( 64, 32) 8 warps   11.2 ms   0.38
        ( 64, 64) 8 warps   11.2 ms   0.39          <- this

    Over 30 GDN layers that is 1.91 s -> 0.33 s of every prefill chunk. The tiling does not
    change what the kernel computes, only how it is cut. ``FREETOKEN_GDN_CHUNK_O_*``
    overrides each field for a card this was not measured on.
    """
    from freetoken.utils.arch import is_pre_ampere

    bk, bv, warps, stages = (64, 64, 8, 2) if is_pre_ampere() else (128, 64, 4, 2)
    return (
        int(os.getenv("FREETOKEN_GDN_CHUNK_O_BK") or bk),
        int(os.getenv("FREETOKEN_GDN_CHUNK_O_BV") or bv),
        int(os.getenv("FREETOKEN_GDN_CHUNK_O_NUM_WARPS") or warps),
        int(os.getenv("FREETOKEN_GDN_CHUNK_O_NUM_STAGES") or stages),
    )


# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    block_k, block_v, num_warps, num_stages = chunk_o_config()

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=block_k,
        BV=block_v,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
