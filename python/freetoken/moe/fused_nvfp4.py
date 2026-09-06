"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swiglu_clamp_and_mul,
    swigluoai_and_mul,
)
from freetoken.moe.fused import moe_align_block_size

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _run_act(
    activation: str,
    gate_up: torch.Tensor,
    out: torch.Tensor,
    act_alpha: float,
    act_limit: float,
) -> None:
    """gemm1 -> gemm2 activation dispatch. ``swigluoai`` (MiniMax-M3, clamped
    gpt-oss swiglu over the banks' uninterleaved [gate; up] halves) and
    ``swiglu_clamp`` (GLM-5.3, same clamp without the +1 up bias) carry the
    per-model ``act_alpha``/``act_limit`` scalars; the plain *_and_mul kinds
    ignore them."""
    if activation == "swigluoai":
        swigluoai_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    if activation == "swiglu_clamp":
        swiglu_clamp_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    _ACT[activation](gate_up, out)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode config (int32 wide loads + deferred reduction). Offline sweep over
# the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16, BLOCK_KW=16 (== 128
# k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM ~43%->~51% of peak BW.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_WARPS = 4
# Deep-K variant: at K > 2048 (qwen4_exp gate_up, K=2560) a narrower N tile with the whole
# K strip in one program iteration measures ~13% faster (18.6 vs 21.0us); short-K shapes
# regress under it, so the split is by K, not by gemm position.
_DECODE_MARLIN_DEEPK_BLOCK_N = 8
_DECODE_MARLIN_DEEPK_BLOCK_KW = 128
_DECODE_MARLIN_DEEPK_THRESHOLD = 2048


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    deep_k = K > _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_MARLIN_WARPS,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


@functools.cache
def _scratch_moe_preferred() -> bool:
    """Prefill MoE below Ampere: dequant + cuBLAS instead of the in-kernel-dequant GEMM. Measured
    on an RTX 2060 (T=2785, 256 experts, top-8): the Triton kernel tops out at 0.57 TFLOPS in
    every tile configuration tried, cuBLAS fp16 runs at 19 TFLOPS on the same card.
    ``FREETOKEN_NVFP4_MOE_SCRATCH=0/1`` overrides anywhere."""
    env = os.environ.get("FREETOKEN_NVFP4_MOE_SCRATCH")
    if env is not None:
        return env == "1"
    from freetoken.utils import is_pre_ampere

    return is_pre_ampere()


_SCRATCH_EXPERTS_PER_CHUNK = 32  # 32 experts x (2I x H + H x I) fp16 ~ 200 MB for Qwen3.5-35B-A3B
_SCRATCH_ROWS_PER_BLOCK = 64     # rows per expert per bmm; bounds the loop's buffers (~45 MB total)
_SCRATCH_BUFFERS: dict = {}


def _scratch_buffer(name: str, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """One persistent buffer per (name, shape, dtype, device). The dequant scratches are
    allocated once and reused across chunks, layers and requests instead of being freed and
    re-allocated per chunk: on a full 6 GB card the caching allocator otherwise ends up
    unmapping expandable segments while the copy / host-callback streams are still busy, which
    surfaces as ``CUDA driver error: device not ready`` in whatever op allocates next."""
    key = (name, tuple(int(d) for d in shape), dtype, str(device))
    buf = _SCRATCH_BUFFERS.get(key)
    if buf is None:
        # first allocation only: let the queued work drain first. Growing an expandable
        # segment while kernels and copies are in flight is what failed with "device not
        # ready" on WSL2; a one-time sync per buffer costs nothing afterwards.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        buf = _SCRATCH_BUFFERS[key] = torch.empty(shape, dtype=dtype, device=device)
    return buf


def _fused_experts_nvfp4_scratch(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """silu-gated prefill MoE as batched cuBLAS GEMMs over dequantised fp16/bf16 experts, with a
    bounded working set. Routes are grouped by expert once (argsort); experts are processed in
    chunks of 32, ordered by their route count so the experts of a chunk have similar queue
    lengths; each chunk's queues are walked in blocks of ``_SCRATCH_ROWS_PER_BLOCK`` rows and
    every block goes through two ``bmm`` calls (``x @ W_gu^T -> silu(gate) * up -> @ W_d^T``)
    over the experts that still have rows in it (a prefix of the chunk, thanks to the ordering).
    The route weights and the sum over a token's routes are applied in fp32.

    Every device tensor of the loop lives in a persistent buffer sized by the block, not by the
    prompt: on a full 6 GB card the earlier per-chunk temporaries (``[experts, L_max, H]`` fp32,
    100+ MB for a 4096-token chunk) made the caching allocator release expandable segments,
    which surfaces as ``CUDA driver error: device not ready``. One host sync per layer (route
    counts)."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    dt = hidden_states.dtype
    dev = hidden_states.device
    inter = gate_up_packed.shape[1] // 2
    C, LB = _SCRATCH_EXPERTS_PER_CHUNK, _SCRATCH_ROWS_PER_BLOCK
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    order = torch.argsort(flat_ids)
    counts = torch.bincount(flat_ids.clamp(min=0), minlength=num_experts)
    starts = torch.cumsum(counts, 0) - counts
    counts_cpu = counts.tolist()
    # experts by queue length, longest first; idle experts drop out
    by_len = sorted((e for e in range(num_experts) if counts_cpu[e] > 0), key=lambda e: -counts_cpu[e])
    out = torch.zeros(M, H, dtype=torch.float32, device=dev)
    if not by_len:
        return out.to(dt)
    x_buf = _scratch_buffer("x", (C * LB, H), dt, dev)
    h_buf = _scratch_buffer("h", (C, LB, 2 * inter), dt, dev)
    a_buf = _scratch_buffer("a", (C, LB, inter), dt, dev)
    y16_buf = _scratch_buffer("y16", (C, LB, H), dt, dev)
    y32_buf = _scratch_buffer("y32", (C, LB, H), torch.float32, dev)
    ar = torch.arange(LB, device=dev)
    last = order.numel() - 1
    for c0 in range(0, len(by_len), C):
        chunk = by_len[c0 : c0 + C]
        n = len(chunk)
        slots = torch.tensor(chunk, dtype=torch.int32, device=dev)
        w_gu = dequant_nvfp4(
            gate_up_packed, gate_up_scale, gate_up_global, slots,
            out=_scratch_buffer("gate_up", (n, 2 * inter, H), dt, dev), dtype=dt,
        )
        w_d = dequant_nvfp4(
            down_packed, down_scale, down_global, slots,
            out=_scratch_buffer("down", (n, H, inter), dt, dev), dtype=dt,
        )
        e_ids = slots.long()
        for l0 in range(0, counts_cpu[chunk[0]], LB):
            # experts of the chunk with rows in this block: a prefix (queues sorted, longest first)
            na = sum(1 for e in chunk if counts_cpu[e] > l0)
            ids = e_ids[:na]
            # padded route table of the block: pos[j, l] = route l0 + l of expert ids[j]
            valid = (l0 + ar)[None, :] < counts[ids][:, None]                     # [na, LB]
            pos = (starts[ids][:, None] + l0 + ar[None, :]).clamp(max=last)
            rows = order[pos]                                                      # flat route ids
            tok = (rows // top_k) * valid                                          # padding -> token 0
            rw = flat_w[rows] * valid                                              # padding -> weight 0
            xb = torch.index_select(hidden_states, 0, tok.reshape(-1), out=x_buf[: na * LB]).view(na, LB, H)
            if apply_router_weight_on_input:
                xb = (xb.float() * rw[..., None]).to(dt)
            hb = torch.bmm(xb, w_gu[:na].transpose(1, 2), out=h_buf[:na])          # [na, LB, 2I]
            gate, up = hb[..., :inter], hb[..., inter:]
            torch.mul(torch.nn.functional.silu(gate, inplace=True), up, out=a_buf[:na])
            yb = torch.bmm(a_buf[:na], w_d[:na].transpose(1, 2), out=y16_buf[:na])  # [na, LB, H]
            scale = valid.to(torch.float32) if apply_router_weight_on_input else rw
            torch.mul(yb, scale[..., None], out=y32_buf[:na])
            out.index_add_(0, tok.reshape(-1), y32_buf[:na].reshape(-1, H))
    return out.to(dt)


@functools.cache
def _arith_dequant() -> bool:
    """Prefill MoE kernel dequant: arithmetic (no LUT gathers) below Ampere by default --
    measured 0.56 TFLOPS for the LUT path on an RTX 2060 against 16 TFLOPS for the dense
    NVFP4 kernels that already use the arithmetic form. ``FREETOKEN_NVFP4_MOE_ARITH=0/1``
    overrides anywhere (the two forms are bit-identical, so this is purely a speed knob)."""
    env = os.environ.get("FREETOKEN_NVFP4_MOE_ARITH")
    if env is not None:
        return env == "1"
    from freetoken.utils import is_pre_ampere

    return is_pre_ampere()


def _prefill_config(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune; these were chosen by an offline sweep
    # over (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages) for the MiniMax-M2 shapes.
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                GROUP_SIZE_M=8, num_warps=8, num_stages=4)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        ARITH_DEQUANT=_arith_dequant(),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    if activation == "silu" and _scratch_moe_preferred():
        return _fused_experts_nvfp4_scratch(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            down_packed, down_scale, down_global, topk_weights, topk_ids, num_experts,
            apply_router_weight_on_input,
        )
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
