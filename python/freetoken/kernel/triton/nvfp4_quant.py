"""bf16/fp32 -> NVFP4 (ModelOpt layout) weight quantizer, pure torch (CPU or GPU).

Produces exactly what the native ``nvfp4`` bank schema and the Triton dequant / GEMV kernels
read (see ``kernel/triton/nvfp4_dequant.py``): two E2M1 codes per byte with the even element
in the LOW nibble, an fp8-e4m3 scale per 16-wide block, and an fp16 global scale per output
row; ``weight = e2m1 * block_scale * global``. Used to turn the checkpoint's unquantized
(bf16) tensors -- the MTP head's stacked experts, and dense projections under a future Q4
mode -- into the same format the routed experts already ship in.
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16
# E2M1 magnitudes by code 0..7 and the rounding midpoints between them
_E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``w [N, K]`` (K % 16 == 0) -> ``(packed [N, K//2] uint8, scale [N, K//16] fp8-e4m3,
    global [N] fp16)``. Row-wise global = amax / (6 * 448) so every block scale fits e4m3;
    block scale = block amax / 6 / global, rounded to e4m3; codes = round-to-nearest E2M1."""
    assert w.dim() == 2 and w.shape[1] % BLOCK == 0, w.shape
    n, k = w.shape
    wf = w.to(torch.float32)
    amax_row = wf.abs().amax(dim=1)
    # fp16-representable global; the tiny floor keeps an all-zero row finite
    g = (amax_row / (E2M1_MAX * FP8_MAX)).clamp(min=2.0 ** -20).to(torch.float16)
    gf = g.to(torch.float32)
    blocks = wf.view(n, k // BLOCK, BLOCK)
    amax_blk = blocks.abs().amax(dim=2)
    s = (amax_blk / E2M1_MAX / gf[:, None]).clamp(max=FP8_MAX)
    s8 = s.to(torch.float8_e4m3fn)
    denom = s8.to(torch.float32) * gf[:, None]
    # a zero block quantizes to code 0 whatever the divisor
    x = blocks / torch.where(denom > 0, denom, torch.ones_like(denom))[:, :, None]
    mag = x.abs().clamp(max=E2M1_MAX)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=w.device)
    code = torch.bucketize(mag.reshape(-1), mids).view(n, k).to(torch.uint8)
    code = code | ((x.view(n, k) < 0).to(torch.uint8) << 3)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, s8.contiguous(), g.contiguous()


def dequant_nvfp4_rows(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Reference inverse of :func:`quantize_nvfp4_rows` (fp32 ``[N, K]``), the same math as
    the Triton dequant kernel: low nibble first, ``e2m1 * block_scale * global``."""
    n, kp = packed.shape
    lut = torch.tensor(
        [*_E2M1_MAGS, *(-m for m in _E2M1_MAGS)], dtype=torch.float32, device=packed.device
    )
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=2).view(n, kp * 2)
    vals = lut[codes]
    s = scale.to(torch.float32).repeat_interleave(BLOCK, dim=1)
    return vals * s * global_scale.to(torch.float32)[:, None]


def nvfp4_expert_bank_specs(e: int, h: int, i: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes/dtypes of the six native NVFP4 bank tensors for ``e`` experts (the
    ``_alloc_nvfp4_host_banks`` layout)."""
    return {
        "gate_up_packed": ((e, 2 * i, h // 2), torch.uint8),
        "gate_up_scale": ((e, 2 * i, h // BLOCK), torch.float8_e4m3fn),
        "gate_up_global": ((e, 2 * i), torch.float16),
        "down_packed": ((e, h, i // 2), torch.uint8),
        "down_scale": ((e, h, i // BLOCK), torch.float8_e4m3fn),
        "down_global": ((e, h), torch.float16),
    }


def quantize_nvfp4_experts(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    chunk: int = 16,
    device: torch.device | None = None,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Stacked unquantized experts ``gate_up [E, 2I, H]``, ``down [E, H, I]`` -> the six
    native NVFP4 bank tensors, ``chunk`` experts at a time. ``device`` is where each chunk is
    quantized (default: the inputs' device; host-resident inputs stream chunk by chunk through
    it, so a 5 GB bf16 layer costs a few hundred MB of VRAM); ``out`` supplies the destination
    tensors (e.g. pinned host banks) instead of allocating them on ``device``."""
    e, two_i, h = gate_up.shape
    assert down.shape == (e, h, two_i // 2), (gate_up.shape, down.shape)
    i = two_i // 2
    dev = device if device is not None else gate_up.device
    if out is None:
        out = {
            n: torch.empty(shape, dtype=dt, device=dev)
            for n, (shape, dt) in nvfp4_expert_bank_specs(e, h, i).items()
        }
    for e0 in range(0, e, chunk):
        e1 = min(e0 + chunk, e)
        p, s, g = quantize_nvfp4_rows(gate_up[e0:e1].to(dev).reshape(-1, h))
        out["gate_up_packed"][e0:e1].copy_(p.view(e1 - e0, two_i, h // 2))
        out["gate_up_scale"][e0:e1].copy_(s.view(e1 - e0, two_i, h // BLOCK))
        out["gate_up_global"][e0:e1].copy_(g.view(e1 - e0, two_i))
        p, s, g = quantize_nvfp4_rows(down[e0:e1].to(dev).reshape(-1, i))
        out["down_packed"][e0:e1].copy_(p.view(e1 - e0, h, i // 2))
        out["down_scale"][e0:e1].copy_(s.view(e1 - e0, h, i // BLOCK))
        out["down_global"][e0:e1].copy_(g.view(e1 - e0, h))
        del p, s, g
    return out


__all__ = ["quantize_nvfp4_rows", "dequant_nvfp4_rows", "quantize_nvfp4_experts", "nvfp4_expert_bank_specs"]
