"""Block-symmetric quantization of the paged KV cache (``--kv-cache-dtype``).

The point is not the KV cache itself. On a small-VRAM card the paged KV and the MoE expert
slot cache draw from the same pool, and ``--moe-cache-auto`` hands whatever the KV does not
take to the experts. Measured on an RTX 2060 6 GB serving Ornith-1.5-35B-A3B at 64k
(``tools/moe_stats.py`` over 872k routed accesses): the KV holds 1.25 GiB and the expert
cache gets 358 slots, which realizes a 42.2% VRAM hit rate -- one decode step touches 152
expert rows, so the cache is barely two tokens deep. Every byte taken off the KV becomes
expert slots, and slots are worth far more here than context length is.

Layout, one (token, kv head) row of ``head_dim`` values:

    row -> head_dim // BLOCK blocks of BLOCK contiguous values along head_dim
    each block -> BLOCK codes (``bits`` wide, packed) + one fp16 scale

Symmetric, no zero point: ``scale = absmax / qmax``, ``code = round(x / scale)``. The codes
live in a plain uint8 buffer with the same page geometry as the 16-bit slab -- only the last
axis narrows -- and the scales ride a parallel buffer of the same geometry with one entry per
block. Page sharing, eviction and the layer-id remap then apply to both without knowing
anything about quantization.

Sub-byte codes pack two per byte, low nibble first (llama.cpp q4_0 order), so an even index
is ``byte & 0xF`` and an odd one is ``byte >> 4``.

The names ``q8_0`` / ``q4_0`` are the spellings people ask for, but this is NOT the GGUF
block layout: the range is symmetric (``[-qmax, qmax]``, so 4-bit spends one of its sixteen
codes on nothing) where GGUF q4_0 carries a signed scale to reach ``[-8, 7]``. Nothing here
is ever serialized -- the KV cache dies with the process -- so matching GGUF byte for byte
buys nothing, and a symmetric range keeps the store kernel branchless. It does mean a 4-bit
step is ``absmax/7`` rather than ``absmax/8``: 14% coarser, which is the price of that
simplicity and is visible in the accuracy runs.

Bytes per value (vs 2.0 for fp16/bf16), at BLOCK 32:

    q8_0   1 + 2/32   = 1.0625   (1.88x)
    q4_0   0.5 + 2/32 = 0.5625   (3.56x)

BLOCK is 32 because head_dim is a multiple of it on every model this serves (Ornith and
Flash-Next are 256, Qwen3 dense is 128) and because a 32-value block keeps the scale
sidecar at 6% of q8_0 and 11% of q4_0 -- a 64-value block would halve that but doubles the
dynamic range each scale has to cover, which is what actually costs accuracy on K rows with
outlier channels.

This module is the reference: pure torch, no kernels, no CUDA. The Triton store and the
dequantizing attention loads are checked against it bit for bit (tests/kvcache/test_kv_quant.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BLOCK = 32


@dataclass(frozen=True)
class KVQuantSpec:
    """One storage format for the paged K/V slabs."""

    name: str
    bits: int
    qmax: int  # codes span [-qmax, qmax] (q8_0) or [-qmax-1, qmax] (q4_0, llama.cpp order)
    block: int = BLOCK

    @property
    def bytes_per_value(self) -> float:
        return self.bits / 8 + 2 / self.block

    @property
    def ratio_vs_16bit(self) -> float:
        return 2.0 / self.bytes_per_value

    def code_bytes_per_row(self, head_dim: int) -> int:
        """Packed code bytes for one (token, kv head) row."""
        if head_dim % self.block:
            raise ValueError(
                f"--kv-cache-dtype {self.name}: head_dim {head_dim} is not a multiple of {self.block}"
            )
        return head_dim * self.bits // 8

    def blocks_per_row(self, head_dim: int) -> int:
        return head_dim // self.block


Q8_0 = KVQuantSpec(name="q8_0", bits=8, qmax=127)
Q4_0 = KVQuantSpec(name="q4_0", bits=4, qmax=7)

SPECS = {spec.name: spec for spec in (Q8_0, Q4_0)}


def resolve(name: str | None) -> KVQuantSpec | None:
    """``None``/``auto``/``bf16``/``fp16`` -> unquantized (None); otherwise the spec."""
    if name in (None, "auto", "bf16", "fp16"):
        return None
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown --kv-cache-dtype {name!r}; choose from auto, {', '.join(SPECS)}"
        ) from None


def quantize_rows(x: torch.Tensor, spec: KVQuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., head_dim]`` values -> (packed uint8 codes, fp16 scales).

    The reference the kernels are held to. Rounding is round-half-away-from-zero
    (``torch.round`` is round-half-to-even, which disagrees with the kernel's
    ``floor(x + 0.5)`` on exact .5 -- rare with real activations, constant in tests).
    """
    head_dim = x.shape[-1]
    nblocks = spec.blocks_per_row(head_dim)
    blocks = x.reshape(*x.shape[:-1], nblocks, spec.block).to(torch.float32)

    absmax = blocks.abs().amax(dim=-1)
    # Round the scale to what actually gets stored FIRST, then quantize against that. Dividing
    # by the fp32 scale and multiplying back by the fp16 one leaves an error up to half a step
    # plus qmax * scale * 2^-11 -- 12% past the half-step bound for q8_0, and for no reason:
    # the reader only ever has the stored scale.
    scale = (absmax / spec.qmax).to(torch.float16).to(torch.float32)
    # An all-zero block gets scale 0; dividing by it would make NaN, and 0 * anything is the
    # zero we want back, so clamp the divisor only.
    codes = blocks / torch.where(scale > 0, scale, torch.ones_like(scale)).unsqueeze(-1)
    codes = torch.where(codes >= 0, torch.floor(codes + 0.5), torch.ceil(codes - 0.5))
    codes = codes.clamp(-spec.qmax, spec.qmax).to(torch.int8)

    if spec.bits == 8:
        packed = codes.reshape(*x.shape[:-1], head_dim).view(torch.uint8)
    elif spec.bits == 4:
        nib = (codes.to(torch.int16) & 0xF).to(torch.uint8)
        nib = nib.reshape(*x.shape[:-1], head_dim)
        packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()
    else:  # pragma: no cover -- SPECS has no other width
        raise ValueError(f"unsupported width {spec.bits}")
    return packed, scale.to(torch.float16)


def dequantize_rows(
    codes: torch.Tensor, scales: torch.Tensor, spec: KVQuantSpec, dtype: torch.dtype
) -> torch.Tensor:
    """(packed uint8 codes, fp16 scales) -> ``[..., head_dim]`` values."""
    if spec.bits == 8:
        signed = codes.view(torch.int8).to(torch.float32)
        head_dim = codes.shape[-1]
    elif spec.bits == 4:
        lo = (codes & 0xF).to(torch.int16)
        hi = (codes >> 4).to(torch.int16)
        # nibbles are two's complement in 4 bits: 8..15 mean -8..-1
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        head_dim = codes.shape[-1] * 2
        signed = torch.stack((lo, hi), dim=-1).reshape(*codes.shape[:-1], head_dim)
        signed = signed.to(torch.float32)
    else:  # pragma: no cover
        raise ValueError(f"unsupported width {spec.bits}")

    nblocks = spec.blocks_per_row(head_dim)
    out = signed.reshape(*signed.shape[:-1], nblocks, spec.block)
    out = out * scales.to(torch.float32).unsqueeze(-1)
    return out.reshape(*signed.shape[:-1], head_dim).to(dtype)


__all__ = [
    "BLOCK",
    "KVQuantSpec",
    "Q4_0",
    "Q8_0",
    "SPECS",
    "dequantize_rows",
    "quantize_rows",
    "resolve",
]
