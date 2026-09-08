"""The reference quantizer behind ``--kv-cache-dtype`` (freetoken.kvcache.kv_quant).

Pure torch on the CPU: this is the definition the Triton store and the dequantizing
attention loads are held to, so it has to be pinned down on its own first -- packing order,
rounding, the all-zero block, and the byte arithmetic the cache budget bills."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.kv_quant import (
    BLOCK,
    Q4_0,
    Q8_0,
    dequantize_rows,
    quantize_rows,
    resolve,
)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_bytes_per_value_matches_the_layout(spec):
    head_dim = 256
    code_bytes = spec.code_bytes_per_row(head_dim)
    scale_bytes = spec.blocks_per_row(head_dim) * 2
    assert (code_bytes + scale_bytes) / head_dim == spec.bytes_per_value


def test_the_two_formats_hit_their_advertised_ratios():
    assert Q8_0.bytes_per_value == pytest.approx(1.0625)
    assert Q4_0.bytes_per_value == pytest.approx(0.5625)
    assert Q4_0.ratio_vs_16bit == pytest.approx(3.5555, rel=1e-3)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_roundtrip_stays_inside_half_a_step_of_the_block_scale(spec, dtype):
    torch.manual_seed(0)
    x = torch.randn(7, 3, 256, dtype=dtype).to(torch.float32)
    codes, scales = quantize_rows(x, spec)
    back = dequantize_rows(codes, scales, spec, torch.float32)

    # Per block, the error a symmetric quantizer can leave is half a step, and the step is
    # absmax/qmax. Checking against the block's own scale (not a global tolerance) is what
    # makes this meaningful for K rows with outlier channels.
    blocks = x.reshape(7, 3, 256 // BLOCK, BLOCK)
    err = (back - x).reshape(7, 3, 256 // BLOCK, BLOCK).abs().amax(dim=-1)
    step = blocks.abs().amax(dim=-1) / spec.qmax
    assert torch.all(err <= step / 2 + 1e-6)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_an_all_zero_block_survives_without_a_nan(spec):
    x = torch.zeros(2, 1, 64)
    codes, scales = quantize_rows(x, spec)
    assert torch.all(scales == 0)
    back = dequantize_rows(codes, scales, spec, torch.float32)
    assert torch.all(back == 0)
    assert not torch.any(torch.isnan(back))


def test_q4_packs_two_values_per_byte_low_nibble_first():
    # One block whose two extremes are +absmax and -absmax. The range is symmetric, so they
    # land on +7 and -7 (0x9 in two's complement) -- code 0x8 (-8) is the one this layout
    # deliberately never emits.
    x = torch.zeros(1, BLOCK)
    x[0, 0] = 8.0
    x[0, 1] = -8.0
    codes, scales = quantize_rows(x, Q4_0)
    assert codes.shape == (1, BLOCK // 2)
    assert scales.shape == (1, 1)
    first = int(codes[0, 0])
    assert first & 0xF == 7, "value 0 belongs in the low nibble"
    assert first >> 4 == 0x9, "value 1 (-7) is two's complement 0x9 in the high nibble"
    back = dequantize_rows(codes, scales, Q4_0, torch.float32)
    assert back[0, 0] > 0 and back[0, 1] < 0
    assert back[0, 0] == pytest.approx(-back[0, 1]), "symmetric range, symmetric roundtrip"


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_codes_never_leave_the_representable_range(spec):
    torch.manual_seed(1)
    x = torch.randn(4, 128) * 100
    codes, _ = quantize_rows(x, spec)
    if spec.bits == 8:
        signed = codes.view(torch.int8).to(torch.int32)
        assert int(signed.min()) >= -spec.qmax and int(signed.max()) <= spec.qmax
    else:
        nib = torch.cat([(codes & 0xF), (codes >> 4)], dim=-1).to(torch.int32)
        nib = torch.where(nib > 7, nib - 16, nib)
        assert int(nib.min()) >= -spec.qmax - 1 and int(nib.max()) <= spec.qmax


def test_head_dim_has_to_divide_into_blocks():
    with pytest.raises(ValueError, match="not a multiple"):
        Q8_0.code_bytes_per_row(100)


def test_resolve_maps_the_cli_spellings():
    assert resolve(None) is None
    assert resolve("auto") is None
    assert resolve("bf16") is None
    assert resolve("q4_0") is Q4_0
    with pytest.raises(ValueError, match="unknown --kv-cache-dtype"):
        resolve("q3_k")
