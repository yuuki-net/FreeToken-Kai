"""The fused quantize+scatter KV store (freetoken.kernel.triton.kv_quant).

Held to the torch reference bit for bit -- same codes, same scale bytes -- because the
attention loads dequantize against that same definition. Anything the kernel rounds
differently shows up as a silent accuracy loss no accuracy test would attribute correctly."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.kv_quant import Q4_0, Q8_0, quantize_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _slabs(slots, heads, head_dim, spec, device):
    codes = torch.zeros(
        slots, heads, spec.code_bytes_per_row(head_dim), dtype=torch.uint8, device=device
    )
    scales = torch.zeros(
        slots, heads, spec.blocks_per_row(head_dim), dtype=torch.float16, device=device
    )
    return codes, scales


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_kernel_matches_the_reference_bit_for_bit(spec, head_dim):
    from freetoken.kernel.triton.kv_quant import quantize_store_kv

    torch.manual_seed(0)
    dev = torch.device("cuda")
    tokens, heads, slots = 37, 2, 128

    k = torch.randn(tokens, heads, head_dim, dtype=torch.float16, device=dev)
    v = torch.randn(tokens, heads, head_dim, dtype=torch.float16, device=dev) * 0.05
    out_loc = torch.randperm(slots, device=dev)[:tokens].to(torch.int32)

    kc, ks = _slabs(slots, heads, head_dim, spec, dev)
    vc, vs = _slabs(slots, heads, head_dim, spec, dev)
    quantize_store_kv(k, v, out_loc, kc, ks, vc, vs, spec)

    for src, codes, scales in ((k, kc, ks), (v, vc, vs)):
        want_c, want_s = quantize_rows(src.float(), spec)
        got_c = codes[out_loc.long()]
        got_s = scales[out_loc.long()]
        assert torch.equal(got_c, want_c.to(dev)), "codes differ from the reference"
        assert torch.equal(
            got_s.view(torch.int16), want_s.to(dev).view(torch.int16)
        ), "scale bytes differ from the reference"


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_only_the_named_slots_are_touched(spec):
    from freetoken.kernel.triton.kv_quant import quantize_store_kv

    torch.manual_seed(1)
    dev = torch.device("cuda")
    tokens, heads, head_dim, slots = 8, 2, 128, 64

    k = torch.randn(tokens, heads, head_dim, dtype=torch.float16, device=dev)
    v = torch.randn(tokens, heads, head_dim, dtype=torch.float16, device=dev)
    out_loc = torch.arange(tokens, device=dev, dtype=torch.int32) * 3  # 0,3,6,...

    kc, ks = _slabs(slots, heads, head_dim, spec, dev)
    vc, vs = _slabs(slots, heads, head_dim, spec, dev)
    quantize_store_kv(k, v, out_loc, kc, ks, vc, vs, spec)

    untouched = torch.ones(slots, dtype=torch.bool, device=dev)
    untouched[out_loc.long()] = False
    assert torch.all(kc[untouched] == 0) and torch.all(ks[untouched] == 0)
    assert torch.all(vc[untouched] == 0) and torch.all(vs[untouched] == 0)


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_an_all_zero_row_stores_a_zero_scale_and_no_nan(spec):
    from freetoken.kernel.triton.kv_quant import quantize_store_kv
    from freetoken.kvcache.kv_quant import dequantize_rows

    dev = torch.device("cuda")
    heads, head_dim, slots = 1, 128, 4
    k = torch.zeros(1, heads, head_dim, dtype=torch.float16, device=dev)
    v = torch.zeros(1, heads, head_dim, dtype=torch.float16, device=dev)
    out_loc = torch.zeros(1, dtype=torch.int32, device=dev)

    kc, ks = _slabs(slots, heads, head_dim, spec, dev)
    vc, vs = _slabs(slots, heads, head_dim, spec, dev)
    quantize_store_kv(k, v, out_loc, kc, ks, vc, vs, spec)

    assert torch.all(ks[0] == 0)
    back = dequantize_rows(kc[0].cpu(), ks[0].cpu(), spec, torch.float32)
    assert torch.all(back == 0) and not torch.any(torch.isnan(back))


def test_a_non_contiguous_source_is_stored_correctly():
    """qkv arrives as one fused projection; k and v are wide-pitch slices of it."""
    from freetoken.kernel.triton.kv_quant import quantize_store_kv

    torch.manual_seed(2)
    dev = torch.device("cuda")
    tokens, heads, head_dim, slots = 16, 2, 128, 32
    fused = torch.randn(tokens, 3, heads, head_dim, dtype=torch.float16, device=dev)
    k, v = fused[:, 1], fused[:, 2]
    assert not k.is_contiguous()
    out_loc = torch.arange(tokens, device=dev, dtype=torch.int32)

    kc, ks = _slabs(slots, heads, head_dim, Q8_0, dev)
    vc, vs = _slabs(slots, heads, head_dim, Q8_0, dev)
    quantize_store_kv(k, v, out_loc, kc, ks, vc, vs, Q8_0)

    want_c, _ = quantize_rows(k.float(), Q8_0)
    assert torch.equal(kc[:tokens], want_c.to(dev))
