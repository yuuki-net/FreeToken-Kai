"""QSA sparse attention reading a block-quantized paged K/V (``--kv-cache-dtype``).

Same discipline as the dense path: held against a *dequantized oracle* — the identical
kernel fed a 16-bit slab built by pushing the codes back through the torch reference — so a
wrong unpack cannot hide inside the format's own error.

This matters more here than for the dense pool. Flash-Next reads a fixed 2048 tokens per
query no matter how long the context is, so quantizing its paged K/V does not put a
context-proportional cost on decode the way it does for a dense model. If the unpack is
right, this is the case where the trade is worth taking.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.kv_quant import Q4_0, Q8_0, dequantize_rows, quantize_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _paged(rows: torch.Tensor, pages: int, page_size: int):
    """[slots, heads, dim] -> [pages, page_size, heads, dim]."""
    heads, dim = rows.shape[1], rows.shape[2]
    return rows.reshape(pages, page_size, heads, dim).contiguous()


def _case(spec, dtype, head_dim=128, kv_heads=2, q_heads=8, pages=8, page_size=16, topk=48,
          n_rows=3):
    dev = torch.device("cuda")
    slots = pages * page_size
    rows = torch.arange(slots, device=dev)

    k = torch.randn(slots, kv_heads, head_dim, device=dev, dtype=dtype)
    v = torch.randn(slots, kv_heads, head_dim, device=dev, dtype=dtype)

    kc, ks = quantize_rows(k.float(), spec)
    vc, vs = quantize_rows(v.float(), spec)
    kc, ks, vc, vs = kc.to(dev), ks.to(dev), vc.to(dev), vs.to(dev)
    k_oracle = dequantize_rows(kc.cpu(), ks.cpu(), spec, dtype).to(dev)
    v_oracle = dequantize_rows(vc.cpu(), vs.cpu(), spec, dtype).to(dev)

    q = torch.randn(n_rows, q_heads, head_dim, device=dev, dtype=dtype)
    # every query looks at the first `topk` logical tokens
    indices = rows[:topk].to(torch.int32).repeat(n_rows, 1).contiguous()
    block_table = torch.arange(pages, device=dev, dtype=torch.int32).repeat(1, 1).contiguous()
    token_to_req = torch.zeros(n_rows, device=dev, dtype=torch.int32)

    return dict(
        q=q,
        codes=(_paged(kc, pages, page_size), _paged(vc, pages, page_size)),
        scales=(_paged(ks, pages, page_size), _paged(vs, pages, page_size)),
        oracle=(_paged(k_oracle, pages, page_size), _paged(v_oracle, pages, page_size)),
        indices=indices,
        block_table=block_table,
        token_to_req=token_to_req,
    )


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qsa_reads_back_what_the_slab_holds(spec, dtype):
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention

    torch.manual_seed(11)
    c = _case(spec, dtype)

    want = qsa_sparse_paged_attention(
        c["q"], c["oracle"][0], c["oracle"][1], c["indices"], c["block_table"],
        c["token_to_req"],
    )
    got = qsa_sparse_paged_attention(
        c["q"], c["codes"][0], c["codes"][1], c["indices"], c["block_table"],
        c["token_to_req"], None,
        k_scales=c["scales"][0], v_scales=c["scales"][1], kv_quant=spec,
    )

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-3)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    assert cos > 0.9999, f"cosine {cos:.6f}"


def test_scales_are_required_with_a_quantized_slab():
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention

    c = _case(Q4_0, torch.float16)
    with pytest.raises(ValueError, match="without scales"):
        qsa_sparse_paged_attention(
            c["q"], c["codes"][0], c["codes"][1], c["indices"], c["block_table"],
            c["token_to_req"], None, kv_quant=Q4_0,
        )


def test_the_unquantized_path_still_works():
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention

    torch.manual_seed(12)
    c = _case(Q8_0, torch.bfloat16)
    a = qsa_sparse_paged_attention(
        c["q"], c["oracle"][0], c["oracle"][1], c["indices"], c["block_table"], c["token_to_req"]
    )
    b = qsa_sparse_paged_attention(
        c["q"], c["oracle"][0], c["oracle"][1], c["indices"], c["block_table"], c["token_to_req"]
    )
    assert torch.equal(a, b) and torch.isfinite(a).all()


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_the_prefill_profile_falls_back_when_it_does_not_fit(spec):
    """The wide tile is tuned on a GB300 (227 KiB of shared memory per block).

    A 3060 has 99 KiB and a 2060 has 64, and Triton refuses the launch outright --
    OutOfResources, raised while it loads the module, before anything runs. That took down
    Qwen3.8-Flash-Next on a 3060 pair on the first request: the boot was clean, the decode
    profile (the narrow tile) captured and replayed, and nothing looked wrong until someone
    asked a question. So the wide profile has to be an offer, not a requirement.

    This drives the wide branch on purpose (base_programs > 512) with Flash-Next's own
    geometry -- 24 query heads over 2 kv heads, head_dim 256 -- and asks for the answer.
    """
    from freetoken.kernel.triton.qsa import attend

    torch.manual_seed(5)
    attend._ACCEPTED.clear()
    attend._REFUSED.clear()
    c = _case(spec, torch.bfloat16, head_dim=256, kv_heads=2, q_heads=24, pages=16,
              page_size=64, topk=256, n_rows=300)

    want = attend.qsa_sparse_paged_attention(
        c["q"], c["oracle"][0], c["oracle"][1], c["indices"], c["block_table"],
        c["token_to_req"],
    )
    got = attend.qsa_sparse_paged_attention(
        c["q"], c["codes"][0], c["codes"][1], c["indices"], c["block_table"],
        c["token_to_req"], None,
        k_scales=c["scales"][0], v_scales=c["scales"][1], kv_quant=spec,
    )

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-3)
    # whatever it settled on is remembered, so the next call of this shape pays no exception
    assert attend._ACCEPTED, "the accepted profile must be cached"
    chosen = {profile[0] for profile in attend._ACCEPTED.values()}
    assert chosen <= {64, 32, 16}


def test_the_ladder_offers_the_tuned_profile_first_then_gives_up_pipelining():
    from freetoken.kernel.triton.qsa.attend import _profile_ladder

    rungs = list(_profile_ladder(("cold",), 64, 1, 2))

    assert rungs[0] == (64, 1, 2, 2), "the tuned profile is tried unchanged"
    assert [(n, s) for n, _, _, s in rungs] == [
        (64, 2), (64, 1), (32, 2), (32, 1), (16, 2), (16, 1)
    ]
    # the narrow tile is the tuned decode profile, which pairs 16 with four warps
    assert [w for n, _, w, _ in rungs if n == 16] == [4, 4]
