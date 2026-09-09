"""Config-time gates on ``--kv-cache-dtype``.

Both of these refuse a combination that would otherwise run and produce wrong numbers
rather than an error, which is the whole reason they exist:

- a backend other than triton/qsa_sparse reads the uint8 code slab as 16-bit floats.
  "auto" resolves to flashinfer on sm_80+, so this is what happens on every card newer
  than the 2060 the feature was developed on, not an exotic case.
- a pool family other than the plain paged one keeps secondary tiers (SWA window, sparse
  index slabs, MLA latents) at 16 bits, read by kernels that never learned the layout.
"""

import pytest
import torch

from tests.engine.test_attention_backend_matrix import _config, _patch_env


@pytest.mark.parametrize("dtype_name", ["q8_0", "q4_0"])
def test_auto_resolved_flashinfer_is_refused_with_a_quantized_kv(monkeypatch, dtype_name):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)  # sm90: auto picks "fa,fi"
    config = _config("full", attention_backend="auto", kv_cache_dtype=dtype_name)
    with pytest.raises(ValueError, match="only read by the"):
        _adjust_config(config)


def test_explicit_triton_is_accepted(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="triton", kv_cache_dtype="q4_0")
    _adjust_config(config)
    assert config.attention_backend == "triton"
    assert config.kv_cache_dtype == "q4_0"


def test_auto_without_the_flag_is_untouched(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="auto")
    _adjust_config(config)
    assert config.attention_backend == "fa,fi"


@pytest.mark.parametrize("kind", ["swa", "bsa", "mla"])
def test_other_pool_families_are_refused_when_the_pool_is_built(kind):
    """The backend gate fires first for these, so the pool gate is checked directly."""
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.kv_quant import Q4_0
    from tests.engine.test_attention_backend_matrix import _model_config

    with pytest.raises(ValueError, match="only implemented for the plain paged pool"):
        create_kvcache_pool(
            model_config=_model_config(kind),
            num_pages=8,
            page_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            kv_quant=Q4_0,
        )


def test_a_head_dim_the_block_does_not_divide_is_refused():
    from freetoken.kvcache.kv_quant import Q4_0

    with pytest.raises(ValueError, match="not a multiple"):
        Q4_0.code_bytes_per_row(100)


def test_qsa_is_accepted_and_quantizes_only_the_paged_slab():
    """Flash-Next: the paged K/V narrows, the index tiers that pick blocks do not."""
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.kv_quant import Q4_0
    from tests.engine.test_attention_backend_matrix import _model_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    pool = create_kvcache_pool(
        model_config=_model_config("qsa"),
        num_pages=16,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_req_slots=2,
        kv_quant=Q4_0,
    )
    assert pool.kv_quant is Q4_0
    assert pool.store_dtype is torch.uint8, "paged K/V should hold codes"
    assert pool.dtype is torch.bfloat16, "the compute dtype must not change"
    # the tiers that decide which blocks are read stay 16-bit
    assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert pool.pending_ring(0).dtype is torch.bfloat16


# A Flash-Next checkpoint resolves to qsa_sparse and *cannot* run on triton -- triton serves
# FULL/SWA, not QSA (tests/engine/test_attention_backend_matrix.py pins ("qsa", "triton") as
# invalid). So a gate that demands triton makes --kv-cache-dtype unreachable on the one model
# family whose fixed-budget reads make it worth having. That is what these two pin.


@pytest.mark.parametrize("dtype_name", ["q8_0", "q4_0"])
def test_flash_next_resolves_to_qsa_sparse_and_keeps_the_flag(monkeypatch, dtype_name):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("qsa", attention_backend="auto", kv_cache_dtype=dtype_name)
    _adjust_config(config)
    assert config.attention_backend == "qsa_sparse"
    assert config.kv_cache_dtype == dtype_name


def test_explicit_qsa_sparse_is_accepted(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("qsa", attention_backend="qsa_sparse", kv_cache_dtype="q4_0")
    _adjust_config(config)
    assert config.attention_backend == "qsa_sparse"
