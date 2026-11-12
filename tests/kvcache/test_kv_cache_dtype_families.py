"""--kv-cache-dtype has two families: Kai's block formats (q8_0 / q4_0, kvcache/kv_quant.py) and
upstream's per-token-head formats (kvcache/format.py). One flag takes both; one of them is on."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.engine.config import EngineConfig
from freetoken.kvcache.format import FP8_PER_TOKEN_HEAD, INT8_PER_TOKEN_HEAD, NONE
from freetoken.kvcache.kv_quant import Q4_0, Q8_0, resolve


@pytest.fixture(autouse=True)
def _one_rank(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1))


def _kv_format(name):
    return EngineConfig.kv_format.fget(SimpleNamespace(kv_cache_dtype=name))


@pytest.mark.parametrize(
    "name, kai, upstream",
    [
        (None, None, NONE),
        ("auto", None, NONE),
        ("q8_0", Q8_0, NONE),
        ("q4_0", Q4_0, NONE),
        ("fp8_per_token_head", None, FP8_PER_TOKEN_HEAD),
        ("int8_per_token_head", None, INT8_PER_TOKEN_HEAD),
    ],
)
def test_each_value_turns_on_at_most_one_family(name, kai, upstream):
    assert resolve(name) is kai
    assert _kv_format(name) is upstream


def test_an_unknown_value_names_both_families():
    with pytest.raises(ValueError) as e:
        resolve("q3_k")
    assert "q8_0" in str(e.value) and "fp8_per_token_head" in str(e.value)


def test_the_engine_config_default_is_auto():
    import dataclasses

    fields = [f for f in dataclasses.fields(EngineConfig) if f.name == "kv_cache_dtype"]
    assert len(fields) == 1 and fields[0].default == "auto"


@pytest.mark.parametrize("value", ["auto", "q8_0", "q4_0", "fp8_per_token_head", "int8_per_token_head"])
def test_the_cli_takes_both_families(value):
    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args = parse_args(["--model", "/models/anon", "--kv-cache-dtype", value])[0]
    assert args.kv_cache_dtype == value


def test_a_pool_refuses_both_families_at_once():
    from freetoken.kvcache.mha_pool import MHAKVCache

    with pytest.raises(ValueError, match="one format"):
        MHAKVCache(num_kv_heads=2, num_layers=2, head_dim=32, num_pages=4, page_size=1,
                   dtype=torch.bfloat16, device=torch.device("cpu"), kv_quant=Q8_0, kv_format=INT8_PER_TOKEN_HEAD)


@pytest.mark.parametrize("kai, upstream", [(Q8_0, NONE), (None, INT8_PER_TOKEN_HEAD), (None, NONE)])
def test_each_family_reads_through_its_own_scales(kai, upstream):
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(num_kv_heads=2, num_layers=2, head_dim=32, num_pages=4, page_size=2,
                      dtype=torch.bfloat16, device=torch.device("cpu"), kv_quant=kai, kv_format=upstream)
    if kai is not None:
        assert pool.k_scales(1).shape[-1] == kai.blocks_per_row(32)
        assert pool.k_scale(1) is None
    elif upstream.quantized:
        assert pool.k_scale(1).shape == (4 * 2, 2, 1)
        with pytest.raises(AssertionError):
            pool.k_scales(1)
    else:
        assert pool.k_scale(1) is None
        assert pool.store_dtype == torch.bfloat16
    assert pool.dtype == torch.bfloat16
