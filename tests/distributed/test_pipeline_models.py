"""The pipeline (layer-split) engine on the models after qwen4_exp: qwen3_5_moe (Ornith /
Qwen3.6-35B-A3B) and gpt_oss. Placement and state-dict layout per rank, the draft head on the
head-owning rank, the gpt-oss expert-bank window. CPU only, meta-device models."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def _pp(monkeypatch, rank, size, start, end, num_layers):
    import freetoken.distributed.info as info_mod

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    if size > 1:
        info_mod.set_pp_info(rank=rank, size=size, start=start, end=end, num_layers=num_layers)


def _meta_build(cls, cfg):
    from freetoken.layers import set_rope_device

    set_rope_device(torch.device("cpu"))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return cls(cfg)
    finally:
        torch.set_default_dtype(prev)


# ----------------------------------------------------------------------------------
# qwen3_5_moe: 8 layers, full attention at 3 and 7, offload experts
# ----------------------------------------------------------------------------------


def _qwen35(num_layers=8):
    pytest.importorskip("freetoken.models.qwen3_5_moe.model")
    from tests.models.qwen3_5_moe.test_spec_mtp import _parsed

    return replace(_parsed(num_layers), moe_backend="offload")


def test_qwen35_window_keeps_one_full_layer_per_rank():
    from freetoken.models.config import window_model_config

    full = _qwen35()
    lo = window_model_config(full, 0, 4)
    hi = window_model_config(full, 4, 8, extra_full_layer=8)
    full_lo = next(g for g in lo.attention_groups if g.name == "full")
    full_hi = next(g for g in hi.attention_groups if g.name == "full")
    assert full_lo.layer_ids == (3,) and full_hi.layer_ids == (7, 8)
    lin_lo = lo.linear_attention_group()
    assert lin_lo is not None and lin_lo.layer_ids == (0, 1, 2)
    assert lo.num_moe_layers == 4 and hi.num_moe_layers == 4
    assert hi.mtp_layer_id == 8 and lo.mtp_layer_id is None
    with pytest.raises(ValueError):
        window_model_config(full, 0, 3)  # no full-attention layer before layer 3


def test_qwen35_first_rank_owns_embedding_and_its_layers(monkeypatch):
    from freetoken.models.config import window_model_config
    from freetoken.models.pipeline import RemoteLayer
    from freetoken.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    full = _qwen35()
    _pp(monkeypatch, 0, 2, 0, 4, 8)
    model = _meta_build(Qwen3_5MoEForCausalLM, window_model_config(full, 0, 4))
    assert model.model.pp_first and not model.model.pp_last
    assert model.model.embed_tokens is not None and model.lm_head is None and model.model.norm is None
    assert model.mtp is None and model.pp_hidden_width == 64
    layers = model.model.layers.op_list
    assert all(isinstance(layers[i], RemoteLayer) for i in range(4, 8))
    keys = set(model.state_dict())
    assert "model.embed_tokens.weight" in keys and "lm_head.weight" not in keys
    assert {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")} == {0, 1, 2, 3}
    # the offload cache sees rank-local MoE layer ids
    assert sorted(l.layer_id for l in iter_offload_moe_layers(model)) == [0, 1, 2, 3]


def test_qwen35_last_rank_owns_head_norm_and_draft_head(monkeypatch):
    from freetoken.models.config import window_model_config
    from freetoken.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    full = _qwen35()
    _pp(monkeypatch, 1, 2, 4, 8, 8)
    model = _meta_build(Qwen3_5MoEForCausalLM, window_model_config(full, 4, 8, extra_full_layer=8))
    assert not model.model.pp_first and model.model.pp_last
    assert model.model.embed_tokens is None and model.lm_head is not None and model.model.norm is not None
    assert model.mtp is not None and model.mtp.embed_tokens is not None  # its own copy of the table
    assert getattr(model.mtp.embed_tokens, "host_resident", False)
    assert model.host_resident_prefixes == ("mtp.embed_tokens.",)
    keys = set(model.state_dict())
    assert "model.embed_tokens.weight" not in keys and "mtp.embed_tokens.weight" in keys
    assert {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")} == {4, 5, 6, 7}
    # 4 local decoder MoE layers + the head's, all rank-local
    assert sorted(l.layer_id for l in iter_offload_moe_layers(model)) == [0, 1, 2, 3, 4]
    # the loader duplicates the target embedding for the head's copy
    t = torch.zeros(2, 2)
    out = model.remap_loaded_weight("model.embed_tokens.weight", t, model.state_dict())
    assert [k for k, _ in out] == ["model.embed_tokens.weight", "mtp.embed_tokens.weight"]


def test_qwen35_single_process_is_first_and_last(monkeypatch):
    from freetoken.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM

    _pp(monkeypatch, 0, 1, 0, 8, 8)
    model = _meta_build(Qwen3_5MoEForCausalLM, _qwen35())
    assert model.model.pp_first and model.model.pp_last
    assert model.model.embed_tokens is not None and model.lm_head is not None
    assert model.host_resident_prefixes == ()


# ----------------------------------------------------------------------------------
# gpt_oss: 4 layers, sliding / full alternating, MXFP4 offload experts
# ----------------------------------------------------------------------------------


def _gpt_oss(num_layers=4):
    pytest.importorskip("freetoken.models.gpt_oss.model")
    from freetoken.models.gpt_oss.config import parse_config

    hf = SimpleNamespace(
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        hidden_size=128, vocab_size=256, intermediate_size=64, hidden_act="silu", rms_norm_eps=1e-5,
        tie_word_embeddings=False, num_local_experts=4, num_experts_per_tok=2,
        layer_types=tuple("sliding_attention" if i % 2 == 0 else "full_attention" for i in range(num_layers)),
        sliding_window=128, rope_theta=150000.0, max_position_embeddings=4096,
        quantization_config={"quant_method": "mxfp4"}, attention_bias=True,
        swiglu_limit=7.0, hidden_act_alpha=1.702, model_type="gpt_oss", architectures=["GptOssForCausalLM"],
    )
    return replace(parse_config(hf), moe_backend="offload")


def test_gpt_oss_window_keeps_both_attention_kinds():
    from freetoken.models.config import window_model_config

    full = _gpt_oss()
    lo = window_model_config(full, 0, 2)
    names = {g.name: g.layer_ids for g in lo.attention_groups}
    assert names == {"swa": (0,), "full": (1,)}
    assert lo.num_moe_layers == 2
    with pytest.raises(ValueError):
        window_model_config(full, 0, 1)  # a sliding-only window has no full layer


def test_gpt_oss_ranks_split_the_stack(monkeypatch):
    from freetoken.models.config import window_model_config
    from freetoken.models.gpt_oss.model import GptOssForCausalLM
    from freetoken.models.pipeline import RemoteLayer
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    full = _gpt_oss()
    _pp(monkeypatch, 0, 2, 0, 2, 4)
    first = _meta_build(GptOssForCausalLM, window_model_config(full, 0, 2))
    assert first.model.embed_tokens is not None and first.lm_head is None and first.model.norm is None
    assert first.pp_hidden_width == 128
    assert all(isinstance(first.model.layers.op_list[i], RemoteLayer) for i in (2, 3))
    assert sorted(l.layer_id for l in iter_offload_moe_layers(first)) == [0, 1]

    _pp(monkeypatch, 1, 2, 2, 4, 4)
    last = _meta_build(GptOssForCausalLM, window_model_config(full, 2, 4))
    assert last.model.embed_tokens is None and last.lm_head is not None and last.model.norm is not None
    keys = set(last.state_dict())
    assert {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")} == {2, 3}
    assert "lm_head.weight" in keys and "model.embed_tokens.weight" not in keys
    assert sorted(l.layer_id for l in iter_offload_moe_layers(last)) == [0, 1]


def test_gpt_oss_bank_window_allocates_local_layers(monkeypatch):
    from freetoken.distributed import DistributedInfo
    from freetoken.models.gpt_oss.weight import _bank_window, _empty_mxfp4_triton_banks

    cfg = _gpt_oss()
    tp = DistributedInfo(rank=0, size=1)
    _pp(monkeypatch, 1, 2, 2, 4, 4)
    assert _bank_window(cfg) == (2, 4)
    banks, _ = _empty_mxfp4_triton_banks(cfg, dtype=torch.bfloat16, tp_info=tp)
    assert all(len(per_layer) == 2 for per_layer in banks.values())
    _pp(monkeypatch, 0, 1, 0, 4, 4)
    assert _bank_window(cfg) == (0, 4)
    banks, _ = _empty_mxfp4_triton_banks(cfg, dtype=torch.bfloat16, tp_info=tp)
    assert all(len(per_layer) == 4 for per_layer in banks.values())
