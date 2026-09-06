
"""MTP speculative decoding for the Qwen3.5-MoE family: host-side arithmetic, request
bookkeeping, the NVFP4 quantizer, the model-config seam and the draft head's wiring against
the checkpoint's mtp.* key list. No GPU, no weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# what the reader yields for Ornith-1.5-35B-A3B-NVFP4's mtp.* tensors after its fusions
# (q|k|v -> qkv_proj, shared gate|up -> gate_up_proj); the stacked experts go to the bank
_FUSED_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.qkv_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
}


def _toy_hf_config(num_layers: int = 4):
    """A scaled-down Ornith-shaped HF config (full attention every 4th layer, 8 experts)."""
    text = SimpleNamespace(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        max_position_embeddings=512, num_hidden_layers=num_layers, rms_norm_eps=1e-6,
        hidden_act="silu", vocab_size=1000, intermediate_size=0,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, tie_word_embeddings=False,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
    )
    return SimpleNamespace(
        text_config=text, model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"], image_token_id=248056,
        quantization_config={
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                "model.language_model.layers.0.mlp.experts.0.gate_proj": {"quant_algo": "W4A16_NVFP4"},
                "lm_head": {"quant_algo": "W4A16_NVFP4"},
            },
        },
    )


def _parsed(num_layers: int = 4):
    from freetoken.models.qwen3_5_moe.config import parse_config

    return parse_config(_toy_hf_config(num_layers))


def test_accept_drafts_rule():
    from freetoken.engine.spec import accept_drafts

    # all three drafts confirmed -> 4 tokens (3 drafts + the bonus sample)
    assert accept_drafts([5, 6, 7, 8], [5, 6, 7]) == [5, 6, 7, 8]
    # first mismatch at row 1 -> the row-1 sample is the correction
    assert accept_drafts([5, 9, 7, 8], [5, 6, 7]) == [5, 9]
    assert accept_drafts([1, 2], [3]) == [1]
    assert accept_drafts([1], []) == [1]


def test_pages_to_free_and_conv_rebuild():
    from freetoken.engine.spec import pages_to_free, rebuild_conv_state

    assert pages_to_free(keep_len=100, alloc_len=100, page_size=64) == (2, 2)
    assert pages_to_free(keep_len=100, alloc_len=130, page_size=64) == (2, 3)
    assert pages_to_free(keep_len=128, alloc_len=129, page_size=64) == (2, 3)
    assert pages_to_free(keep_len=129, alloc_len=129, page_size=64) == (3, 3)

    prev = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])  # [dim=2, K-1=3]
    conv_in = torch.tensor([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0]])  # [T=4, dim]
    assert torch.equal(rebuild_conv_state(prev, conv_in, 0), prev)
    assert torch.equal(
        rebuild_conv_state(prev, conv_in, 2), torch.tensor([[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]])
    )
    assert torch.equal(
        rebuild_conv_state(prev, conv_in, 4), torch.tensor([[5.0, 6.0, 7.0], [50.0, 60.0, 70.0]])
    )


def test_req_spec_window_bookkeeping():
    from freetoken.core import Req

    req = Req(
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=20, uid=1, sampling_params=None, cache_handle=None,
    )
    req.complete_one()  # the prefill ran: cached 4, device 5, the sampled token pending
    req.append_host(torch.tensor([7], dtype=torch.int32))
    assert (req.cached_len, req.device_len, req.extend_len) == (4, 5, 1)
    req.spec_extend([8, 9, 10])
    assert (req.cached_len, req.device_len, req.extend_len) == (4, 8, 4)
    assert req.remain_len == 20 + 4 - 5  # judged from the committed length
    req.spec_commit([8, 9, 42])  # two drafts confirmed, 42 is the correction
    assert req.input_ids.tolist() == [1, 2, 3, 4, 7, 8, 9, 42]
    assert (req.cached_len, req.device_len, req.extend_len) == (7, 8, 1)
    assert req.spec_base_len is None and req.spec_drafts == []


def test_nvfp4_quantizer_roundtrip_and_layout():
    from freetoken.kernel.triton.nvfp4_quant import (
        dequant_nvfp4_rows,
        nvfp4_expert_bank_specs,
        quantize_nvfp4_experts,
        quantize_nvfp4_rows,
    )

    torch.manual_seed(0)
    w = torch.randn(6, 64) * 0.1
    w[0, 5] = 3.0
    packed, scale, g = quantize_nvfp4_rows(w)
    assert packed.shape == (6, 32) and packed.dtype == torch.uint8
    assert scale.shape == (6, 4) and scale.dtype == torch.float8_e4m3fn
    assert g.shape == (6,) and g.dtype == torch.float16
    deq = dequant_nvfp4_rows(packed, scale, g)
    assert torch.isclose(deq[0, 5], torch.tensor(3.0), rtol=0.08)
    assert (deq - w).abs().mean() < 0.02
    # E=4 experts, H=32, I=16: gate_up [E, 2I, H], down [E, H, I]
    exp = quantize_nvfp4_experts(torch.randn(4, 32, 32), torch.randn(4, 32, 16), chunk=3)
    assert exp["gate_up_packed"].shape == (4, 32, 16) and exp["down_packed"].shape == (4, 32, 8)
    assert exp["gate_up_scale"].shape == (4, 32, 2) and exp["down_global"].shape == (4, 32)
    gu, dn = torch.randn(4, 32, 32), torch.randn(4, 32, 16)
    direct = quantize_nvfp4_experts(gu, dn, chunk=4)
    dest = {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in nvfp4_expert_bank_specs(4, 32, 16).items()}
    streamed = quantize_nvfp4_experts(gu, dn, chunk=1, device=torch.device("cpu"), out=dest)
    assert streamed is dest
    for n in direct:
        assert torch.equal(direct[n].view(torch.uint8), dest[n].view(torch.uint8)), n


def test_stack_mtp_experts_layout():
    from freetoken.models.qwen3_5_moe.weight import _MTP_EXPERT_RE, _stack_mtp_experts

    m = _MTP_EXPERT_RE.match("mtp.layers.0.mlp.experts.17.up_proj.weight")
    assert m is not None and (m.group("expert"), m.group("proj")) == ("17", "up_proj")
    assert _MTP_EXPERT_RE.match("model.language_model.layers.3.mlp.experts.17.up_proj.weight") is None
    inter, hidden = 4, 6
    parts = {
        e: {
            "gate_proj": torch.full((inter, hidden), float(10 * e + 1)),
            "up_proj": torch.full((inter, hidden), float(10 * e + 2)),
            "down_proj": torch.full((hidden, inter), float(10 * e + 3)),
        }
        for e in range(3)
    }
    out = dict(_stack_mtp_experts(parts))
    assert not parts  # consumed
    gu, dn = out["mtp.layers.0.mlp.experts.gate_up_proj"], out["mtp.layers.0.mlp.experts.down_proj"]
    assert gu.shape == (3, 2 * inter, hidden) and dn.shape == (3, hidden, inter)
    assert gu[2, 0, 0] == 21 and gu[2, inter, 0] == 22 and dn[1, 0, 0] == 13
    assert _stack_mtp_experts({}) == []


def test_expand_sampling_args():
    from freetoken.engine.engine import _expand_sampling_args
    from freetoken.engine.sample import BatchSamplingArgs

    greedy = BatchSamplingArgs(None)
    assert _expand_sampling_args(greedy, 4) is greedy
    one = BatchSamplingArgs(torch.tensor([0.5]), top_k=torch.tensor([20]), top_p=None)
    four = _expand_sampling_args(one, 4)
    assert four.temperatures.tolist() == [0.5] * 4 and four.top_k.tolist() == [20] * 4
    assert four.top_p is None and four.temperatures.is_contiguous()


def test_with_mtp_layer_joins_the_full_attention_group():
    from freetoken.models.config import with_mtp_layer

    cfg = _parsed(8)
    assert cfg.mtp_layer_id is None
    full = next(g for g in cfg.attention_groups if g.name == "full")
    assert full.layer_ids == (3, 7)
    with_head = with_mtp_layer(cfg, 8)
    assert with_head.mtp_layer_id == 8 and with_head.num_layers == 8
    full2 = next(g for g in with_head.attention_groups if g.name == "full")
    assert full2.layer_ids == (3, 7, 8)
    linear2 = next(g for g in with_head.attention_groups if g.name == "linear")
    assert linear2.layer_ids == (0, 1, 2, 4, 5, 6)
    assert not with_head.is_linear_layer(8) and with_head.is_linear_layer(6)
    # the paged-KV group spec (what sizes the pool) sees the extra slab
    spec = [s for s in with_head.kv_cache_group_specs() if s.num_layers > 0]
    assert len(spec) == 1 and spec[0].layer_ids == (3, 7, 8)


def test_engine_config_spec_mtp_adds_head_layer(monkeypatch):
    pytest.importorskip("freetoken.engine.config")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig

    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: SimpleNamespace(module="m", parse_config="p"))
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: (lambda hf: _parsed(4)))
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path: _toy_hf_config(4))
    kw = dict(model_path="x", dtype=torch.bfloat16, tp_info=DistributedInfo(0, 1))
    assert EngineConfig(spec_mtp=0, **kw).model_config.mtp_layer_id is None
    spec = EngineConfig(spec_mtp=2, **kw).model_config
    assert spec.mtp_layer_id == 4 and spec.num_layers == 4
    assert spec.num_moe_layers == 4  # the bank layer is appended by the engine, not counted here


def _parse(argv):
    import json
    import os
    import tempfile

    args_mod = pytest.importorskip("freetoken.server.args")
    with tempfile.TemporaryDirectory() as d:
        # parse_args reads config.json for the dtype / parser inference
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                       "model_type": "qwen3_5_moe", "torch_dtype": "bfloat16"}, f)
        args, _ = args_mod.parse_args(
            ["--model", d, "--dtype", "bfloat16", "--tool-call-parser", "llama3",
             "--reasoning-parser", "off", *argv],
            False,
        )
    return args


def test_spec_mtp_cli_flag():
    assert _parse(["--spec-mtp", "3"]).spec_mtp == 3
    assert _parse([]).spec_mtp == 0


def test_channels_first_copy_never_aliases_the_conv_input():
    gdn = pytest.importorskip("freetoken.models.qwen3_5_moe.gdn")

    for total in (1, 4):
        conv_in = torch.arange(total * 6, dtype=torch.float32).reshape(total, 6)
        x = gdn.channels_first_copy(conv_in)
        assert x.shape == (6, total) and x.is_contiguous() and x.stride(-1) == 1
        assert x.data_ptr() != conv_in.data_ptr()
        x.fill_(-1.0)
        assert torch.equal(conv_in, torch.arange(total * 6, dtype=torch.float32).reshape(total, 6))
    # the old expression really did alias for total == 1
    one = torch.zeros(1, 6)
    assert one.transpose(0, 1).contiguous().data_ptr() == one.data_ptr()


def _build(cfg):
    from freetoken.layers import set_rope_device
    from freetoken.models.qwen3_5_moe.model import Qwen3_5MoEForCausalLM

    set_rope_device(torch.device("cpu"))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return Qwen3_5MoEForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)


def test_mtp_head_matches_checkpoint_keys(monkeypatch):
    """Under --spec-mtp the model grows a draft head whose state dict is exactly the
    checkpoint's (fused) mtp.* key set, built unquantized (bf16 in the checkpoint), and the
    head's MoE layer is the last bank layer."""
    pytest.importorskip("freetoken.models.qwen3_5_moe.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod
    from freetoken.models.config import with_mtp_layer
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    info_mod.set_tp_info(0, 1)
    cfg = with_mtp_layer(replace(_parsed(4), moe_backend="offload"), 4)
    model = _build(cfg)
    assert model.mtp is not None and model.mtp.layer_id == 4
    keys = model.state_dict()
    mtp_keys = {k for k in keys if k.startswith("mtp.")}
    assert mtp_keys == _FUSED_MTP_KEYS, sorted(mtp_keys ^ _FUSED_MTP_KEYS)
    assert keys["mtp.fc.weight"].shape == (64, 128)
    # bf16 head: the fused q|k|v is one plain weight (the decoder's is fp8 + weight_scale)
    assert keys["mtp.layers.0.self_attn.qkv_proj.weight"].shape == (4 * 64 * 2 + 2 * 64 + 2 * 64, 64)
    assert keys["mtp.layers.0.self_attn.qkv_proj.weight"].dtype == torch.bfloat16
    assert keys["model.layers.3.self_attn.qkv_proj.weight"].dtype == torch.float8_e4m3fn
    assert keys["mtp.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (64, 64)
    moe = list(iter_offload_moe_layers(model))
    assert sorted(l.layer_id for l in moe) == list(range(5))
    assert moe[-1].layer_id == 4  # walked last: the cache appends the head's bank at the end


def test_mtp_head_absent_without_spec(monkeypatch):
    pytest.importorskip("freetoken.models.qwen3_5_moe.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    info_mod.set_tp_info(0, 1)
    model = _build(replace(_parsed(4), moe_backend="offload"))
    assert model.mtp is None
    assert not any(k.startswith("mtp.") for k in model.state_dict())
