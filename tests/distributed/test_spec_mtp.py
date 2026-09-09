"""MTP speculative decoding: host-side arithmetic, request bookkeeping, the NVFP4 quantizer
and the draft head's wiring against the checkpoint's mtp.* key list. No GPU, no weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .test_pipeline import _parse, _parsed_qwen4, _qwen4_hf_config

# what the reader yields for RadixArk/Qwen3.8-Flash-Next-NVFP4's mtp.* tensors after its
# fusions (q|k|v -> qkv_proj, shared gate|up -> gate_up_proj, HC down|inject ->
# input_mix_weight_down_block_inject); the stacked experts go to the bank instead
_FUSED_MTP_KEYS = {
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.qkv_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def test_accept_drafts_rule():
    from freetoken.engine.spec import accept_drafts

    # all three drafts confirmed -> 4 tokens (3 drafts + the bonus sample)
    assert accept_drafts([5, 6, 7, 8], [5, 6, 7]) == [5, 6, 7, 8]
    # first mismatch at row 1 -> the row-1 sample is the correction
    assert accept_drafts([5, 9, 7, 8], [5, 6, 7]) == [5, 9]
    assert accept_drafts([1, 2], [3]) == [1]
    assert accept_drafts([1], []) == [1]


def test_spec_message_roundtrip():
    from freetoken.engine.spec import (
        SpecResult,
        pack_spec_message,
        spec_message_len,
        unpack_spec_message,
    )

    k = 3
    for res in (SpecResult([5, 6, 7, 8], [1, 2, 3]), SpecResult([5], [9]), SpecResult([5, 9], [])):
        vec = pack_spec_message(res, k)
        assert vec.numel() == spec_message_len(k) and vec.dtype == torch.int32
        back = unpack_spec_message(vec, k)
        assert back.accepted == res.accepted and back.drafts == res.drafts


def test_pages_to_free_and_state_helpers():
    from freetoken.engine.spec import ngram_context_after, pages_to_free, rebuild_conv_state

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

    # window [t_last=13, d1=21, d2=22]; keeping a=2 rows processed ids 13, 21
    assert ngram_context_after([10, 11, 12, 13], [21, 22], accepted=2, ctx_len=2, boundary=0) == [13, 21]
    assert ngram_context_after([10, 11, 12, 13], [21, 22], accepted=1, ctx_len=2, boundary=0) == [12, 13]
    assert ngram_context_after([13], [21, 22], accepted=1, ctx_len=2, boundary=0) == [0, 13]


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
    # 4-bit with 16-wide block scales: the block amax is exact-ish, the rest within a step
    assert torch.isclose(deq[0, 5], torch.tensor(3.0), rtol=0.08)
    assert (deq - w).abs().mean() < 0.02
    # E=4 experts, H=32, I=16: gate_up [E, 2I, H], down [E, H, I]
    exp = quantize_nvfp4_experts(torch.randn(4, 32, 32), torch.randn(4, 32, 16), chunk=3)
    assert exp["gate_up_packed"].shape == (4, 32, 16) and exp["down_packed"].shape == (4, 32, 8)
    assert exp["gate_up_scale"].shape == (4, 32, 2) and exp["down_global"].shape == (4, 32)
    assert exp["gate_up_global"].shape == (4, 32) and exp["down_scale"].shape == (4, 32, 1)
    # host-resident inputs streamed chunk-wise into caller-owned destinations give the same bytes
    from freetoken.kernel.triton.nvfp4_quant import nvfp4_expert_bank_specs

    gu, dn = torch.randn(4, 32, 32), torch.randn(4, 32, 16)
    direct = quantize_nvfp4_experts(gu, dn, chunk=4)
    dest = {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in nvfp4_expert_bank_specs(4, 32, 16).items()}
    streamed = quantize_nvfp4_experts(gu, dn, chunk=1, device=torch.device("cpu"), out=dest)
    assert streamed is dest
    for n in direct:
        assert torch.equal(direct[n].view(torch.uint8), dest[n].view(torch.uint8)), n


def _build(cfg):
    from freetoken.layers import set_rope_device
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    set_rope_device(torch.device("cpu"))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return Qwen4ExpForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)


def test_mtp_head_matches_checkpoint_keys(monkeypatch):
    """Under --spec-mtp on the last pipeline rank the model grows a draft head whose state dict
    is exactly the checkpoint's (fused) mtp.* key set, plus its own embedding copy, and the
    head's MoE layer is the rank's last bank layer."""
    pytest.importorskip("freetoken.models.qwen4_exp.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod
    from freetoken.models.config import window_model_config
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    info_mod.set_pp_info(rank=1, size=2, start=24, end=48, num_layers=48)
    full = replace(_parsed_qwen4(), moe_strategy="offload")
    cfg = window_model_config(full, 24, 48, extra_full_layer=48)
    assert cfg.mtp_layer_id == 48
    full_group = next(g for g in cfg.attention_groups if g.name == "full")
    assert full_group.layer_ids[-1] == 48 and full_group.num_index_layers == 7
    assert not cfg.is_linear_layer(48)
    model = _build(cfg)
    assert model.mtp is not None
    assert model.mtp.embed_tokens is not None  # rank 1 holds no target embedding -> own copy
    assert getattr(model.mtp.embed_tokens, "host_resident", False)  # ... kept in pinned RAM
    assert model.host_resident_prefixes == ("mtp.embed_tokens.",)
    keys = model.state_dict()
    mtp_keys = {k for k in keys if k.startswith("mtp.")}
    expected = _FUSED_MTP_KEYS | {"mtp.embed_tokens.weight"}
    assert mtp_keys == expected, sorted(mtp_keys ^ expected)
    assert keys["mtp.pre_fc_norm_hidden.weight"].shape == (4 * 2560,)
    assert keys["mtp.fc_hidden.weight"].shape == (2560, 2560)
    assert keys["mtp.layers.0.self_attn.qkv_proj.weight"].shape == (12288 + 512 + 512, 2560)
    assert keys["mtp.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (1280, 2560)
    # the head's MoE layer is the 25th (local id 24) bank layer of this rank
    moe = list(iter_offload_moe_layers(model))
    assert sorted(l.layer_id for l in moe) == list(range(25))
    # the loader hook duplicates the target embedding for the head's copy
    dup = model.remap_loaded_weight("model.embed_tokens.weight", torch.zeros(2, 2), keys)
    assert [k for k, _ in dup] == ["model.embed_tokens.weight", "mtp.embed_tokens.weight"]


def test_mtp_head_absent_without_spec(monkeypatch):
    pytest.importorskip("freetoken.models.qwen4_exp.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    model = _build(replace(_parsed_qwen4(), moe_strategy="offload"))
    assert model.mtp is None
    assert not any(k.startswith("mtp.") for k in model.state_dict())


def test_engine_config_spec_mtp_adds_head_layer_on_last_rank(monkeypatch):
    pytest.importorskip("freetoken.engine.config")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig

    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: SimpleNamespace(module="m", parse_config="p"))
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: (lambda hf: _parsed_qwen4()))
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path: _qwen4_hf_config())
    monkeypatch.setattr(cfg_mod, "checkpoint_quant_config", lambda *a, **k: None)
    kw = dict(model_path="x", dtype=torch.bfloat16)
    last = EngineConfig(tp_info=DistributedInfo(1, 2), parallel="pp", spec_mtp=3, **kw)
    first = EngineConfig(tp_info=DistributedInfo(0, 2), parallel="pp", spec_mtp=3, **kw)
    single = EngineConfig(tp_info=DistributedInfo(0, 1), spec_mtp=2, **kw)
    assert last.model_config.mtp_layer_id == 48 and first.model_config.mtp_layer_id is None
    assert single.model_config.mtp_layer_id == 48
    assert single.model_config.num_moe_layers == 48  # the bank layer is appended by the engine


def test_spec_mtp_cli_flag():
    pytest.importorskip("freetoken.server.args")
    args = _parse(["--pp-size", "2", "--gpu", "0,1", "--spec-mtp", "3"])
    assert args.spec_mtp == 3
    assert _parse([]).spec_mtp == 0


def test_ple_disk_stages_draft_window_ids():
    """The disk PLE backend hashes its windows on the host: an open verify window must stage
    ``[t_last, *drafts]`` even though the drafts only live in the device token pool."""
    import torch
    from types import SimpleNamespace
    from freetoken.models.qwen4_exp.ple_disk import _extend_ids

    ids = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    plain = SimpleNamespace(input_ids=ids, cached_len=3, device_len=4, spec_drafts=[], spec_base_len=None)
    assert _extend_ids(plain).tolist() == [8]
    window = SimpleNamespace(input_ids=ids, cached_len=3, device_len=7, spec_drafts=[11, 12, 13], spec_base_len=4)
    assert _extend_ids(window).tolist() == [8, 11, 12, 13]
    chunk = SimpleNamespace(input_ids=ids, cached_len=1, device_len=4, spec_drafts=[], spec_base_len=None)
    assert _extend_ids(chunk).tolist() == [6, 7, 8]


def test_channels_first_copy_never_aliases_the_conv_input():
    """A [dim, 1] transposed view counts as contiguous, so transpose().contiguous() returned the
    same storage and the in-place conv kernel overwrote the raw inputs the rollback needs."""
    import torch
    from freetoken.models.qwen4_exp.gdn import channels_first_copy

    for total in (1, 4):
        conv_in = torch.arange(total * 6, dtype=torch.float32).reshape(total, 6)
        x = channels_first_copy(conv_in)
        assert x.shape == (6, total) and x.is_contiguous() and x.stride(-1) == 1
        assert x.data_ptr() != conv_in.data_ptr()
        x.fill_(-1.0)
        assert torch.equal(conv_in, torch.arange(total * 6, dtype=torch.float32).reshape(total, 6))
    # the old expression really did alias for total == 1
    one = torch.zeros(1, 6)
    assert one.transpose(0, 1).contiguous().data_ptr() == one.data_ptr()


def test_materialize_places_host_prefixes_on_cpu():
    import torch
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict

    model_state = {"mtp.embed_tokens.weight": torch.empty(2, 2, device="meta"), "x.weight": torch.empty(2, 2, device="meta")}
    out = _materialize_loaded_weight_state_dict(
        model_state,
        [("mtp.embed_tokens.weight", torch.ones(2, 2)), ("x.weight", torch.ones(2, 2))],
        device=torch.device("cpu"),
        host_prefixes=("mtp.embed_tokens.",),
    )
    assert out["mtp.embed_tokens.weight"].device.type == "cpu" and out["x.weight"].device.type == "cpu"
    assert torch.equal(out["mtp.embed_tokens.weight"], torch.ones(2, 2))


def test_spec_graph_applicable_only_for_full_windows():
    from types import SimpleNamespace
    from freetoken.engine.spec_graph import spec_graph_applicable

    def batch(rows, cached=100, verify=True, n=1):
        req = SimpleNamespace(cached_len=cached, device_len=cached + rows)
        return SimpleNamespace(spec_verify=verify, reqs=[req] * n)

    assert spec_graph_applicable(batch(6), 6, 6)
    assert not spec_graph_applicable(batch(4), 4, 6)          # short window (budget tail)
    assert not spec_graph_applicable(batch(6, verify=False), 6, 6)
    assert not spec_graph_applicable(batch(6, cached=0), 6, 6)  # no cached prefix to continue
    assert not spec_graph_applicable(batch(6, n=2), 6, 6)
