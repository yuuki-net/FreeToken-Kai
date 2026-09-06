"""Pipeline (layer-split) engine: placement math, config windowing, CLI wiring and the
gloo point-to-point transport. Everything here runs without a model checkpoint; the
transport test needs CUDA (pinned staging) and spawns two processes on one GPU."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import PipelineInfo, pp_layer_range

# ----------------------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------------------


def test_pp_layer_range_even_split():
    assert pp_layer_range(48, None, 0, 2) == (0, 24)
    assert pp_layer_range(48, None, 1, 2) == (24, 48)
    # uneven counts round, every layer is served exactly once
    ranges = [pp_layer_range(49, None, r, 3) for r in range(3)]
    assert ranges[0][0] == 0 and ranges[-1][1] == 49
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_pp_layer_range_explicit_boundaries():
    assert pp_layer_range(48, (20,), 0, 2) == (0, 20)
    assert pp_layer_range(48, (20,), 1, 2) == (20, 48)
    with pytest.raises(ValueError):
        pp_layer_range(48, (20, 30), 0, 2)  # wrong count
    with pytest.raises(ValueError):
        pp_layer_range(48, (48,), 0, 2)  # empty last segment
    with pytest.raises(ValueError):
        pp_layer_range(48, (0,), 0, 2)  # empty first segment


def test_pipeline_info_bank_window():
    info = PipelineInfo(rank=1, size=2, start=24, end=48, num_layers=48)
    assert info.is_last and not info.is_first and not info.is_primary()
    assert info.bank_window(0) == (24, 48)
    # GLM-style leading dense layers shift the bank index
    assert info.bank_window(3) == (21, 45)
    assert PipelineInfo(0, 2, 0, 24, 48).bank_window(3) == (0, 21)


# ----------------------------------------------------------------------------------
# config windowing (a real Qwen3.8-Flash-Next geometry, text-only)
# ----------------------------------------------------------------------------------

_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)
)


def _qwen4_hf_config():
    text = SimpleNamespace(
        dtype="bfloat16",
        eos_token_id=248044,
        full_attention_interval=4,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
        heads_per_ngram=8,
        hidden_act="silu",
        hidden_size=2560,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_kv_heads=1,
        indexer_n_heads=4,
        layer_types=list(_LAYER_TYPES),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        make_ngram_vocab_size_divisible_by=128,
        max_position_embeddings=262144,
        moe_intermediate_size=640,
        ngram_size=3,
        ngram_vocab_size_base=20000000,
        num_attention_heads=24,
        num_experts=512,
        num_experts_per_tok=10,
        num_hidden_layers=48,
        num_key_value_heads=2,
        output_gate_type="sigmoid",
        partial_rotary_factor=0.25,
        ple_conv_kernel_size=4,
        ple_embed_dim=2560,
        ple_layer_ids=[2],
        rms_norm_eps=1e-6,
        rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        shared_expert_intermediate_size=640,
        split_ngram_parts=128,
        tie_word_embeddings=False,
        vocab_size=248320,
    )
    return SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp",
        text_config=text,
        quantization_config={
            "quant_algo": "NVFP4",
            "ignore": [
                "model.embed_tokens", "mtp.*", "*.self_attn.*", "*.linear_attn.*",
                "*.mlp.gate*", "*.mlp.shared_expert.*", "*hyper_connection*", "*.ple.*",
                "lm_head",
            ],
        },
    )


def _parsed_qwen4():
    pytest.importorskip("freetoken.models.qwen4_exp.config")
    from freetoken.models.qwen4_exp.config import parse_config

    return parse_config(_qwen4_hf_config())


def test_window_model_config_halves_qwen4():
    from freetoken.models.config import window_model_config

    full = _parsed_qwen4()
    assert full.num_moe_layers == 48
    lo = window_model_config(full, 0, 24)
    hi = window_model_config(full, 24, 48)

    # layer ids stay global; each half keeps both attention kinds
    for cfg, ids in ((lo, range(0, 24)), (hi, range(24, 48))):
        assert cfg.num_layers == 48
        assert cfg.num_moe_layers == 24
        owned = sorted(l for g in cfg.attention_groups for l in g.layer_ids)
        assert owned == list(ids)
        full_group = next(g for g in cfg.attention_groups if g.name == "full")
        assert full_group.layer_ids == tuple(l for l in ids if (l + 1) % 4 == 0)
        assert full_group.num_index_layers == 6
        assert cfg.is_linear_layer(ids[0]) and not cfg.is_linear_layer(ids[-1])

    # the PLE conv state (HF ple_layer_ids=[2] is one-indexed -> zero-based layer 1) rides
    # only the first half's slot states
    conv_lo = next(s for s in lo.slot_states if s.name == "ple_conv")
    conv_hi = next(s for s in hi.slot_states if s.name == "ple_conv")
    assert conv_lo.layer_ids == (1,) and conv_hi.layer_ids == ()


def test_window_model_config_rejects_segment_without_full_attention():
    from freetoken.models.config import window_model_config

    full = _parsed_qwen4()
    with pytest.raises(ValueError, match="full"):
        window_model_config(full, 0, 3)  # layers 0-2 are all linear_attention


# ----------------------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------------------


def _parse(argv):
    pytest.importorskip("freetoken.server.args")
    from freetoken.server.args import parse_args

    with tempfile.TemporaryDirectory() as d:
        # parse_args reads config.json for the dtype / parser inference
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"],
                       "model_type": "qwen4_exp", "torch_dtype": "bfloat16"}, f)
        args, _ = parse_args(
            ["--model", d, "--dtype", "bfloat16", "--tool-call-parser", "llama3",
             "--reasoning-parser", "off", *argv],
            False,
        )
    return args


def test_pp_size_sets_parallel_and_world_size():
    args = _parse(["--pp-size", "2", "--gpu", "0,1", "--pp-layers", "20"])
    assert args.parallel == "pp"
    assert args.tp_info.size == 2
    assert args.pp_split == (20,)
    assert args.is_pp


def test_pp_size_default_is_tp_single():
    args = _parse([])
    assert args.parallel == "tp" and args.tp_info.size == 1 and not args.is_pp


@pytest.mark.parametrize(
    "argv",
    [
        ["--pp-size", "2", "--tp-size", "2"],
        ["--pp-size", "2", "--pp-layers", "20,30"],
        ["--pp-size", "2", "--pp-layers", "30,20"],
        ["--pp-layers", "20"],
        ["--pp-size", "2", "--gpu", "0"],
    ],
)
def test_pp_cli_rejections(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


# ----------------------------------------------------------------------------------
# transport (two processes over gloo, one GPU)
# ----------------------------------------------------------------------------------


def _comm_worker(rank: int, init_file: str) -> None:
    import torch.distributed as dist

    from freetoken.distributed.pipeline import PipelineComm

    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=2)
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")  # same byte path, unpinned staging
    info = PipelineInfo(rank, 2, 0 if rank == 0 else 5, 5 if rank == 0 else 10, 10)
    comm = PipelineComm(info, dist.group.WORLD, device)
    comm.configure(hidden_width=16, hidden_dtype=torch.bfloat16)
    torch.manual_seed(0)
    rows_list = [3, 1, 7]  # a prefill chunk, a decode step, a bigger chunk (staging regrowth)
    for rows in rows_list:
        expect = (torch.arange(rows * 16, dtype=torch.float32).view(rows, 16) * 0.5).to(torch.bfloat16)
        if rank == 0:
            comm.send_hidden(expect.to(device))
            toks = comm.recv_tokens(rows)
            assert toks.tolist() == [rows * 10 + i for i in range(rows)], toks
        else:
            got = comm.recv_hidden(rows)
            assert got.dtype == torch.bfloat16 and got.shape == (rows, 16)
            assert torch.equal(got.cpu(), expect), (got, expect)
            comm.send_tokens(torch.tensor([rows * 10 + i for i in range(rows)], dtype=torch.int32))
    # prefill overlap: two chunk-only steps carry no tokens back; rank 0 sends the next
    # residual stream before rank 1 has looked at the previous one, then a final chunk with
    # tokens. The two directions use distinct tags, so the order still works out.
    plan = [(2, False), (4, False), (3, True)]
    for step, (rows, tokens) in enumerate(plan):
        expect = torch.full((rows, 16), float(step + 1), dtype=torch.bfloat16)
        if rank == 0:
            comm.send_hidden(expect.to(device))
            if tokens:
                assert comm.recv_tokens(rows).tolist() == [step] * rows
        else:
            got = comm.recv_hidden(rows)
            assert torch.equal(got.cpu(), expect), (step, got)
            if tokens:
                comm.send_tokens(torch.full((rows,), step, dtype=torch.int32))
    dist.destroy_process_group()


def test_pipeline_comm_roundtrip():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(_comm_worker, args=(init_file,), nprocs=2, join=True)


# ----------------------------------------------------------------------------------
# model segmentation (meta device, no weights)
# ----------------------------------------------------------------------------------


def test_qwen4_segment_owns_only_its_layers(monkeypatch):
    pytest.importorskip("freetoken.models.qwen4_exp.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod
    from freetoken.models.config import window_model_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM, _RemoteLayer
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    full = replace(_parsed_qwen4(), moe_backend="offload")
    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    info_mod.set_pp_info(rank=1, size=2, start=24, end=48, num_layers=48)
    cfg = window_model_config(full, 24, 48)
    from freetoken.layers import set_rope_device

    set_rope_device(torch.device("cpu"))  # rope tables cannot live on meta
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)

    assert not model.model.pp_first and model.model.pp_last
    assert model.model.embed_tokens is None and model.lm_head is not None
    assert model.pp_hidden_width == 4 * 2560
    ops = model.model.layers.op_list
    assert all(isinstance(ops[i], _RemoteLayer) for i in range(24))
    assert not any(isinstance(ops[i], _RemoteLayer) for i in range(24, 48))
    keys = model.state_dict().keys()
    assert not any(k.startswith("model.layers.0.") for k in keys)
    assert any(k.startswith("model.layers.47.") for k in keys)
    assert not any(k.startswith("model.embed_tokens") for k in keys)
    # the offload-cache walk must see each local MoE layer exactly once, rank-locally indexed
    moe_layers = list(iter_offload_moe_layers(model))
    assert len(moe_layers) == 24
    assert sorted(l.layer_id for l in moe_layers) == list(range(24))


def test_engine_config_tp_size_is_one_under_pp():
    """The pipeline ranks split layers, never tensors: the KV cost model, the GDN state pool
    and the layers must all shard by 1 (world size 2 halved the conv state and broke decode)."""
    pytest.importorskip("freetoken.engine.config")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    pp = EngineConfig(model_path="x", tp_info=DistributedInfo(1, 2), dtype=torch.bfloat16, parallel="pp")
    tp = EngineConfig(model_path="x", tp_info=DistributedInfo(1, 2), dtype=torch.bfloat16)
    single = EngineConfig(model_path="x", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    assert pp.is_pp and pp.tp_size == 1 and pp.tp_info.size == 2
    assert not tp.is_pp and tp.tp_size == 2
    assert single.tp_size == 1


# ----------------------------------------------------------------------------------
# --dense-quant fp8: load-time per-row fp8 for the bf16 dense projections
# ----------------------------------------------------------------------------------


def test_quantize_fp8_per_row_roundtrip():
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row

    torch.manual_seed(0)
    w = torch.randn(300, 64, dtype=torch.bfloat16) * 0.05
    w[7, 3] = 4.0  # an outlier row must not crush the other rows' precision
    q, s = quantize_fp8_per_row(w, chunk_rows=128)  # chunking crosses row 256
    assert q.dtype == torch.float8_e4m3fn and s.shape == (300,)
    deq = q.float() * s[:, None]
    err = (deq - w.float()).abs() / w.float().abs().clamp(min=1e-3)
    assert err[torch.arange(300) != 7].median() < 0.03
    assert torch.isclose(deq[7, 3], torch.tensor(4.0), rtol=0.05)


def test_dense_quant_override_wires_fp8_layers(monkeypatch):
    """The CLI override turns every bf16 dense projection of qwen4_exp into its per-row fp8
    twin (with a weight_scale buffer), while the NVFP4 routed experts keep their format."""
    pytest.importorskip("freetoken.models.qwen4_exp.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod
    from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged, Fp8PerTensorLinear
    from freetoken.layers import Fp8ParallelLMHead, Fp8VocabParallelEmbedding, set_rope_device
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    base = replace(_parsed_qwen4(), moe_backend="offload")
    assert base.attn_quant == "none" and base.expert_quant == "nvfp4"
    cfg = replace(base, attn_quant="fp8_pertensor", dense_quant="fp8_pertensor",
                  lm_head_quant="fp8_pertensor", embed_quant="fp8_pertensor")
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch.no_grad():
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = Qwen4ExpForCausalLM(cfg)
        finally:
            torch.set_default_dtype(prev)

    layer0, layer3 = model.model.layers.op_list[0], model.model.layers.op_list[3]
    assert isinstance(layer3.self_attn.qkv_proj, Fp8PerTensorColMerged)
    assert isinstance(layer3.self_attn.o_proj, Fp8PerTensorLinear)
    assert isinstance(layer0.linear_attn.in_proj_qkvz, Fp8PerTensorColMerged)
    assert isinstance(layer0.linear_attn.out_proj, Fp8PerTensorLinear)
    assert isinstance(layer0.mlp.shared_expert.gate_up_proj, Fp8PerTensorColMerged)
    assert isinstance(model.lm_head, Fp8ParallelLMHead)
    assert isinstance(model.model.embed_tokens, Fp8VocabParallelEmbedding)
    keys = model.state_dict()
    assert keys["model.layers.0.linear_attn.in_proj_qkvz.weight"].dtype == torch.float8_e4m3fn
    assert "model.layers.0.linear_attn.in_proj_qkvz.weight_scale" in keys
    assert keys["model.layers.0.linear_attn.in_proj_ba.weight"].dtype == torch.bfloat16
    # HC / PLE / norms stay bf16
    assert keys["model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"].dtype == torch.bfloat16

    # the loader hook splits the reader's fused in_proj the way the fp8 GDN declares it
    fused = torch.zeros(16384 + 96, 2560, dtype=torch.bfloat16)
    out = model.remap_loaded_weight("model.layers.0.linear_attn.in_proj.weight", fused, keys)
    assert [k.rsplit(".", 2)[-2] for k, _ in out] == ["in_proj_qkvz", "in_proj_ba"]
    assert out[0][1].shape[0] == 16384 and out[1][1].shape[0] == 96


def test_quantize_at_load_emits_scale_for_fp8_buffers():
    pytest.importorskip("freetoken.engine.engine")
    from freetoken.engine.engine import _quantize_at_load

    model_state = {
        "a.weight": torch.empty(4, 8, dtype=torch.float8_e4m3fn),
        "a.weight_scale": torch.empty(4, dtype=torch.float32),
        "b.weight": torch.empty(4, 8, dtype=torch.bfloat16),
    }
    loaded = [("a.weight", torch.randn(4, 8, dtype=torch.bfloat16)),
              ("b.weight", torch.randn(4, 8, dtype=torch.bfloat16))]
    out = dict(_quantize_at_load(iter(loaded), model_state))
    assert out["a.weight"].dtype == torch.float8_e4m3fn and out["a.weight_scale"].shape == (4,)
    assert out["b.weight"].dtype == torch.bfloat16  # bf16 buffer: untouched


def test_engine_config_dense_quant_override(monkeypatch):
    pytest.importorskip("freetoken.engine.config")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig

    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: SimpleNamespace(module="m", parse_config="p"))
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: (lambda hf: _parsed_qwen4()))
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path: _qwen4_hf_config())
    ec = EngineConfig(model_path="x", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, dense_quant="fp8")
    mc = ec.model_config
    assert (mc.attn_quant, mc.dense_quant, mc.lm_head_quant, mc.embed_quant) == ("fp8_pertensor",) * 4
    assert mc.expert_quant == "nvfp4"  # the routed experts keep the checkpoint's format
    plain = EngineConfig(model_path="x", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    assert plain.model_config.attn_quant == "none"


def test_fp8_embedding_forward_matches_bf16_table(monkeypatch):
    import freetoken.distributed.info as info_mod
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row
    from freetoken.layers import Fp8VocabParallelEmbedding

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    monkeypatch.setattr(info_mod, "_PP_INFO", None)
    info_mod.set_tp_info(0, 1)
    if not torch.cuda.is_available():  # nvtx_annotate needs the CUDA build's NVTX; the math does not
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda *a, **k: None)
    torch.manual_seed(1)
    table = torch.randn(50, 32, dtype=torch.bfloat16)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        emb = Fp8VocabParallelEmbedding(50, 32)
    finally:
        torch.set_default_dtype(prev)
    emb.weight, emb.weight_scale = quantize_fp8_per_row(table)
    ids = torch.tensor([0, 7, 49, 7], dtype=torch.int32)
    out = emb.forward(ids)
    assert out.dtype == torch.bfloat16 and out.shape == (4, 32)
    ref = table[ids.long()].float()
    assert ((out.float() - ref).abs() / ref.abs().clamp(min=1e-2)).median() < 0.05
    assert torch.equal(out[1], out[3])
