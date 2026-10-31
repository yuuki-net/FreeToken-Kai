"""--spec-mtp-gguf (FreeToken-Kai#6): Qwen3.8-Flash-Next's MTP draft head from a separate GGUF.

The Flash-Next GGUFs carry no MTP layer; Unsloth ships the head on its own (``MTP/mtp-*.gguf``) as block
``num_layers`` with llama.cpp's NextN tensors. The reader names it as the safetensors reader names the
checkpoint's ``mtp.*``. Checked once against Qwen's own bf16 head (Qwen/Qwen3.8-Flash-Next, range-read)
with the real ``shared-Q8_0`` file: every matrix within Q8_0's 0.6-1.2 %, every norm exact, ``eh_proj``
the embedding half first (the other order is 115 % off). Here: a small file this test writes itself."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from freetoken.models.gguf.dequant import GGML_BF16, GGML_F32, GGML_Q8_0, dequantize, quantize_q8_0
from freetoken.models.gguf.rawreader import write_gguf
from freetoken.models.qwen4_exp.gguf import iter_gguf_mtp_weights, mtp_gguf_block, to_gguf

H, HC, R, E, I, Q, KV, IQ, IK = 64, 4, 32, 4, 32, 96, 32, 32, 16
BLOCK = 3


def _q8(w: torch.Tensor):
    """(ggml bytes, what the reader must give back): Q8_0 rows and their dequantized bf16."""
    packed = quantize_q8_0(w.float())
    back = dequantize(packed.reshape(-1), GGML_Q8_0, torch.bfloat16).view(*w.shape)
    return packed.numpy().tobytes(), back


def _write(path, *, block=BLOCK, eh_cols=2 * H, with_embd=False):
    """A NextN head in the llama.cpp converter's layout; returns the reader's expected output."""
    torch.manual_seed(6)
    tensors, want = [], {}
    b = f"blk.{block}"

    def mat(name, rows, cols, key=None):
        raw, back = _q8(torch.randn(rows, cols) * 0.05)
        tensors.append((f"{b}.{name}.weight", (cols, rows), GGML_Q8_0, raw))
        if key:
            want[key] = back
        return back

    def norm(name, n, key):
        stored = torch.randn(n) * 0.1 + 1.0  # the converter's w + 1
        tensors.append((f"{b}.{name}.weight", (n,), GGML_F32, stored.numpy().tobytes()))
        want[key] = stored - 1.0

    def f32(name, shape, key, reshape=None):
        w = torch.randn(*shape)
        tensors.append((f"{b}.{name}.weight", tuple(reversed(shape)), GGML_F32, w.numpy().tobytes()))
        want[key] = w if reshape is None else w.reshape(reshape)

    eh = mat("nextn.eh_proj", H, eh_cols)
    want["mtp.fc_embedding.weight"], want["mtp.fc_hidden.weight"] = eh[:, :H], eh[:, H:]
    norm("nextn.enorm", H, "mtp.pre_fc_norm_embedding.weight")
    norm("nextn.hnorm", HC * H, "mtp.pre_fc_norm_hidden.weight")
    mat("nextn.hc_head_down", R, HC * H, "mtp.hyper_connection_mixer.input_mix_weight_down.weight")
    mat("nextn.hc_head_up", HC * H, R, "mtp.hyper_connection_mixer.input_mix_weight_up.weight")
    norm("nextn.hc_head_norm", HC * H, "mtp.hyper_connection_mixer.hc_norm.weight")
    sa, mlp = "mtp.layers.0.self_attn", "mtp.layers.0.mlp"
    want[f"{sa}.qkv_proj.weight"] = torch.cat([mat("attn_q", Q, H), mat("attn_k", KV, H), mat("attn_v", KV, H)])
    mat("attn_output", H, Q, f"{sa}.o_proj.weight")
    norm("attn_q_norm", 32, f"{sa}.q_norm.weight")
    norm("attn_k_norm", 32, f"{sa}.k_norm.weight")
    iq, ik = torch.randn(IQ, H).bfloat16(), torch.randn(IK, H).bfloat16()
    tensors.append((f"{b}.indexer.q_proj.weight", (H, IQ), GGML_BF16, iq.view(torch.int16).numpy().tobytes()))
    tensors.append((f"{b}.indexer.k_proj.weight", (H, IK), GGML_BF16, ik.view(torch.int16).numpy().tobytes()))
    want[f"{sa}.indexer.index_qk_proj.weight"] = torch.cat([iq, ik])
    norm("indexer.q_norm", 16, f"{sa}.indexer.q_layernorm.weight")
    norm("indexer.k_norm", 16, f"{sa}.indexer.k_layernorm.weight")
    f32("ffn_gate_inp", (E, H), f"{mlp}.gate.weight")
    want[f"{mlp}.shared_expert.gate_up_proj.weight"] = torch.cat([mat("ffn_gate_shexp", I, H), mat("ffn_up_shexp", I, H)])
    mat("ffn_down_shexp", H, I, f"{mlp}.shared_expert.down_proj.weight")
    f32("ffn_gate_inp_shexp", (H,), f"{mlp}.shared_expert_gate.weight", reshape=(1, H))
    for hc, src in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        rows = torch.cat([mat(f"{src}_down", R, HC * H), mat(f"{src}_inject", HC, HC * H)])
        want[f"mtp.layers.0.{hc}.input_mix_weight_down_block_inject.weight"] = torch.cat([rows, rows.new_zeros(-rows.shape[0] % 16, HC * H)])
        mat(f"{src}_up", HC * H, R, f"mtp.layers.0.{hc}.input_mix_weight_up.weight")
        norm(f"{src}_norm", HC * H, f"mtp.layers.0.{hc}.hc_norm.weight")
    experts = {}
    for p, (rows, cols) in {"gate": (I, H), "up": (I, H), "down": (H, I)}.items():
        raw, back = _q8(torch.randn(E * rows, cols) * 0.05)
        tensors.append((f"{b}.ffn_{p}_exps.weight", (cols, rows, E), GGML_Q8_0, raw))
        experts[p] = back.view(E, rows, cols)
    want[f"{mlp}.experts.gate_up_proj"] = torch.cat([experts["gate"], experts["up"]], dim=1)
    want[f"{mlp}.experts.down_proj"] = experts["down"]
    if with_embd:
        raw, back = _q8(torch.randn(8, H))
        tensors.append(("token_embd.weight", (H, 8), GGML_Q8_0, raw))
        want["mtp.embed_tokens.weight"] = back
    write_gguf(str(path), tensors, {"general.architecture": "qwen4exp", "qwen4exp.block_count": block + 1,
                                    "qwen4exp.nextn_predict_layers": 1})
    return want


def _read(path, device="cpu", **kw):
    args = dict(num_layers=BLOCK, hidden=H, hc_count=HC, experts=True)
    args.update(kw)
    return dict(iter_gguf_mtp_weights(str(path), torch.device(device), **args))


def test_the_head_reads_back_under_the_checkpoints_names(tmp_path):
    want = _write(tmp_path / "mtp.gguf")
    got = _read(tmp_path / "mtp.gguf")
    assert sorted(got) == sorted(want)
    for name, t in want.items():
        assert got[name].shape == t.shape, name
        assert torch.equal(got[name].float(), t.float()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ggml's CUDA dequantizers")
def test_ggmls_cuda_dequantizers_give_the_same_bf16(tmp_path):
    _write(tmp_path / "mtp.gguf")
    cpu, cuda = _read(tmp_path / "mtp.gguf"), _read(tmp_path / "mtp.gguf", "cuda")
    for name in cpu:
        assert cuda[name].device.type == "cpu" and torch.equal(cuda[name], cpu[name]), name


def test_a_gguf_target_keeps_the_experts_and_a_rank_without_the_embedding_gets_one(tmp_path):
    want = _write(tmp_path / "mtp.gguf", with_embd=True)
    got = _read(tmp_path / "mtp.gguf", experts=False, embedding_from=str(tmp_path / "unused.gguf"))
    assert not any(".experts." in name for name in got)
    assert torch.equal(got["mtp.embed_tokens.weight"], want["mtp.embed_tokens.weight"])  # the file's own copy first


def test_a_head_for_another_model_is_refused(tmp_path):
    _write(tmp_path / "mtp.gguf")
    with pytest.raises(ValueError, match="block 3, this model has 48"):
        _read(tmp_path / "mtp.gguf", num_layers=48)
    _write(tmp_path / "wide.gguf", eh_cols=3 * H)
    with pytest.raises(ValueError, match="eh_proj"):
        _read(tmp_path / "wide.gguf")


def test_a_file_without_a_head_is_not_an_mtp_gguf(tmp_path):
    path = tmp_path / "experts.gguf"
    write_gguf(str(path), [("blk.0.ffn_gate_inp.weight", (H, E), GGML_F32, bytes(4 * H * E))], {"general.architecture": "qwen4exp"})
    with pytest.raises(ValueError, match="no MTP head"):
        mtp_gguf_block(str(path))
    other = tmp_path / "other.gguf"
    write_gguf(str(other), [], {"general.architecture": "qwen3moe"})
    with pytest.raises(ValueError, match="'qwen3moe' GGUF"):
        mtp_gguf_block(str(other))
    _write(tmp_path / "mtp.gguf")
    assert mtp_gguf_block(str(tmp_path / "mtp.gguf")) == BLOCK


def test_the_heads_layer_maps_to_the_mtp_block_only_in_a_table_that_has_one():
    mtp = {"blk.48.nextn.eh_proj.weight": GGML_Q8_0, "blk.48.ffn_gate_exps.weight": GGML_Q8_0}
    model = {"blk.0.ffn_gate_exps.weight": 21}
    assert to_gguf("mtp.layers.0.mlp.experts.0.gate_proj", mtp) == ("blk.48.ffn_gate_exps",)
    assert to_gguf("mtp.layers.0.self_attn.q_proj", mtp) == ("blk.48.attn_q",)
    assert to_gguf("mtp.layers.0.self_attn.q_proj", model) == ()  # the model's table: the head stays bf16
    assert to_gguf("model.language_model.layers.0.mlp.experts.0.gate_proj", model) == ("blk.0.ffn_gate_exps",)


@dataclass(frozen=True)
class _Config:
    quant: object
    mtp_gguf: str | None = None
    first_k_dense_replace: int = 0
    extra: dict = field(default_factory=dict)


def test_a_gguf_models_head_takes_its_experts_types_from_the_mtp_file(tmp_path):
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.mtp_quant import draft_head_config
    from freetoken.models.register import get_model_spec

    _write(tmp_path / "mtp.gguf")
    spec = get_model_spec("Qwen4ExpGGUFForCausalLM")
    names = NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping)
    model_types = {f"blk.{n}.ffn_{p}_exps.weight": 21 for n in range(BLOCK) for p in ("gate", "up", "down")}  # IQ3_S
    model = GGUFConfig(model_types, to_gguf, name_map=names)

    head = draft_head_config(_Config(model, mtp_gguf=str(tmp_path / "mtp.gguf")), dense_from_model=True).quant
    experts = head.experts
    assert head.expert_prefix == "mtp.layers.0.mlp.experts"
    assert [experts._part_types(n) for n in names.to_checkpoint(head.expert_prefix)] == [(GGML_Q8_0,)] * 3
    assert head.scheme_for("mtp.fc_hidden") is None and head.scheme_for("mtp.layers.0.self_attn.qkv_proj") is None

    plain = draft_head_config(_Config(model), dense_from_model=True).quant
    assert plain.expert_prefix == "model.layers.0.mlp.experts" and plain.experts is model
