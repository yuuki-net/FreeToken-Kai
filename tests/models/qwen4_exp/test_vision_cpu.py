"""The CPU vision tower loads ``model.visual.*`` from the shards and reproduces the HF module."""

from __future__ import annotations

import json

import pytest
import torch

tcfg = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
tmod = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
safetensors_torch = pytest.importorskip("safetensors.torch")

from freetoken.models.qwen4_exp.vision_cpu import Qwen4ExpCpuVision, load_prefixed_state  # noqa: E402


def _tiny_cfg():
    cfg = tcfg.Qwen4ExpVisionConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4,
                                    out_hidden_size=32, num_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    return cfg


def test_load_prefixed_state_reads_visual_tensors_from_indexed_shards(tmp_path):
    ref = tmod.Qwen4ExpVisionModel(_tiny_cfg()).eval()
    state = {f"model.visual.{k}": v.to(torch.bfloat16) for k, v in ref.state_dict().items()}
    state["model.layers.0.dummy"] = torch.zeros(2)
    safetensors_torch.save_file(state, str(tmp_path / "model-bf16-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-bf16-00001.safetensors" for k in state}}))
    loaded = load_prefixed_state(str(tmp_path))
    assert set(loaded) == set(ref.state_dict())
    assert "layers.0.dummy" not in loaded


def test_cpu_tower_matches_hf_module(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    ref = tmod.Qwen4ExpVisionModel(cfg).eval()
    state = {f"model.visual.{k}": v for k, v in ref.state_dict().items()}
    safetensors_torch.save_file(state, str(tmp_path / "w.safetensors"))  # no index: header scan
    tower = Qwen4ExpCpuVision(tmod.Qwen4ExpVisionModel(cfg).eval(), cfg.spatial_merge_size, torch.float32)
    tower.model.load_state_dict(load_prefixed_state(str(tmp_path)), strict=True)
    grid = torch.tensor([[1, 4, 4], [1, 2, 6]])
    pv = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16)
    out = tower.encode(pv, grid)
    with torch.no_grad():
        want = ref(pv, grid_thw=grid).pooler_output
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (7, 32)
    torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)
