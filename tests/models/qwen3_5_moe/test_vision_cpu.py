"""The CPU vision tower also serves the Qwen3.5-MoE family (Qwen3.6-35B-A3B, Ornith-1.5):
``model.visual.*`` from the shards into transformers' ``Qwen3_5MoeVisionModel``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

tcfg = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
tmod = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
safetensors_torch = pytest.importorskip("safetensors.torch")

from freetoken.models.qwen4_exp import vision_cpu  # noqa: E402
from freetoken.models.qwen4_exp.vision_cpu import (  # noqa: E402
    HOST_VISION_MODEL_TYPES,
    Qwen4ExpCpuVision,
    load_prefixed_state,
    vision_model_class,
)


def _tiny_cfg():
    cfg = tcfg.Qwen3_5MoeVisionConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4,
                                      out_hidden_size=32, num_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    return cfg


def test_model_types_and_classes():
    assert set(HOST_VISION_MODEL_TYPES) == {"qwen4_exp", "qwen3_5_moe"}
    assert vision_model_class("qwen3_5_moe") is tmod.Qwen3_5MoeVisionModel
    with pytest.raises(ValueError):
        vision_model_class("gpt_oss")


def test_cpu_tower_matches_hf_module(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    ref = tmod.Qwen3_5MoeVisionModel(cfg).eval()
    state = {f"model.visual.{k}": v.to(torch.bfloat16) for k, v in ref.state_dict().items()}
    state["model.language_model.layers.0.dummy"] = torch.zeros(2)
    safetensors_torch.save_file(state, str(tmp_path / "model-00003-of-00003.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00003-of-00003.safetensors" for k in state}}))
    loaded = load_prefixed_state(str(tmp_path))
    assert set(loaded) == set(ref.state_dict())
    tower = Qwen4ExpCpuVision(tmod.Qwen3_5MoeVisionModel(cfg).eval(), cfg.spatial_merge_size, torch.float32)
    tower.model.load_state_dict({k: v.float() for k, v in loaded.items()}, strict=True)
    ref.load_state_dict({k: v.float() for k, v in loaded.items()}, strict=True)
    grid = torch.tensor([[1, 4, 4], [1, 2, 6]])
    pv = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16)
    out = tower.encode(pv, grid)
    with torch.no_grad():
        want = ref(pv, grid_thw=grid).pooler_output
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (7, 32)
    torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)


def test_from_checkpoint_picks_the_class_by_model_type_and_refuses_deepstack(tmp_path, monkeypatch):
    cfg = _tiny_cfg()
    ref = tmod.Qwen3_5MoeVisionModel(cfg).eval()
    safetensors_torch.save_file(
        {f"model.visual.{k}": v for k, v in ref.state_dict().items()}, str(tmp_path / "w.safetensors"))
    hf = SimpleNamespace(model_type="qwen3_5_moe", vision_config=cfg)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *_a, **_k: hf)
    tower = Qwen4ExpCpuVision.from_checkpoint(str(tmp_path))
    assert isinstance(tower.model, tmod.Qwen3_5MoeVisionModel) and tower.spatial_merge_size == 2
    cfg.deepstack_visual_indexes = [5, 11, 17]
    with pytest.raises(ValueError, match="DeepStack"):
        Qwen4ExpCpuVision.from_checkpoint(str(tmp_path))
