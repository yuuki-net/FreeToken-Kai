"""--mm-encoder-dtype: the Qwen VL vision tower computes in its own dtype. In bfloat16 (7 fraction bits)
it loses about 9% of its output against float32 -- transformers' code and upstream's alike, measured on
an RTX 2060 with Ornith-1.5 -- so "auto" builds it in float32 when the model runs bfloat16."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.mm.config import MultimodalConfig


@pytest.fixture(autouse=True)
def _single_rank():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.mark.parametrize("engine, flag, want", [
    (torch.bfloat16, "auto", "float32"),
    (torch.float16, "auto", "float16"),
    (torch.float32, "auto", "float32"),
    (torch.bfloat16, "bfloat16", "bfloat16"),
    (torch.float16, "float32", "float32"),
])
def test_auto_takes_float32_under_bfloat16_only(engine, flag, want):
    assert MultimodalConfig(encoder_dtype=flag).resolve_encoder_dtype(engine) == want


def _vc(**over):
    from freetoken.models.qwen3_vl import VisionConfig

    fields = dict(hidden_size=64, depth=2, num_heads=4, intermediate_size=128, patch_size=16, temporal_patch_size=2,
                  spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=32, in_channels=3)
    fields.update(over)
    return VisionConfig(**fields)


def _tower(vc, model_dtype):
    from freetoken.models.qwen3_vl import Qwen3VLVisionModel
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(model_dtype):  # as the engine builds the model
        return Qwen3VLVisionModel(vc)


def test_the_tower_is_built_in_its_own_dtype_inside_a_bf16_model():
    tower = _tower(_vc(dtype="float32"), torch.bfloat16)
    assert {t.dtype for t in tower.state_dict().values()} == {torch.float32}
    assert torch.get_default_dtype() == torch.float32  # the context is restored


def test_without_a_dtype_the_tower_follows_the_model():
    tower = _tower(_vc(), torch.bfloat16)
    assert {t.dtype for t in tower.state_dict().values()} == {torch.bfloat16}


def test_the_engine_config_hands_the_resolved_dtype_to_the_tower(monkeypatch):
    pytest.importorskip("freetoken.engine.config")
    from dataclasses import replace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig
    from freetoken.models.register import EncoderSpec
    from tests.distributed.test_pipeline import _parsed_qwen4, _qwen4_hf_config

    spec = SimpleNamespace(module="m", parse_config="p", encoders=(EncoderSpec("vision", "vision_config", ("image",)),))
    hf = _qwen4_hf_config()
    hf.vision_config = SimpleNamespace(depth=2)
    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: spec)
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: (
        lambda h: replace(_parsed_qwen4(), vision_config=_vc()) if getattr(h, "vision_config", None) else _parsed_qwen4()))
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path, overrides: hf)
    monkeypatch.setattr(cfg_mod, "checkpoint_quant_config", lambda *a, **k: None)

    def make(dtype, **mm):
        return EngineConfig(model_path="x", tp_info=DistributedInfo(0, 1), dtype=dtype, mm=MultimodalConfig(**mm))

    assert make(torch.bfloat16).model_config.vision_config.dtype == "float32"
    assert make(torch.float16).model_config.vision_config.dtype == "float16"
    assert make(torch.bfloat16, encoder_dtype="bfloat16").model_config.vision_config.dtype == "bfloat16"
    assert make(torch.bfloat16, encoder_weights="cpu").model_config.vision_config is None  # no GPU tower at all
