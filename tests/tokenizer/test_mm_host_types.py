"""``build_host_image_encoder`` builds a CPU tower for every model_type the tower supports."""

from __future__ import annotations

import json

import pytest

from freetoken.tokenizer import mm_host


class _FakeTower:
    spatial_merge_size = 2


@pytest.mark.parametrize("model_type,expected", [
    ("qwen4_exp", True), ("qwen3_5_moe", True), ("gpt_oss", False), ("gemma4", False),
])
def test_host_encoder_by_model_type(tmp_path, monkeypatch, model_type, expected):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.vision_cpu.Qwen4ExpCpuVision.from_checkpoint",
        classmethod(lambda cls, path: _FakeTower()),
    )
    enc = mm_host.build_host_image_encoder(str(tmp_path))
    assert (enc is not None) is expected
