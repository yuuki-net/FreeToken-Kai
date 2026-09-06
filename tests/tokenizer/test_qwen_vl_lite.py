"""The torchvision-free Qwen-VL processor: template via the tokenizer, placeholders repeated
per image to the grid's token count, tensors from the PIL image processor."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch

from freetoken.tokenizer.qwen_vl_lite import IMAGE_TOKEN, QwenVLLiteProcessor
from freetoken.tokenizer.tokenize import TokenizeManager


class FakeTokenizer:
    """ids: 7 for <|image_pad|>, 1 for any other whitespace-separated word."""

    def apply_chat_template(self, messages, **kwargs):
        return f"user: look {IMAGE_TOKEN} and {IMAGE_TOKEN} ok"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert add_special_tokens is False
        toks = re.findall(r"<\|image_pad\|>|\S+", text)
        return {"input_ids": torch.tensor([[7 if t == IMAGE_TOKEN else 1 for t in toks]])}


class FakeImageProcessor:
    merge_size = 2
    size = SimpleNamespace(longest_edge=16777216, shortest_edge=65536)  # SizeDict-like

    def __init__(self):
        self.size_seen = None

    def __call__(self, *, images, return_tensors, size=None):
        self.size_seen = size
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]][: len(images)])
        return {"pixel_values": torch.zeros(int(grid.prod(-1).sum()), 1536), "image_grid_thw": grid}


def test_placeholders_are_repeated_to_each_images_token_count():
    proc = QwenVLLiteProcessor(FakeTokenizer(), FakeImageProcessor())
    out = proc(text=[proc.apply_chat_template([])], images=[["a", "b"]], size={"longest_edge": 4096})
    ids = out["input_ids"][0].tolist()
    # "user: look" (2) + 4 pads + "and" + 2 pads + "ok"
    assert ids == [1, 1, 7, 7, 7, 7, 1, 7, 7, 1]
    assert out["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert tuple(out["pixel_values"].shape) == (24, 1536)
    assert proc.image_processor.size_seen == {"longest_edge": 4096}


def test_placeholder_image_count_mismatch_is_an_error():
    proc = QwenVLLiteProcessor(FakeTokenizer(), FakeImageProcessor())
    with pytest.raises(ValueError, match="placeholder"):
        proc(text=[proc.apply_chat_template([])], images=[["only-one"]])


def test_from_pretrained_ignores_non_qwen_configs(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "Gemma4ImageProcessor"}))
    assert QwenVLLiteProcessor.from_pretrained(str(tmp_path), FakeTokenizer()) is None
    assert QwenVLLiteProcessor.from_pretrained(str(tmp_path / "missing"), FakeTokenizer()) is None


def test_size_cap_reads_sizedict_like_objects(monkeypatch):
    monkeypatch.setenv("FT_IMAGE_MAX_PIXELS", "262144")
    manager = TokenizeManager(FakeTokenizer())
    proc = QwenVLLiteProcessor(FakeTokenizer(), FakeImageProcessor())
    assert manager._image_size_kwargs(proc) == {"size": {"shortest_edge": 65536, "longest_edge": 262144}}
