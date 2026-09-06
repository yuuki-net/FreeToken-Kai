"""Checkpoints whose vision tower runs on the CPU: the tokenizer worker hands the scheduler
encoded soft tokens (plus the grid), and token counting never encodes."""

from __future__ import annotations

import io

import torch
from PIL import Image

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    return buf.getvalue()


class FakeTokenizer:
    name_or_path = "fake/qwen"

    def encode(self, *a, **k):
        return torch.tensor([[1]])


class FakeQwenProcessor:
    image_token = "<|image_pad|>"

    class image_processor:  # noqa: N801 - mimics the HF attribute
        size = {"shortest_edge": 3136, "longest_edge": 16777216}

    def __init__(self):
        self.size_seen = None

    def apply_chat_template(self, messages, **kw):
        return "<|vision_start|><|image_pad|><|vision_end|>?"

    def __call__(self, *, text, images, return_tensors, add_special_tokens, size=None):
        self.size_seen = size
        return {
            "input_ids": torch.tensor([[3, 7, 7, 7, 7, 4, 9]]),
            "pixel_values": torch.rand(16, 1536),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, mm):
        self.calls += 1
        assert set(mm) == {"pixel_values", "image_grid_thw"} and mm["pixel_values"].dtype == torch.float16
        return {"mm_embeds": torch.zeros(4, 8, dtype=torch.bfloat16), "image_grid_thw": mm["image_grid_thw"],
                "merge_size": torch.tensor(2)}


def _manager(proc, enc):
    m = TokenizeManager(FakeTokenizer())
    m._processor, m._processor_tried = proc, True
    m._host_encoder, m._host_encoder_tried = enc, True
    return m


def _msg():
    return TokenizeMsg(uid=1, text=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}],
                       sampling_params=SamplingParams(), images=[_png()])


def test_host_encoder_output_replaces_raw_tensors_and_size_is_capped(monkeypatch):
    monkeypatch.setenv("FT_IMAGE_MAX_PIXELS", "65536")
    proc, enc = FakeQwenProcessor(), FakeEncoder()
    [(ids, mm)] = _manager(proc, enc).tokenize_with_images([_msg()])
    assert ids.tolist() == [3, 7, 7, 7, 7, 4, 9]
    assert set(mm) == {"mm_embeds", "image_grid_thw", "merge_size"}
    assert enc.calls == 1
    assert proc.size_seen == {"shortest_edge": 3136, "longest_edge": 65536}


def test_counting_expands_placeholders_without_encoding():
    proc, enc = FakeQwenProcessor(), FakeEncoder()
    m = _manager(proc, enc)
    assert m.tokenize([_msg()])[0].tolist() == [3, 7, 7, 7, 7, 4, 9]
    assert enc.calls == 0


def test_without_host_encoder_raw_tensors_pass_through():
    proc = FakeQwenProcessor()
    [(_, mm)] = _manager(proc, None).tokenize_with_images([_msg()])
    assert set(mm) == {"pixel_values", "image_grid_thw"}
