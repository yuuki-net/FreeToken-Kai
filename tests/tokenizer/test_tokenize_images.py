"""Image requests go through the checkpoint's HF processor: its template places the
placeholder, its __call__ expands it and yields the vision tensors."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


class FakeTokenizer:
    name_or_path = "fake/model"

    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError("text path must not be used for an image request")

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        return torch.tensor([[1, 2]], dtype=torch.long)


class FakeProcessor:
    image_processor = object()
    image_token = "<image>"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "mm prompt <image>"

    def __call__(self, *, text, images, return_tensors, add_special_tokens):
        self.calls.append({"text": text, "images": images, "add_special_tokens": add_special_tokens})
        n = len(images[0])
        return {
            "input_ids": torch.tensor([[2] + [7] * (3 * n) + [5]], dtype=torch.long),
            "pixel_values": torch.rand(n, 6, 12, dtype=torch.float32),
            "image_position_ids": torch.full((n, 6, 2), -1, dtype=torch.int64),
        }


def _manager(proc):
    m = TokenizeManager(FakeTokenizer())
    m._processor, m._processor_tried = proc, True
    return m


def _msg(n_images: int, images=None):
    parts = [{"type": "image"}] * n_images + [{"type": "text", "text": "what?"}]
    return TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": parts}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": False},
        images=[_png()] * n_images if images is None else images,
    )


def test_image_request_uses_processor_template_and_returns_mm_inputs():
    proc = FakeProcessor()
    [(ids, mm)] = _manager(proc).tokenize_with_images([_msg(2)])
    assert ids.dtype == torch.int32 and ids.tolist() == [2] + [7] * 6 + [5]
    assert mm["pixel_values"].dtype == torch.float16 and tuple(mm["pixel_values"].shape) == (2, 6, 12)
    assert mm["image_position_ids"].dtype == torch.int64
    call = proc.calls[0]
    assert call["text"] == ["mm prompt <image>"]
    assert [im.size for im in call["images"][0]] == [(4, 4), (4, 4)]
    assert call["add_special_tokens"] is False  # the template rendered bos already
    assert proc.template_kwargs["enable_thinking"] is False
    assert proc.template_kwargs["add_generation_prompt"] is True


def test_tokenize_returns_only_ids_and_text_requests_are_unchanged():
    proc = FakeProcessor()
    m = _manager(proc)
    assert m.tokenize([_msg(1)])[0].tolist() == [2, 7, 7, 7, 5]
    text_only = TokenizeMsg(uid=2, text="plain", sampling_params=SamplingParams())
    [(ids, mm)] = m.tokenize_with_images([text_only])
    assert ids.tolist() == [1, 2] and mm is None


def test_image_count_must_match_image_parts():
    with pytest.raises(ValueError, match="image content part"):
        _manager(FakeProcessor()).tokenize_with_images([_msg(2, images=[_png()])])


def test_checkpoint_without_processor_rejects_images_per_request():
    with pytest.raises(ValueError, match="does not accept image input"):
        _manager(None).tokenize_with_images([_msg(1)])


def test_render_prompt_for_images_uses_processor_template():
    proc = FakeProcessor()
    assert _manager(proc).render_prompt(_msg(1)) == "mm prompt <image>"


def test_undecodable_image_is_a_request_error():
    with pytest.raises(ValueError, match="could not decode image"):
        _manager(FakeProcessor()).tokenize_with_images([_msg(1, images=[b"not an image"])])
