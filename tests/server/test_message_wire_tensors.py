"""The wire codec carries N-D tensors (image tensors ride UserMsg.mm_inputs) and stays
compatible with the 1-D form older peers send."""

from __future__ import annotations

import msgpack
import torch

from freetoken.core import SamplingParams
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, TokenizeMsg, UserMsg
from freetoken.message.utils import deserialize_type, serialize_type


def _wire(obj):
    return msgpack.unpackb(msgpack.packb(obj, use_bin_type=True), raw=False)


def test_tensor_roundtrip_any_rank_and_dtype():
    for t in (
        torch.arange(24, dtype=torch.float16).reshape(2, 3, 4),
        torch.arange(12, dtype=torch.int64).reshape(2, 3, 2) - 1,
        torch.tensor([[1.5, -2.0], [0.25, 8.0]], dtype=torch.bfloat16),
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.zeros(0, dtype=torch.float32),
    ):
        out = deserialize_type({}, _wire(serialize_type(t)))
        assert out.dtype == t.dtype and out.shape == t.shape
        assert torch.equal(out, t)


def test_legacy_1d_message_without_shape_decodes():
    legacy = {"__type__": "Tensor", "buffer": torch.tensor([7, 8], dtype=torch.int32).numpy().tobytes(),
              "dtype": "torch.int32"}
    out = deserialize_type({}, _wire(legacy))
    assert out.tolist() == [7, 8]


def test_user_msg_mm_inputs_roundtrip():
    msg = UserMsg(
        uid=3,
        input_ids=torch.tensor([1, 9, 9, 2], dtype=torch.int32),
        sampling_params=SamplingParams(),
        mm_inputs={
            "pixel_values": torch.rand(1, 5, 12).to(torch.float16),
            "image_position_ids": torch.tensor([[[0, 0], [1, 0], [0, 1], [-1, -1], [-1, -1]]]),
        },
    )
    out = BaseBackendMsg.decoder(_wire(msg.encoder()))
    assert isinstance(out, UserMsg) and out.mm_embeds is None
    assert torch.equal(out.mm_inputs["pixel_values"], msg.mm_inputs["pixel_values"])
    assert torch.equal(out.mm_inputs["image_position_ids"], msg.mm_inputs["image_position_ids"])
    assert out.input_ids.tolist() == [1, 9, 9, 2]


def test_tokenize_msg_images_roundtrip():
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}],
        sampling_params=SamplingParams(),
        images=[b"\x89PNG...", b"\xff\xd8\xff..."],
    )
    out = BaseTokenizerMsg.decoder(_wire(BaseTokenizerMsg.encoder(msg)))
    assert out.images == msg.images
    assert out.text == msg.text
