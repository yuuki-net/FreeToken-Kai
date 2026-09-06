"""The scheduler turns a request's processor tensors into soft-token embeddings on
admission, and turns every failure into that request's error instead of raising."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import UserMsg
from freetoken.scheduler.scheduler import Scheduler

IMG = 7


class FakeVisionModel:
    vision_tower = object()

    def __init__(self, rows: int, fail: bool = False) -> None:
        self.rows, self.fail, self.seen = rows, fail, None

    def encode_images(self, pixel_values, image_position_ids):
        if self.fail:
            raise RuntimeError("tower exploded")
        self.seen = (pixel_values.shape, image_position_ids.shape)
        return torch.ones(self.rows, 4)


def _sched(model):
    s = Scheduler.__new__(Scheduler)
    s.engine = SimpleNamespace(model=model)
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(model_config=SimpleNamespace(image_token_id=IMG))
    return s


def _msg(n_slots: int):
    return UserMsg(
        uid=1,
        input_ids=torch.tensor([2] + [IMG] * n_slots + [9], dtype=torch.int32),
        sampling_params=SamplingParams(),
        mm_inputs={
            "pixel_values": torch.zeros(1, 9, 12, dtype=torch.float16),
            "image_position_ids": torch.zeros(1, 9, 2, dtype=torch.int64),
        },
    )


def test_matching_slots_attach_embeddings_and_drop_raw_inputs():
    model = FakeVisionModel(rows=3)
    msg = _msg(3)
    assert _sched(model)._encode_multimodal(msg) is None
    assert msg.mm_inputs is None
    assert tuple(msg.mm_embeds.shape) == (3, 4)
    assert model.seen == ((1, 9, 12), (1, 9, 2))


def test_slot_mismatch_is_a_request_error():
    msg = _msg(2)
    err = _sched(FakeVisionModel(rows=3))._encode_multimodal(msg)
    assert err and "placeholder count (2)" in err and msg.mm_embeds is None


def test_text_only_model_is_a_request_error():
    class TextOnly:
        pass

    err = _sched(TextOnly())._encode_multimodal(_msg(1))
    assert err and "not serving image input" in err


def test_vision_off_checkpoint_is_a_request_error():
    class VisionOff:  # encode_images exists, tower was not built (FREETOKEN_LOAD_VISION unset)
        def encode_images(self, *a):
            raise AssertionError("must not be called")

    err = _sched(VisionOff())._encode_multimodal(_msg(1))
    assert err and "FREETOKEN_LOAD_VISION" in err


def test_tower_failure_is_a_request_error():
    err = _sched(FakeVisionModel(rows=1, fail=True))._encode_multimodal(_msg(1))
    assert err and "tower exploded" in err
