"""Scheduler side of M-RoPE: host-encoded soft tokens are accepted as-is, the prompt gets a
rope table + delta, and rope positions/tables are built per batch."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import MMRope, SamplingParams
from freetoken.message import UserMsg
from freetoken.scheduler.scheduler import Scheduler, _make_rope_positions, _prefill_rope_table

IMG = 7


def _sched(mrope=True):
    s = Scheduler.__new__(Scheduler)
    s.engine = SimpleNamespace(model=object())
    s.device = torch.device("cpu")
    rotary = SimpleNamespace(rotary_dim=64, base=1e7, mrope_section=(11, 11, 10) if mrope else None,
                             mrope_interleaved=True)
    s.config = SimpleNamespace(model_config=SimpleNamespace(image_token_id=IMG, rotary_config=rotary))
    return s


def _msg(n_img_tokens=6):
    ids = [1, 1] + [IMG] * n_img_tokens + [2]
    return UserMsg(
        uid=1, input_ids=torch.tensor(ids, dtype=torch.int32), sampling_params=SamplingParams(),
        mm_inputs={"mm_embeds": torch.ones(6, 8, dtype=torch.bfloat16),
                   "image_grid_thw": torch.tensor([[1, 4, 6]]), "merge_size": torch.tensor(2)},
    )


def test_host_encoded_images_attach_embeds_and_mrope():
    msg = _msg()
    assert _sched()._encode_multimodal(msg) is None
    assert tuple(msg.mm_embeds.shape) == (6, 8) and msg.mm_inputs is None
    assert isinstance(msg.mm_rope, MMRope)
    assert msg.mm_rope.delta == (2 + 3 + 1) - 9  # max position 5 (2 + max(2,3)) -> 6 - len 9
    assert tuple(msg.mm_rope.cos_sin.shape) == (9, 64)


def test_models_without_mrope_get_no_rope_table():
    msg = _msg()
    assert _sched(mrope=False)._encode_multimodal(msg) is None
    assert msg.mm_rope is None


def test_slot_mismatch_still_rejected_for_host_encoded():
    msg = _msg(n_img_tokens=5)
    err = _sched()._encode_multimodal(msg)
    assert err and "placeholder count (5)" in err


def _req(cached, device_len, rope=None):
    return SimpleNamespace(cached_len=cached, device_len=device_len, extend_len=device_len - cached, mm_rope=rope)


def test_rope_positions_apply_each_requests_delta():
    batch = SimpleNamespace(padded_reqs=[_req(10, 11, MMRope(delta=-3)), _req(4, 5), _req(0, 3, MMRope(delta=-1))])
    out = _make_rope_positions(batch, torch.device("cpu"))
    assert out.tolist() == [7, 4, -1, 0, 1]


def test_rope_positions_are_none_when_no_request_has_a_delta():
    batch = SimpleNamespace(padded_reqs=[_req(10, 11), _req(4, 5)])
    assert _make_rope_positions(batch, torch.device("cpu")) is None


def test_prefill_table_only_for_a_solo_uncached_image_prompt():
    table = torch.zeros(9, 64)
    solo = SimpleNamespace(is_prefill=True, padded_reqs=[_req(0, 9, MMRope(delta=-3, cos_sin=table))])
    assert _prefill_rope_table(solo) is table
    decode = SimpleNamespace(is_prefill=False, padded_reqs=solo.padded_reqs)
    assert _prefill_rope_table(decode) is None
    pair = SimpleNamespace(is_prefill=True, padded_reqs=solo.padded_reqs + [_req(0, 2)])
    assert _prefill_rope_table(pair) is None
