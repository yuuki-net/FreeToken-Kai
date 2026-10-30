"""The QSA attention tile ladder (``kernel/triton/qsa/attend.py``).

Guards the step-down order a refused tuned tile takes. On a 64 KiB-shared-memory part (Turing)
BLOCK_N=64 never fits; the ladder used to accept (32, 2 stages, 2 warps), and the narrow
decode-tuned tile with four warps prefills 41 % faster there (measured on a Quadro RTX 8000,
Qwen3.8-Flash-Next: 415 vs 584 tok/s). A future "tidy-up" that puts the half tile or the
2-warp variants back in front would turn this red without any kernel change. Pure Python:
the ladder is a generator over tuples, no GPU needed.
"""

from __future__ import annotations

import pytest

from freetoken.kernel.triton.qsa import attend


@pytest.fixture(autouse=True)
def _fresh_ladder_state(monkeypatch):
    monkeypatch.setattr(attend, "_ACCEPTED", {})
    monkeypatch.setattr(attend, "_REFUSED", set())
    monkeypatch.delenv("FT_QSA_PREFILL_PROFILE", raising=False)


def test_tuned_tile_first_then_narrow_four_warp_tile():
    # the prefill profile: BLOCK_N=64, 1 split, 2 warps
    steps = list(attend._profile_ladder(("k",), 64, 1, 2))
    assert steps[:2] == [(64, 1, 2, 2), (64, 1, 2, 1)], "the tuned tile is still tried first"
    assert steps[2] == (16, 1, 4, 2), "a refused tuned tile steps to 16 columns / 4 warps first"
    assert (32, 1, 4, 2) in steps and steps.index((32, 1, 4, 2)) > steps.index((16, 1, 4, 1))
    # the half tile with the tuned two warps is still on offer, but only after every 4-warp step
    assert steps[-2:] == [(32, 1, 2, 2), (32, 1, 2, 1)], "the old 2-warp half tile is the last resort"
    assert all(w == 4 for n, _, w, _ in steps[2:-2]), "the step-downs before the last resort run four warps"


def test_decode_profile_has_no_step_down_below_sixteen():
    steps = list(attend._profile_ladder(("d",), 16, 64, 4))
    assert steps == [(16, 64, 4, 2), (16, 64, 4, 1)]


def test_accepted_profile_short_circuits():
    attend._ACCEPTED[("a",)] = (16, 1, 4, 2)
    assert list(attend._profile_ladder(("a",), 64, 1, 2)) == [(16, 1, 4, 2)]


def test_forced_prefill_profile_is_tried_first_and_only_for_prefill_shapes(monkeypatch):
    monkeypatch.setenv("FT_QSA_PREFILL_PROFILE", "32,1,4,2")
    forced = attend._forced_prefill_profile()
    assert forced == (32, 1, 4, 2)
    steps = list(attend._profile_ladder(("f",), 64, 1, 2, forced))
    assert steps[0] == (32, 1, 4, 2)
    assert steps[1:3] == [(64, 1, 2, 2), (64, 1, 2, 1)], "the tuned tile follows if the forced one is refused"
    # the caller only passes the forced profile for > 512-program shapes; without it nothing changes
    assert list(attend._profile_ladder(("g",), 16, 64, 4)) == [(16, 64, 4, 2), (16, 64, 4, 1)]


def test_forced_profile_unset_is_none(monkeypatch):
    monkeypatch.delenv("FT_QSA_PREFILL_PROFILE", raising=False)
    assert attend._forced_prefill_profile() is None


@pytest.mark.parametrize("value", ["32,1,4", "32,1,4,x", "24,1,4,2", "8,1,4,2", "32,0,4,2", "32,1,3,2", "32,1,4,0"])
def test_malformed_forced_profile_is_ignored_not_raised(monkeypatch, value):
    # read inside the first prefill: raising there would fail a request, not the start
    monkeypatch.setattr(attend, "_BAD_FORCED", set())
    monkeypatch.setenv("FT_QSA_PREFILL_PROFILE", value)
    assert attend._forced_prefill_profile() is None
