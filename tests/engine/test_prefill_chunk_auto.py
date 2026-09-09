"""``--prefill-chunk-budget``: the arithmetic that turns a measured transient into a chunk size.

The measurement itself needs a GPU and a loaded model, so it is stubbed here and only the
decision is exercised — which is the part with the edge cases: the rounding, the floor, the
refusal to ever raise the configured value, and the promise that a failed probe leaves the
configuration alone rather than taking the server down at boot.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from freetoken.engine.engine import Engine


@dataclass
class _Config:
    """Stands in for the frozen EngineConfig: only the two fields the sizer reads."""

    max_extend_tokens: int
    prefill_chunk_budget: float = Engine._PREFILL_TRANSIENT_BUDGET


class _Sizer:
    """Engine's sizer bound to a stubbed measurement."""

    _PREFILL_TRANSIENT_BUDGET = Engine._PREFILL_TRANSIENT_BUDGET
    _PREFILL_PROBE_TOKENS = Engine._PREFILL_PROBE_TOKENS
    _PREFILL_CHUNK_FLOOR = Engine._PREFILL_CHUNK_FLOOR
    _autosize_prefill_chunk = Engine._autosize_prefill_chunk.__wrapped__

    def __init__(self, per_token: float, free: int, max_seq_len: int = 65536, boom=None):
        self._per_token = per_token
        self._free = free
        self.max_seq_len = max_seq_len
        self._boom = boom
        self.probed_with: int | None = None

    def _measure_prefill_transient(self, length: int):
        self.probed_with = length
        if self._boom is not None:
            raise self._boom
        return self._per_token, self._free


GIB = 2**30
KIB = 1024


def test_lowers_the_chunk_to_what_the_free_vram_holds():
    # 124.5 KiB/token was measured on the 2060, with 0.39 GiB free that morning.
    sizer = _Sizer(per_token=124.5 * KIB, free=int(0.39 * GIB))
    config = _Config(max_extend_tokens=8192)
    sizer._autosize_prefill_chunk(config)
    # 0.39 GiB * 0.55 / 127,488 B = 1806 tokens, floored to a multiple of 256.
    # (The live run picked 1536 from the same log line: the "0.39 GiB" there is rounded for
    # display and the real free number was a little lower. The rounding is why this is a
    # multiple of 256 and not an exact quotient.)
    assert config.max_extend_tokens == 1792


def test_keeps_the_configured_value_when_it_already_fits():
    sizer = _Sizer(per_token=8 * KIB, free=4 * GIB)
    config = _Config(max_extend_tokens=8192)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == 8192


def test_never_raises_an_explicitly_smaller_setting():
    """Plenty of room, but the operator asked for 2048. Auto only ever lowers."""
    sizer = _Sizer(per_token=8 * KIB, free=8 * GIB)
    config = _Config(max_extend_tokens=2048)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == 2048


def test_does_not_go_below_the_floor():
    sizer = _Sizer(per_token=4 * 1024 * KIB, free=int(0.1 * GIB))  # absurdly expensive
    config = _Config(max_extend_tokens=8192)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == Engine._PREFILL_CHUNK_FLOOR


def test_a_zero_budget_turns_the_whole_thing_off():
    sizer = _Sizer(per_token=124.5 * KIB, free=int(0.39 * GIB))
    config = _Config(max_extend_tokens=8192, prefill_chunk_budget=0.0)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == 8192
    assert sizer.probed_with is None, "no probe should run when the budget is 0"


@pytest.mark.parametrize(
    "share, expected",
    [
        (0.30, 768),    # a desktop that grabs VRAM in big jumps
        (0.55, 1792),   # the default
        (0.90, 2816),   # a box that serves and nothing else
    ],
)
def test_the_budget_share_is_what_moves_the_answer(share, expected):
    """Same machine, same measurement -- the operator's tolerance picks the chunk."""
    sizer = _Sizer(per_token=124.5 * KIB, free=int(0.39 * GIB))
    config = _Config(max_extend_tokens=8192, prefill_chunk_budget=share)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == expected


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_nonsense_budget_is_ignored_rather_than_obeyed(bad):
    sizer = _Sizer(per_token=124.5 * KIB, free=int(0.39 * GIB))
    config = _Config(max_extend_tokens=8192, prefill_chunk_budget=bad)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == 8192
    assert sizer.probed_with is None


def test_a_failed_probe_does_not_take_the_boot_down():
    sizer = _Sizer(per_token=0, free=0, boom=RuntimeError("CUDA driver error: device not ready"))
    config = _Config(max_extend_tokens=8192)
    sizer._autosize_prefill_chunk(config)  # must not raise
    assert config.max_extend_tokens == 8192


def test_probe_is_small_and_never_larger_than_the_configured_chunk():
    sizer = _Sizer(per_token=8 * KIB, free=4 * GIB)
    sizer._autosize_prefill_chunk(_Config(max_extend_tokens=8192))
    assert sizer.probed_with == Engine._PREFILL_PROBE_TOKENS

    small = _Sizer(per_token=8 * KIB, free=4 * GIB)
    small._autosize_prefill_chunk(_Config(max_extend_tokens=768))
    assert small.probed_with == 768, "a tiny configured chunk must not be probed above itself"


def test_probe_is_clamped_by_the_model_context():
    sizer = _Sizer(per_token=8 * KIB, free=4 * GIB, max_seq_len=256)
    sizer._autosize_prefill_chunk(_Config(max_extend_tokens=8192))
    assert sizer.probed_with == 256


@pytest.mark.parametrize("configured", [0, 512])
def test_nothing_to_do_at_or_below_the_floor(configured):
    sizer = _Sizer(per_token=124.5 * KIB, free=int(0.39 * GIB))
    config = _Config(max_extend_tokens=configured)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == configured
    assert sizer.probed_with is None


class _Live:
    """Engine's runtime re-solve bound to stubbed memory readings."""

    _PREFILL_TRANSIENT_BUDGET = Engine._PREFILL_TRANSIENT_BUDGET
    _PREFILL_CHUNK_FLOOR = Engine._PREFILL_CHUNK_FLOOR
    prefill_chunk_now = Engine.prefill_chunk_now

    def __init__(self, per_token, free, reserved=0, allocated=0, share=None):
        self._prefill_bytes_per_token = per_token
        self.device = "cuda:0"
        self._readings = (free, reserved, allocated)
        if share is not None:
            self._prefill_budget_share = share


@pytest.fixture
def patched_cuda(monkeypatch):
    """Point torch.cuda's three memory queries at whatever _Live is holding."""
    import torch

    holder = {}

    def bind(live):
        holder["live"] = live
        return live

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (holder["live"]._readings[0], 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev: holder["live"]._readings[1])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: holder["live"]._readings[2])
    return bind


def test_runtime_resolve_lowers_the_chunk_when_the_desktop_takes_vram(patched_cuda):
    live = patched_cuda(_Live(per_token=124.5 * KIB, free=int(0.40 * GIB)))
    assert live.prefill_chunk_now(8192) == 1792

    # the same engine, half an hour later, with a browser open
    live._readings = (int(0.20 * GIB), 0, 0)
    assert live.prefill_chunk_now(8192) == 768


def test_runtime_resolve_counts_the_allocator_cache_as_available(patched_cuda):
    """Blocks torch is holding but not using will serve exactly this transient."""
    free = int(0.20 * GIB)
    cached = int(0.30 * GIB)  # reserved but not allocated
    live = patched_cuda(_Live(per_token=124.5 * KIB, free=free, reserved=cached, allocated=0))
    without_cache = _Live(per_token=124.5 * KIB, free=free)
    patched_cuda(without_cache)
    small = without_cache.prefill_chunk_now(8192)
    patched_cuda(live)
    assert live.prefill_chunk_now(8192) > small


def test_runtime_resolve_never_raises_the_ceiling(patched_cuda):
    live = patched_cuda(_Live(per_token=8 * KIB, free=8 * GIB))
    assert live.prefill_chunk_now(2048) == 2048


def test_runtime_resolve_is_a_no_op_without_a_measurement(patched_cuda):
    """--prefill-chunk-budget 0, or a probe that failed: leave the scheduler's number alone."""
    live = patched_cuda(_Live(per_token=0.0, free=int(0.01 * GIB)))
    assert live.prefill_chunk_now(8192) == 8192


def test_runtime_resolve_uses_the_configured_share(patched_cuda):
    tight = patched_cuda(_Live(per_token=124.5 * KIB, free=int(0.40 * GIB), share=0.30))
    assert tight.prefill_chunk_now(8192) == 768
    generous = _Live(per_token=124.5 * KIB, free=int(0.40 * GIB), share=0.90)
    patched_cuda(generous)
    assert generous.prefill_chunk_now(8192) == 2816


# --- the scheduler wiring -----------------------------------------------------------------
# The engine method above is only useful if the scheduler asks it at the right moments: before
# a prefill, and not on the decode turns that make up nearly every scheduling pass.


def _scheduler_with(pending, asked):
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 8192
    s.prefill_manager = SimpleNamespace(
        pending_list=pending,
        schedule_next_batch=lambda budget: asked.append(budget) or None,
    )
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    s.engine = SimpleNamespace(prefill_chunk_now=lambda ceiling: 1792)
    return s


def test_scheduler_re_solves_the_chunk_when_a_prefill_is_waiting():
    asked: list[int] = []
    s = _scheduler_with([object()], asked)
    from freetoken.scheduler.scheduler import Scheduler

    Scheduler._schedule_next_batch(s)
    assert asked == [1792]


def test_scheduler_does_not_query_the_driver_on_a_decode_turn():
    asked: list[int] = []
    s = _scheduler_with([], asked)
    s.engine = None  # any access at all would raise -- that is the assertion
    from freetoken.scheduler.scheduler import Scheduler

    Scheduler._schedule_next_batch(s)
    assert asked == [8192]


# --------------------------------------------------------------------------------------
# --pp-size: the chunk sizes a message between ranks, so the ranks have to agree on it
# --------------------------------------------------------------------------------------


@dataclass
class _PPConfig(_Config):
    is_pp: bool = True


def test_pipeline_ranks_solve_the_chunk_from_the_tightest_numbers(monkeypatch):
    """Each rank measures its own half of the model against its own free VRAM. Solving from
    those independently gave rank 0 3328 tokens and rank 1 3072, and the first prefill died in
    gloo with "Received data size doesn't match expected size" (68157440 vs 62914560 bytes)."""
    import torch

    from freetoken.engine import engine as engine_mod

    seen = {}

    def _all_reduce(tensor, op=None, group=None):
        seen["op"] = op
        # the other rank: more transient per token, less VRAM free
        other = torch.tensor([0.0, 300.0 * KIB, -float(int(1.0 * GIB))], dtype=torch.float64)
        torch.maximum(tensor, other, out=tensor)

    monkeypatch.setattr(engine_mod.torch.distributed, "all_reduce", _all_reduce)

    sizer = _Sizer(per_token=255.0 * KIB, free=int(1.48 * GIB))
    sizer.tp_cpu_group = None
    config = _PPConfig(max_extend_tokens=4096)
    sizer._autosize_prefill_chunk(config)

    assert seen["op"] is torch.distributed.ReduceOp.MAX
    # solved from the other rank's numbers: 1.0 GiB * 0.55 / 307,200 B = 1922 -> 1792
    assert config.max_extend_tokens == 1792


def test_a_failed_probe_on_one_rank_leaves_every_rank_configured(monkeypatch):
    """A rank whose probe failed cannot chunk; the others must not chunk either."""
    import torch

    from freetoken.engine import engine as engine_mod

    def _all_reduce(tensor, op=None, group=None):
        other = torch.tensor([1.0, 0.0, -float(int(8.0 * GIB))], dtype=torch.float64)  # failed
        torch.maximum(tensor, other, out=tensor)

    monkeypatch.setattr(engine_mod.torch.distributed, "all_reduce", _all_reduce)

    sizer = _Sizer(per_token=255.0 * KIB, free=int(1.48 * GIB))
    sizer.tp_cpu_group = None
    config = _PPConfig(max_extend_tokens=4096)
    sizer._autosize_prefill_chunk(config)
    assert config.max_extend_tokens == 4096


def test_the_runtime_resolve_is_frozen_under_pp(monkeypatch):
    """Re-solving per prefill cannot be agreed without a collective on the hot path, and each
    rank reaches it on its own schedule, so under --pp-size the boot value stands."""
    import torch

    # tight enough that a re-solve would drop to the floor, which is what must NOT happen here
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(0.02 * GIB), 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)

    class _PPLive(_Live):
        config = _PPConfig(max_extend_tokens=4096)

    single = _Live(per_token=255.0 * KIB, free=0, reserved=0, allocated=0)
    assert single.prefill_chunk_now(3072) == Engine._PREFILL_CHUNK_FLOOR  # one GPU: re-solves

    live = _PPLive(per_token=255.0 * KIB, free=0, reserved=0, allocated=0)
    assert live.prefill_chunk_now(3072) == 3072  # two ranks: frozen at what they agreed on
