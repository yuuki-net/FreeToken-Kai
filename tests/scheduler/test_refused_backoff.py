"""A prefill refused with nothing running must not spin the scheduler loop (guides/25 §8.7)."""

from types import SimpleNamespace

import pytest

from freetoken.scheduler import scheduler as sched_mod
from freetoken.scheduler.scheduler import Scheduler


def _stuck(pending=True, running=False, disk=None):
    s = Scheduler.__new__(Scheduler)
    s.prefill_manager = SimpleNamespace(runnable=pending)
    s.decode_manager = SimpleNamespace(runnable=running)
    s.prefix_disk = disk
    return s


@pytest.fixture
def sleeps(monkeypatch):
    got = []
    monkeypatch.setattr(sched_mod.time, "sleep", got.append)
    return got


def test_a_refused_head_with_nothing_running_backs_off_to_a_cap(sleeps):
    s = _stuck()
    for _ in range(10):
        Scheduler._wait_nothing_scheduled(s)
    assert sleeps[:4] == pytest.approx([0.001, 0.002, 0.004, 0.008])
    assert sleeps == sorted(sleeps)
    assert max(sleeps) == sched_mod._REFUSED_BACKOFF_MAX_S
    assert sleeps[-1] == sched_mod._REFUSED_BACKOFF_MAX_S


def test_no_sleep_while_something_runs_or_nothing_waits(sleeps):
    Scheduler._wait_nothing_scheduled(_stuck(running=True))
    Scheduler._wait_nothing_scheduled(_stuck(pending=False))
    assert sleeps == []


def test_a_disk_load_is_waited_on_instead(sleeps):
    waited = []
    disk = SimpleNamespace(loading=True, wait_for_load=waited.append)
    Scheduler._wait_nothing_scheduled(_stuck(disk=disk))
    assert waited == [0.01] and sleeps == []


def test_a_scheduled_batch_resets_the_backoff(sleeps):
    s = _stuck()
    for _ in range(6):
        Scheduler._wait_nothing_scheduled(s)
    batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    s.prefill_budget = 99
    s.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: batch, pending_list=[object()],
                                        runnable=True)
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None, runnable=False)
    s.engine = SimpleNamespace(prefill_chunk_now=lambda ceiling: ceiling)
    s._prepare_batch = lambda b: "forward-input"
    s.send_result = lambda msgs: None
    assert Scheduler._schedule_next_batch(s) == "forward-input"
    sleeps.clear()
    Scheduler._wait_nothing_scheduled(s)
    assert sleeps == [0.001]
