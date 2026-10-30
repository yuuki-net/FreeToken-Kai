"""``FT_KERNEL_PROFILE_PREFILL`` (``utils/kernel_profile.py``).

Guards the sampling decision, which is easy to get subtly wrong: decode steps must keep
being sampled by default, prefill chunks only when asked, ``only`` must stop the few samples
being spent on the short requests that precede the first long prompt, and the row floor must
apply to prefill chunks alone. No profiler is run: ``wants()`` is a pure decision.
"""

from __future__ import annotations

import pytest

from freetoken.utils.kernel_profile import KernelProfiler


def _kp(monkeypatch, **env):
    for k in ("FT_KERNEL_PROFILE", "FT_KERNEL_PROFILE_COUNT", "FT_KERNEL_PROFILE_PREFILL", "FT_KERNEL_PROFILE_MIN_ROWS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return KernelProfiler.from_env(lambda *_: None)


def test_off_by_default(monkeypatch):
    assert _kp(monkeypatch) is None


def test_default_samples_decode_and_verify_but_not_prefill(monkeypatch):
    kp = _kp(monkeypatch, FT_KERNEL_PROFILE=1)
    assert kp.wants(is_prefill=False, spec_verify=False, rows=1)
    assert kp.wants(is_prefill=True, spec_verify=True, rows=4)
    assert not kp.wants(is_prefill=True, spec_verify=False, rows=4096)


def test_prefill_one_samples_both_with_row_floor(monkeypatch):
    kp = _kp(monkeypatch, FT_KERNEL_PROFILE=1, FT_KERNEL_PROFILE_PREFILL=1, FT_KERNEL_PROFILE_MIN_ROWS=2048)
    assert kp.wants(is_prefill=False, spec_verify=False, rows=1)
    assert kp.wants(is_prefill=True, spec_verify=False, rows=4096)
    assert not kp.wants(is_prefill=True, spec_verify=False, rows=55), "token-count probes stay unsampled"


def test_prefill_only_skips_decode_steps(monkeypatch):
    kp = _kp(monkeypatch, FT_KERNEL_PROFILE=1, FT_KERNEL_PROFILE_PREFILL="only")
    assert not kp.wants(is_prefill=False, spec_verify=False, rows=1)
    assert not kp.wants(is_prefill=True, spec_verify=True, rows=4)
    assert kp.wants(is_prefill=True, spec_verify=False, rows=2911)


def test_count_and_every_schedule_unchanged(monkeypatch):
    kp = _kp(monkeypatch, FT_KERNEL_PROFILE=2, FT_KERNEL_PROFILE_COUNT=1)
    seq = []
    for _ in range(4):
        d = kp.due()
        seq.append(d)
        if d:
            kp.samples += 1  # profile() does this once the sample is taken
    assert seq == [False, True, False, False]
