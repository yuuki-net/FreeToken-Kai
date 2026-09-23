"""What a refused cudaHostRegister reports, and what it leaves behind for the next start.

Issue #2: the refusal named the size of the bank that happened to ask for the byte over the
cap ("failed for 0.0 GiB") and then told the reporter to pass a flag they had already passed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from freetoken.engine import engine  # noqa: E402
from freetoken.moe import host_banks, pin_probe  # noqa: E402

GiB = pin_probe.GiB
MiB = 1 << 20


def test_a_refusal_reports_what_was_locked_not_the_bank_that_asked():
    exc = host_banks.PinFailed(
        "cudaHostRegister refused a 48 MiB bank after 1.00 GiB of this process was already page-locked",
        bank_bytes=48 * MiB, registered_bytes=1 * GiB,
    )
    assert exc.registered_bytes == 1 * GiB
    assert exc.bank_bytes == 48 * MiB
    assert "after 1.00 GiB" in str(exc)


def test_a_bank_that_pins_counts_towards_the_process_total(monkeypatch):
    """The cap is a process-wide quota, so the running total is what a refusal can report."""
    before = host_banks.registered_bytes()
    monkeypatch.setattr(host_banks, "_pinned_total", before)
    host_banks._count_pinned(256 * MiB)
    assert host_banks.registered_bytes() == before + 256 * MiB


def test_the_hint_does_not_ask_for_a_flag_that_is_already_set(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "1.0")
    unset = engine._pin_hint(0)
    already = engine._pin_hint(0, cpu_layers_set=True)
    assert "pass --moe-cpu-layers auto" in unset
    assert "pass --moe-cpu-layers auto" not in already
    assert "already set" in already and "FREETOKEN_PIN_BUDGET_GB" in already


def test_the_hint_for_an_uncapped_host_is_about_the_host(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    monkeypatch.setattr(pin_probe, "is_pin_capped", lambda release=None: False)
    for kwargs in ({}, {"cpu_layers_set": True}):
        assert "more page-locked host RAM than this host has" in engine._pin_hint(0, **kwargs)


def test_a_refusal_records_the_cap_it_measured(monkeypatch, tmp_path):
    """The bytes locked before the refusal ARE this host's cap, so the next start plans for it
    instead of dying in the same place."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       24000000 kB\nMemAvailable:   20000000 kB\n")
    monkeypatch.setattr(pin_probe, "is_pin_capped", lambda release=None: True)
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)

    assert pin_probe.remembered(str(proc)) is None
    pin_probe.remember(1 * GiB, how="refused mid-load", proc=str(proc))
    assert pin_probe.budget(proc=str(proc)) == 1 * GiB
    assert pin_probe.source(str(proc)) == "measured (refused mid-load)"

# ------------------------------------------------- a mapping registered only in part


class _Banks:
    def __init__(self, registered_layers, hot_blocks=144, registered_blocks=144, layers=24):
        self.registered_layers = set(registered_layers)
        self.hot_blocks = hot_blocks
        self.registered_blocks = registered_blocks
        self.layers = list(range(layers))
        self.registered_bytes = registered_blocks << 20

    @property
    def fully_registered(self) -> bool:  # mapped_bank's own rule
        return self.hot_blocks > 0 and self.registered_blocks == self.hot_blocks


class _Tier:
    def __init__(self, banks):
        self.banks = banks


def test_the_layers_the_budget_missed_decode_on_the_cpu():
    """Registration walks the file layer by layer and stops when the budget is spent, so the
    layers outside the covered set have no device address. Reading "some bytes registered"
    instead labelled every layer PINNED and the copy plan then asked for the device address of a
    block that was never registered -- issue #2's crash with an explicit --moe-bank-ram size."""
    part = _Tier(_Banks(range(5), registered_blocks=30))
    assert engine._mapped_cpu_layers(part, 24) == frozenset(range(5, 24))


def test_every_layer_covered_leaves_nothing_on_the_cpu():
    whole = _Tier(_Banks(range(24)))
    assert engine._mapped_cpu_layers(whole, 24) == frozenset()


def test_overlap_runs_only_when_every_mapped_layer_is_registered():
    """3060, 2026-09-23: a pin budget that covered 23 of 24 layers left one LOCKED layer beside
    the prefill overlap, and set_bank_sources refused the pair at boot. Only "nothing registered"
    used to turn the overlap off."""
    assert engine._mapped_overlap_blocker(None) is None  # no --moe-bank-ram at all
    assert engine._mapped_overlap_blocker(_Tier(_Banks(range(24)))) is None
    part = engine._mapped_overlap_blocker(_Tier(_Banks(range(23), registered_blocks=138)))
    assert part and "23 of 24" in part
    assert "nothing registered" in engine._mapped_overlap_blocker(_Tier(_Banks([], registered_blocks=0)))
    assert "nothing registered" in engine._mapped_overlap_blocker(_Tier(None))


def test_the_draft_head_layer_is_not_a_mapped_layer():
    """--spec-mtp appends the head's expert layer after the mapped ones, in pinned memory of its
    own. The engine passes only the mapped count, so a fully covered mapping leaves nothing on
    the CPU; asked about the appended index it would call it LOCKED -- the boot crash every
    Flash-Next run with --spec-mtp and --moe-bank-ram hit on its last rank (da98521)."""
    whole = _Tier(_Banks(range(24)))
    sources_with_head = 25
    assert engine._mapped_cpu_layers(whole, sources_with_head - 1) == frozenset()
    assert engine._mapped_cpu_layers(whole, sources_with_head) == frozenset({24})  # the old call


def test_nothing_covered_and_no_mapping():
    assert engine._mapped_cpu_layers(_Tier(_Banks([], registered_blocks=0)), 3) == frozenset(range(3))
    assert engine._mapped_cpu_layers(_Tier(None), 3) == frozenset(range(3))
    assert engine._mapped_cpu_layers(None, 3) == frozenset()  # no --moe-bank-ram at all


def test_a_bank_object_without_the_set_falls_back_to_all_or_nothing():
    """An older bank object (or a stub in a test) has no registered_layers."""

    class _Old:
        fully_registered = True

    class _OldPartial:
        fully_registered = False

    assert engine._mapped_cpu_layers(_Tier(_Old()), 3) == frozenset()
    assert engine._mapped_cpu_layers(_Tier(_OldPartial()), 3) == frozenset(range(3))
