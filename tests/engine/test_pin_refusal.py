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
    def __init__(self, registered_bytes, hot_blocks, registered_blocks):
        self.registered_bytes = registered_bytes
        self.hot_blocks = hot_blocks
        self.registered_blocks = registered_blocks

    @property
    def fully_registered(self) -> bool:  # mapped_bank's own rule
        return self.hot_blocks > 0 and self.registered_blocks == self.hot_blocks


class _Tier:
    def __init__(self, banks):
        self.banks = banks


def test_a_mapping_registered_in_part_is_not_addressable():
    """The warning promises every miss goes to the CPU; the decision has to agree.

    Reading "some bytes registered" left every layer labelled PINNED, and the copy plan then
    asked for the device address of a block that was never registered -- issue #2's crash with
    an explicit --moe-bank-ram size on a host that stops registering partway.
    """
    part = _Tier(_Banks(registered_bytes=1 * GiB, hot_blocks=144, registered_blocks=30))
    assert not engine._mapped_bank_is_addressable(part)


def test_a_mapping_registered_whole_is_addressable():
    whole = _Tier(_Banks(registered_bytes=16 * GiB, hot_blocks=144, registered_blocks=144))
    assert engine._mapped_bank_is_addressable(whole)


def test_nothing_registered_and_no_mapping_are_both_unaddressable():
    none = _Tier(_Banks(registered_bytes=0, hot_blocks=144, registered_blocks=0))
    assert not engine._mapped_bank_is_addressable(none)
    assert not engine._mapped_bank_is_addressable(_Tier(None))
    assert not engine._mapped_bank_is_addressable(None)
