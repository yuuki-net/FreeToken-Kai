"""``--moe-bank-ram auto`` against a measured pin cap, which need not be large.

Until the cap was measured it was always a fraction of the guest's RAM and so comfortably over
PIN_RESERVE_BYTES. A measured one need not be: 1 GiB on Windows 10 (FreeToken-Kai#2). It used to
size the resident rows -- ``min(by_ram, pin_cap)`` -- so a 1 GiB cap refused the flag outright on
a host with 22 GiB free. Residency is a RAM decision now; the cap limits REGISTRATION, which is
decided per layer while the mapping is built (moe/mapped_bank.py), because resident rows earn
their keep on the CPU executor too: it reads the mapping directly and residency is what saves it
the disk read.
"""

from __future__ import annotations

import pytest

from freetoken.moe import disk_probe as dp

GiB = dp.GiB
MEM = {"MemTotal": 24 * GiB, "MemAvailable": 22 * GiB}
BY_RAM = 22 * GiB - dp.NONBANK_PER_RANK_BYTES - max(dp.HEADROOM_MIN_BYTES, int(24 * GiB * dp.HEADROOM_FRACTION))


def test_a_cap_does_not_size_the_resident_rows(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "9.4")
    auto = dp.auto_bank_ram(MEM, ranks=1, proc="/proc")
    assert auto.total_bytes == BY_RAM  # RAM alone, not the cap
    assert "MemAvailable" in auto.reason()


@pytest.mark.parametrize("gb", ["2.0", "1.0", "0.5"])
def test_a_cap_at_or_under_the_reserve_no_longer_refuses(monkeypatch, gb):
    """This is issue #2's host. It used to raise; now the rows go resident and the message says
    how far registration will reach."""
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", gb)
    auto = dp.auto_bank_ram(MEM, ranks=1, proc="/proc")
    assert auto.total_bytes == BY_RAM
    why = auto.reason()
    assert "-0." not in why and "-1." not in why, why  # never a negative size
    assert "CUDA pin budget" in why and "the rest decode on the CPU" in why


def test_ram_can_still_be_the_thing_that_binds(monkeypatch):
    """The original message stays for the case it was written for."""
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "64")
    with pytest.raises(ValueError) as exc:
        dp.auto_bank_ram({"MemTotal": 24 * GiB, "MemAvailable": 5 * GiB}, ranks=1, proc="/proc")
    assert "Free some memory or pass a size" in str(exc.value)


def test_a_cap_over_the_residency_is_not_mentioned(monkeypatch):
    """Nothing to warn about when every resident row can be registered."""
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "64")
    assert "CUDA pin budget" not in dp.auto_bank_ram(MEM, ranks=1, proc="/proc").reason()
