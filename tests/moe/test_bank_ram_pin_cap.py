"""``--moe-bank-ram auto`` against a pin cap smaller than the reserve it keeps.

Until the cap was measured it was always a fraction of the guest's RAM and so comfortably over
PIN_RESERVE_BYTES. A measured one need not be: 1 GiB on Windows 10 (FreeToken-Kai#2), which used
to reach the refusal as a negative size, with advice to free memory that was not the constraint.
"""

from __future__ import annotations

import pytest

from freetoken.moe import disk_probe as dp

GiB = dp.GiB
MEM = {"MemTotal": 24 * GiB, "MemAvailable": 22 * GiB}


def test_a_roomy_cap_sizes_the_banks_from_it(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "9.4")
    auto = dp.auto_bank_ram(MEM, ranks=1, proc="/proc")
    assert auto.total_bytes == int(9.4 * GiB) - dp.PIN_RESERVE_BYTES
    assert "the CUDA pin budget" in auto.reason()


@pytest.mark.parametrize("gb", ["2.0", "1.0", "0.5"])
def test_a_cap_at_or_under_the_reserve_refuses_without_a_negative_size(monkeypatch, gb):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", gb)
    with pytest.raises(ValueError) as exc:
        dp.auto_bank_ram(MEM, ranks=1, proc="/proc")
    msg = str(exc.value)
    assert "-0." not in msg and "-1." not in msg and "-2." not in msg, msg
    # and it blames the cap, not memory the host has plenty of
    assert "Freeing memory does not raise this cap" in msg
    assert "pass an explicit size" in msg and "--moe-cpu-layers 1.0" in msg
    # what RAM alone would have allowed (22 available, less the server's 4.5 and 2.0 of margin),
    # so the reader can see which of the two terms is the one that binds
    assert "15.5 GiB" in msg


def test_ram_can_still_be_the_thing_that_binds(monkeypatch):
    """The original message stays for the case it was written for."""
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "64")
    with pytest.raises(ValueError) as exc:
        dp.auto_bank_ram({"MemTotal": 24 * GiB, "MemAvailable": 5 * GiB}, ranks=1, proc="/proc")
    assert "Free some memory or pass a size" in str(exc.value)
