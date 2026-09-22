"""The pin cap: which figure wins, what a record is valid for, and where a ladder stops."""

from __future__ import annotations

import json

from freetoken.moe import pin_probe as pp

GiB = pp.GiB


def _proc(root, total_gib=24.0, available_gib=None):
    d = root / "proc"
    d.mkdir(parents=True, exist_ok=True)
    total = int(total_gib * GiB) // 1024
    available = int((available_gib if available_gib is not None else total_gib * 0.9) * GiB) // 1024
    (d / "meminfo").write_text(f"MemTotal:       {total} kB\nMemAvailable:   {available} kB\n")
    return str(d)


def test_the_estimate_is_a_fraction_of_the_guests_ram(tmp_path):
    assert pp.estimate(_proc(tmp_path, 24.0)) == int(24 * GiB * 0.4)


def test_a_recorded_figure_beats_the_estimate(monkeypatch, tmp_path):
    proc = _proc(tmp_path, 24.0)
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    assert pp.budget(proc=proc) == int(24 * GiB * 0.4)
    assert "estimated" in pp.source(proc)

    assert pp.remember(1 * GiB, how="ft doctor pin", proc=proc)
    assert pp.remembered(proc) == (1 * GiB, "ft doctor pin")
    assert pp.budget(proc=proc) == 1 * GiB
    assert pp.source(proc) == "measured (ft doctor pin)"


def test_the_environment_beats_a_recorded_figure(monkeypatch, tmp_path):
    proc = _proc(tmp_path, 24.0)
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    pp.remember(1 * GiB, how="refused mid-load", proc=proc)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "3.5")
    assert pp.budget(proc=proc) == int(3.5 * GiB)
    assert pp.source(proc) == "FREETOKEN_PIN_BUDGET_GB"


def test_an_uncapped_platform_has_no_budget(monkeypatch, tmp_path):
    proc = _proc(tmp_path, 24.0)
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: False)
    assert pp.budget(proc=proc) is None
    # even with a record: a host that does not cap pinning is not asking this question
    pp.remember(1 * GiB, how="ft doctor pin", proc=proc)
    assert pp.budget(proc=proc) is None


def test_reserved_bytes_come_off_the_budget(monkeypatch, tmp_path):
    proc = _proc(tmp_path, 24.0)
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    pp.remember(4 * GiB, how="ft doctor pin", proc=proc)
    assert pp.budget(reserved=1 * GiB, proc=proc) == 3 * GiB
    assert pp.budget(reserved=99 * GiB, proc=proc) == 0  # never negative


def test_a_record_does_not_carry_across_guest_sizes(monkeypatch, tmp_path):
    """.wslconfig memory= changes what is usable, so the old figure must not be reused."""
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    small, large = _proc(tmp_path / "a", 8.0), _proc(tmp_path / "b", 24.0)
    pp.remember(5 * GiB, how="ft doctor pin", proc=small)
    assert pp.remembered(small) == (5 * GiB, "ft doctor pin")
    assert pp.remembered(large) is None
    # both keys live in the one file, kept apart
    with open(pp.cache_path()) as fh:
        doc = json.load(fh)
    assert len(doc) == 1 and "memtotal=" in next(iter(doc))


def test_a_junk_record_is_ignored_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    proc = _proc(tmp_path, 24.0)
    with open(pp.cache_path(), "w") as fh:
        fh.write("not json at all")
    assert pp.remembered(proc) is None
    assert pp.budget(proc=proc) == int(24 * GiB * 0.4)
    # and a later write repairs the file
    assert pp.remember(2 * GiB, how="ft doctor pin", proc=proc)
    assert pp.remembered(proc) == (2 * GiB, "ft doctor pin")


def test_a_refusal_is_used_as_is_including_below_the_estimate(monkeypatch, tmp_path):
    """The Windows 10 case: the cap really is 1 GiB against a 9.6 GiB estimate, and the whole
    point is that the smaller figure wins."""
    proc = _proc(tmp_path, 24.0)
    monkeypatch.setattr(pp, "is_pin_capped", lambda release=None: True)
    assert pp.budget(proc=proc) == pp.estimate(proc)
    pp.remember(1 * GiB, how="refused mid-load", proc=proc)
    assert pp.budget(proc=proc) == 1 * GiB
    assert pp.source(proc) == "measured (refused mid-load)"


def test_the_ladder_stops_short_of_the_guests_own_ram(tmp_path):
    """Past MemAvailable the OOM killer answers before the driver does, and a killed process
    measures nothing -- so that outcome is reported instead of being recorded as a cap.

    With this little available the ladder returns before locking anything, which is also why
    it never reaches CUDA: measuring is what needs a GPU, declining to measure is not."""
    proc = _proc(tmp_path, 24.0, available_gib=0.25)
    m = pp.measure(24 * GiB, proc=proc)
    assert m.ram_limited and not m.refused and not m.ceiling_hit
    assert m.locked_bytes == 0
    assert "free RAM" in m.summary()
