"""The benchmark must not propose or try a start this host's pin cap would refuse.

``--moe-strategy offload`` page-locks every expert bank, so on a host whose measured cap is
short of them the server does not start (FreeToken-Kai#2). Before the cap was measured the
benchmark had no way to know, and would spend a trial finding out.
"""

from __future__ import annotations

from freetoken.webui import hwbench, search

GiB = 1 << 30


def _hw(cap_gib=None, banks_gib=None, **measurements):
    pin = {}
    if cap_gib is not None:
        pin["cap_bytes"] = int(cap_gib * GiB)
    if banks_gib is not None:
        pin["banks_bytes"] = int(banks_gib * GiB)
    return {"measurements": {"pin": pin or None, **measurements}}


MOE = {"num_experts": 128, "quant": "modelopt"}


def test_a_cap_over_the_banks_leaves_the_candidate_alone():
    assert search.offload_unpinnable(_hw(12, 4)) is None
    assert hwbench.offload_possible(_hw(12, 4)["measurements"])[0] is True


def test_no_measurement_leaves_the_candidate_alone():
    """Nothing measured is not evidence against the candidate."""
    for hw in ({}, {"measurements": {}}, _hw(), _hw(cap_gib=12), _hw(banks_gib=17)):
        assert search.offload_unpinnable(hw) is None
        assert hwbench.offload_possible((hw.get("measurements") or {}))[0] is True


def test_a_cap_short_of_the_banks_rules_it_out():
    hw = _hw(1, 17)  # the reporter's host: 1 GiB against 17 GiB of experts
    why = search.offload_unpinnable(hw)
    assert why and "1.00 GiB" in why[0] and "17.0 GiB" in why[0]
    assert "1.00 GiB" in why[1] and "does not start" in why[1]

    ok, ja, en = hwbench.offload_possible(hw["measurements"])
    assert not ok and "1.00 GiB" in ja and "cannot start here" in en


def test_the_plan_skips_the_offload_trial_and_says_why():
    args = ["--moe-strategy", "hybrid", "--moe-cpu-layers", "auto"]
    keys = lambda hw: {c.key for c in search.plan(args, MOE, hw, modules=set())}  # noqa: E731

    assert "strategy_offload" in keys(_hw(12, 4))
    assert "strategy_offload" not in keys(_hw(1, 17))

    listed = search.unavailable(MOE, _hw(1, 17), modules=set())
    assert any(e["flag"] == "--moe-strategy offload" for e in listed)
    assert not any(e["flag"] == "--moe-strategy offload"
                   for e in search.unavailable(MOE, _hw(12, 4), modules=set()))


def test_a_dense_model_is_not_told_about_expert_banks():
    assert not any(e["flag"] == "--moe-strategy offload"
                   for e in search.unavailable({}, _hw(1, 17), modules=set()))


def test_a_cap_far_below_the_banks_makes_fetching_nothing_a_standard_trial():
    """The fetch reads page-locked memory, so a cap that covers almost none of the banks leaves
    a couple of layers fetching -- measured slower than not fetching at all (guides/58 7.3)."""
    args = ["--moe-strategy", "hybrid", "--moe-cpu-layers", "auto"]
    std = lambda hw: {c.key for c in search.plan(args, MOE, hw, "standard", modules=set())}  # noqa: E731

    assert "fetch_none" in std(_hw(1, 9.47))  # the reporter's regime: 11% of the banks
    assert "fetch_none" not in std(_hw(9.39, 9.47))  # the 2060: the cap covers them
    assert "fetch_none" not in std({})  # nothing measured stays a thorough-only trial
    assert "fetch_none" in {c.key for c in search.plan(args, MOE, _hw(9.39, 9.47), "thorough", modules=set())}


def test_the_starved_fetch_says_what_it_measured():
    why = search.pin_starves_the_fetch(_hw(1, 9.47))
    assert why and "1.00 GiB" in why[0] and "9.5 GiB" in why[0]
    assert "1.00 GiB" in why[1] and "not fetching can be faster" in why[1]
    assert search.pin_starves_the_fetch(_hw(9.39, 9.47)) is None
    for hw in ({}, {"measurements": {}}, _hw(), _hw(cap_gib=1), _hw(banks_gib=9.47)):
        assert search.pin_starves_the_fetch(hw) is None


def test_the_starved_host_is_told_why_in_the_candidate_text():
    args = ["--moe-strategy", "hybrid", "--moe-cpu-layers", "auto"]
    c = next(c for c in search.plan(args, MOE, _hw(1, 9.47), "standard", modules=set()) if c.key == "fetch_none")
    assert "1.00 GiB" in c.what_ja and "1.00 GiB" in c.what_en


# ------------------------------------------------------------------ what derive proposes

def _measurements(cpu_gbs, pcie_gbs, **extra):
    return {"cpu_moe": {"best_gbs": cpu_gbs, "threads": 6, "cores": 8},
            "gather": {"0": {"gbs": pcie_gbs}}, **extra}


def _note(notes, flag):
    return next((n for n in notes if n["flag"] == flag), None)


def test_derive_proposes_offload_where_the_transfer_wins_and_it_can_pin():
    notes = hwbench.derive({**_measurements(10.0, 20.0), "pin": {"cap_bytes": 12 * GiB, "banks_bytes": 4 * GiB}})
    assert _note(notes, "--moe-strategy")["value"] == "offload"


def test_derive_proposes_hybrid_where_offload_could_not_start(monkeypatch):
    """The transfer is still the faster side; it just cannot be used on this host."""
    notes = hwbench.derive({**_measurements(10.0, 20.0), "pin": {"cap_bytes": 1 * GiB, "banks_bytes": 17 * GiB}})
    strategy = _note(notes, "--moe-strategy")
    assert strategy["value"] == "hybrid"
    assert "1.00 GiB" in strategy["why"] and "cannot start here" in strategy["why_en"]
    assert _note(notes, "--moe-cpu-threads")["value"] == "6"  # and the threads it needs come with it


def test_derive_still_proposes_offload_for_a_format_the_cpu_cannot_serve():
    """No CPU measurement at all: offload is the only thing there is, so it is proposed with
    the caveat rather than replaced by something that cannot run."""
    notes = hwbench.derive({"gather": {"0": {"gbs": 20.0}},
                            "pin": {"cap_bytes": 1 * GiB, "banks_bytes": 17 * GiB}})
    strategy = _note(notes, "--moe-strategy")
    assert strategy["value"] == "offload"
    assert "--moe-bank-ram" in strategy["why"] and "--moe-bank-ram" in strategy["why_en"]
