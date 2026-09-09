"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None, bank_ram=None):
    return SimpleNamespace(moe_strategy=backend, moe_cpu_layers=spec, moe_bank_ram=bank_ram, model_config=None)


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


def test_auto_defers_to_bank_ram(monkeypatch):
    """The split exists to answer "more banks than we may pin"; --moe-bank-ram answers it first
    and differently, so `auto` picks nothing there (an explicit spec still means what it says)."""
    import freetoken.engine.engine as engine

    # a pin budget the banks blow through: without --moe-bank-ram this is what makes auto fire
    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda reserved=0: 8 << 30)
    monkeypatch.setattr(engine, "_bank_bytes", lambda config, method=None: 56 << 30)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: True)

    assert resolve(_cfg("hybrid", "auto"), L) != frozenset()
    assert resolve(_cfg("hybrid", "auto", bank_ram="60G"), L) == frozenset()
    assert len(resolve(_cfg("hybrid", "8", bank_ram="60G"), L)) == 8


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
