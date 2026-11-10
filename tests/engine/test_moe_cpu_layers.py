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


@pytest.mark.parametrize("local_safe,remote_safe", [(True, True), (False, True), (True, False)])
def test_cpu_moe_graph_fallback_agrees_across_tp_ranks(monkeypatch, local_safe, remote_safe):
    import torch
    from freetoken.engine.engine import _disable_unsafe_cpu_moe_graphs

    config = SimpleNamespace(tp_info=SimpleNamespace(size=2), cuda_graph_bs=[1, 4], cuda_graph_max_bs=4)
    executor = SimpleNamespace(graph_capture_safe=lambda dtype: local_safe)
    group = object()
    calls = []

    def reduce(tensor, *, op, group):
        calls.append((op, group))
        tensor.fill_(bool(tensor.item()) and remote_safe)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    _disable_unsafe_cpu_moe_graphs(config, executor, torch.bfloat16, group)
    assert calls == [(torch.distributed.ReduceOp.MIN, group)]
    assert config.cuda_graph_bs == ([1, 4] if local_safe and remote_safe else [])
    assert config.cuda_graph_max_bs == (4 if local_safe and remote_safe else 0)


def _ram_cfg(*, pp: bool, local: int = 20, full: int = 40):
    return SimpleNamespace(
        is_pp=pp,
        tp_info=SimpleNamespace(size=2 if pp else 1),
        model_path="/nonexistent",
        model_config=SimpleNamespace(num_moe_layers=local),
        full_model_config=SimpleNamespace(num_moe_layers=full),
    )


def test_host_ram_check_compares_the_whole_models_banks_under_pp(monkeypatch, caplog):
    """A pp rank estimates its own slice; the RAM is shared by both, so the slice is scaled up."""
    import freetoken.engine.engine as engine
    import freetoken.moe.disk_probe as dp
    import freetoken.moe.expert_banks as eb

    GiB = 2**30
    monkeypatch.setattr(eb, "ftw_bank_bytes", lambda path: None)
    monkeypatch.setattr(eb, "bank_bytes_estimate", lambda mc, method=None, layer_methods=None: 28 * GiB)
    monkeypatch.setattr(dp, "cgroup_memory_headroom", lambda: None)
    monkeypatch.setattr(dp, "process_held_bytes", lambda: 0)  # not the test runner's own memory
    monkeypatch.setattr(dp, "meminfo", lambda: {"MemTotal": 62 * GiB, "MemAvailable": 57 * GiB})
    # 28 GiB for this rank's 20 of 40 layers -> 56 GiB for the model, + 2 GiB > 57 GiB free
    with pytest.raises(ValueError, match="56.0 GiB of host RAM across 2 ranks"):
        engine._check_host_ram(_ram_cfg(pp=True))
    # the same slice on one GPU is the whole model: 28 GiB fits
    engine._check_host_ram(_ram_cfg(pp=False, local=40, full=40))
    # an FTW checkpoint's metadata is the whole model already: not scaled again
    monkeypatch.setattr(eb, "ftw_bank_bytes", lambda path: 28 * GiB)
    engine._check_host_ram(_ram_cfg(pp=True))


def test_host_ram_check_skips_what_it_cannot_size(monkeypatch):
    import freetoken.engine.engine as engine
    import freetoken.moe.disk_probe as dp
    import freetoken.moe.expert_banks as eb

    monkeypatch.setattr(eb, "ftw_bank_bytes", lambda path: None)
    monkeypatch.setattr(eb, "bank_bytes_estimate", lambda mc, method=None, layer_methods=None: None)
    monkeypatch.setattr(dp, "meminfo", lambda: (_ for _ in ()).throw(AssertionError("must not read")))
    engine._check_host_ram(_ram_cfg(pp=False))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
