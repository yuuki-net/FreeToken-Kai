"""utils/decode_sample.py: the decode timeline (event records captured into the decode graph) and
the sampler that reads it now and then."""
from __future__ import annotations

import pytest
import torch

from freetoken.utils import decode_sample as ds


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _FakeTimeline:
    def __init__(self, reading):
        self.reading = reading
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.reading


def _reading(total, **ms):
    parts = dict.fromkeys(ds.PARTS, 0.0)
    parts.update(ms)
    return {"total_ms": total, "ms": parts}


def test_first_reading_waits_one_interval_then_every_interval():
    clock = _Clock()
    tl = _FakeTimeline(_reading(30.0, other=30.0))
    s = ds.DecodeSampler(tl, 30.0, clock=clock)
    assert not s.due()
    clock.now += 30
    assert s.due()
    s.collect(rows=1)
    assert tl.reads == 1 and not s.due()
    clock.now += 29
    assert not s.due()
    clock.now += 1
    assert s.due()


def test_zero_interval_never_reads():
    clock = _Clock()
    s = ds.DecodeSampler(_FakeTimeline(None), 0.0, clock=clock)
    clock.now += 1e6
    assert not s.due()


def test_nothing_recorded_keeps_nothing():
    s = ds.DecodeSampler(_FakeTimeline(None), 30.0, clock=_Clock())
    s.collect(rows=1)
    assert s.snapshot() is None


def test_snapshot_is_the_mean_of_the_kept_readings():
    tl = _FakeTimeline(None)
    s = ds.DecodeSampler(tl, 30.0, clock=_Clock())
    for i in range(ds.DecodeSampler.KEEP + 5):
        tl.reading = _reading(10.0 + i, fetch=float(i))
        s.collect(rows=2)
    snap = s.snapshot()
    assert snap["samples"] == ds.DecodeSampler.KEEP and snap["rows"] == 2
    assert snap["ms"]["fetch"] == sum(range(5, 15)) / 10
    assert snap["last"]["total_ms"] == 24.0


def test_run_forward_without_a_timeline_just_runs():
    ds.TIMELINE = None
    assert ds.run_forward(lambda: 7) == 7
    assert ds.timeline() is None


def test_interval_from_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DECODE_SAMPLE_S", "0")
    assert ds.interval_from_env() == 0
    monkeypatch.setenv("FREETOKEN_DECODE_SAMPLE_S", "x")
    assert ds.interval_from_env() == 30.0


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_cuda
def test_a_captured_forward_times_its_parts_on_every_replay():
    """The parts are read from event records inside a CUDA graph and add up to the step."""
    a = torch.randn(1024, 1024, device="cuda")
    tl = ds.DecodeTimeline(num_layers=2)
    s = torch.cuda.Stream()

    def forward():
        x = a
        for layer, kind in ((0, "gpu"), (1, "cpu")):
            t = ds.timeline()
            assert t is tl
            t.mark(layer, ds.START)
            x = x @ a
            if kind == "gpu":
                t.mark(layer, ds.ROUTED)
                for _ in range(4):
                    x = x @ a  # "fetch"
                t.mark(layer, ds.FETCHED)
                for _ in range(8):
                    x = x @ a  # "gpu_experts"
                t.mark(layer, ds.COMPUTED)
            x = x @ a
            t.layer_kind(layer, kind)
            t.mark(layer, ds.DONE)
        return x @ a  # "other"

    ds.TIMELINE = tl
    try:
        with torch.cuda.stream(s):
            ds.run_forward(forward)  # eager warm-up records too
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                ds.run_forward(forward)
        assert ds.timeline() is None  # only inside the forward
        g.replay()
        r = tl.read()
    finally:
        ds.TIMELINE = None
    ms = r["ms"]
    assert abs(sum(ms.values()) - r["total_ms"]) < 1e-3
    assert ms["gpu_experts"] > ms["fetch"] > 0 and ms["cpu"] > 0 and ms["other"] > 0


def test_the_sampled_step_is_read_after_the_host_context_exits():
    """3060, 2026-09-23: Flash-Next's disk PLE releases a decode graph from forward_host_ctx's exit
    (the deferred fill signals the flag the graph waits on). Reading the timed step inside that
    context waited for a forward that was waiting for the exit: every sampled plain decode step
    hung for good (--spec-mtp 0, the one-card flash1 profile). The read has to come after it."""
    import ast
    import inspect
    import textwrap

    from freetoken.engine import engine

    tree = ast.parse(textwrap.dedent(inspect.getsource(engine.Engine._forward_batch)))

    def calls(node):
        return [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "collect"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "sampler"
        ]

    host = [
        w for w in ast.walk(tree) if isinstance(w, ast.With)
        and any("forward_host_ctx" in ast.unparse(item.context_expr) for item in w.items)
    ]
    assert len(host) == 1
    assert not calls(host[0]), "sampler.collect inside forward_host_ctx deadlocks the disk PLE"
    assert calls(tree), "the sampled step is no longer read at all"
