"""Where a decode step's time goes, read now and then for the web console.

A decode step runs as one captured CUDA graph, so nothing inside it can be timed from the host.
So a second copy of each decode graph is captured with CUDA events
(``torch.cuda.Event(enable_timing=True, external=True)``) recorded at its start and end and at the
seams of every offloaded MoE layer's expert work. Every ``interval_s`` seconds the engine replays
that copy for one step instead of the plain one, waits for it and reads the events. The step is a
real one -- the same kernels, the same overlap -- so the parts add up to its GPU time:

* ``route`` -- the slot-cache bookkeeping (routing record, cache lookup, the CPU/GPU split)
* ``fetch`` -- copying missed experts from host RAM into the GPU cache (PCIe; a row the page
  cache does not hold is read from disk inside it)
* ``gpu_experts`` -- the expert GEMMs on the GPU
* ``cpu`` -- the GPU waiting for experts computed on the CPU: a whole CPU layer, or what the
  hybrid backend's CPU share takes beyond the GPU's own work on that layer
* ``other`` -- the rest of the forward: attention, dense layers, the head

The events cost about 1 us each on the GPU (200 of them, 5 per layer on a 40-layer model, added
0.22 ms to a step on an RTX 2060), which is why they live in a separate graph: the plain steps
carry none. The timed copy shares the plain graph's memory pool, so it costs capture time rather
than VRAM. The read costs one host wait every ``interval_s`` (default 30 s,
FREETOKEN_DECODE_SAMPLE_S; 0 turns the timed copies off). Kept free of the engine and the
kernels, like utils/prefill_profile.py.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator

import torch

PARTS = ("route", "fetch", "gpu_experts", "cpu", "other")

# Seams of one MoE layer's expert work (the event index within the layer's row).
START, ROUTED, FETCHED, COMPUTED, DONE = range(5)

TIMELINE: "DecodeTimeline | None" = None


def timeline() -> "DecodeTimeline | None":
    """The timeline to mark from a MoE layer, or None outside a timed decode forward."""
    tl = TIMELINE
    return tl if tl is not None and tl.active else None


def run_forward(forward: Callable[[], torch.Tensor], timed: bool = True) -> torch.Tensor:
    """Run (or capture) one decode forward, inside the timeline when ``timed`` and there is one."""
    tl = TIMELINE
    if tl is None or not timed:
        return forward()
    with tl.forward():
        return forward()


def interval_from_env(default: float = 30.0) -> float:
    try:
        return max(0.0, float(os.getenv("FREETOKEN_DECODE_SAMPLE_S", default)))
    except ValueError:
        return default


def _event() -> torch.cuda.Event:
    # external: recordable inside a stream capture, where it becomes an event record node
    return torch.cuda.Event(enable_timing=True, external=True)


class DecodeTimeline:
    """The events one decode forward records. Allocated once, before graph capture, and never
    freed while a graph that records them can be replayed: replaying a record node of a freed
    event crashes the process."""

    def __init__(self, num_layers: int) -> None:
        self.begin, self.end = _event(), _event()
        self.marks = [[_event() for _ in range(DONE + 1)] for _ in range(num_layers)]
        self.kind: dict[int, str] = {}  # layer -> "cpu" | "gpu", as last recorded
        self.active = False
        self.recorded = False  # a timed forward has been recorded (captured or run) at least once

    def mark(self, layer: int, seam: int) -> None:
        self.marks[layer][seam].record()

    def layer_kind(self, layer: int, kind: str) -> None:
        self.kind[layer] = kind

    @contextmanager
    def forward(self) -> Iterator[None]:
        """Wrap one decode forward (eager, or being captured into a graph)."""
        self.begin.record()
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.end.record()
            self.recorded = True

    def read(self) -> dict | None:
        """Durations (ms) of the last recorded forward; waits for it to finish."""
        if not self.recorded:
            return None
        self.end.synchronize()
        el = torch.cuda.Event.elapsed_time
        total = el(self.begin, self.end)
        ms = dict.fromkeys(PARTS, 0.0)
        for layer, kind in self.kind.items():
            m = self.marks[layer]
            try:
                if kind == "cpu":
                    ms["cpu"] += el(m[START], m[DONE])
                else:
                    ms["route"] += el(m[START], m[ROUTED])
                    ms["fetch"] += el(m[ROUTED], m[FETCHED])
                    ms["gpu_experts"] += el(m[FETCHED], m[COMPUTED])
                    ms["cpu"] += el(m[COMPUTED], m[DONE])
            except RuntimeError:  # a layer whose events this forward did not record
                continue
        if total <= 0:
            return None
        ms["other"] = max(0.0, total - sum(ms[k] for k in PARTS if k != "other"))
        return {"total_ms": total, "ms": ms}


class DecodeSampler:
    """Decides when to read the timeline and keeps the recent readings."""

    KEEP = 10

    def __init__(
        self, timeline: DecodeTimeline, interval_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.timeline = timeline
        self.interval_s = interval_s
        self.clock = clock
        # the first reading waits one interval: the opening steps of a session are not typical
        self._next = clock() + interval_s
        self.samples: deque[dict] = deque(maxlen=self.KEEP)

    def due(self) -> bool:
        return self.interval_s > 0 and self.clock() >= self._next

    def collect(self, rows: int) -> None:
        """Read the step that was just launched (host wait) and keep it."""
        self._next = self.clock() + self.interval_s
        got = self.timeline.read()
        if got is not None:
            self.samples.append({"time": time.time(), "rows": int(rows), **got})

    def snapshot(self) -> dict | None:
        """Mean of the kept readings (what the console shows) plus the latest one."""
        if not self.samples:
            return None
        n = len(self.samples)
        return {
            "interval_s": self.interval_s,
            "samples": n,
            "total_ms": sum(x["total_ms"] for x in self.samples) / n,
            "ms": {k: sum(x["ms"][k] for x in self.samples) / n for k in PARTS},
            "rows": self.samples[-1]["rows"],
            "last": self.samples[-1],
        }
