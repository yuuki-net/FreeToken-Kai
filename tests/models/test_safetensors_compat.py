"""Staging the checkpoint read through host memory: when it turns on, and what it changes.

safetensors#858 / FreeToken-Kai#2: reading straight onto the GPU keeps page-locked host memory,
which only matters where the pin budget is small -- so the decision is about the budget, and the
patch must leave the loaders that already read on the host alone.
"""

from __future__ import annotations

import sys
import types

import pytest

from freetoken.models import safetensors_compat as sc
from freetoken.moe import pin_probe

GiB = 1 << 30


@pytest.fixture
def checkpoint(tmp_path):
    """A model directory whose shards come to 12 GiB, without writing 12 GiB."""
    for i in range(3):
        (tmp_path / f"model-{i}.safetensors").write_bytes(b"")
    sizes = [4 * GiB, 4 * GiB, 4 * GiB]
    return tmp_path, sizes


@pytest.fixture(autouse=True)
def _sized(monkeypatch):
    monkeypatch.setattr(sc, "checkpoint_bytes", lambda path: 12 * GiB)


def _capped(monkeypatch, budget_bytes):
    monkeypatch.setattr(pin_probe, "budget", lambda reserved=0, proc="/proc": budget_bytes)


def test_the_environment_decides_when_it_is_set(monkeypatch, tmp_path):
    _capped(monkeypatch, 1 * GiB)
    monkeypatch.setenv("FREETOKEN_SAFETENSORS_CPU_LOAD", "0")
    assert sc.wanted(str(tmp_path)) is False
    monkeypatch.setenv("FREETOKEN_SAFETENSORS_CPU_LOAD", "1")
    assert sc.wanted(str(tmp_path)) is True
    _capped(monkeypatch, None)  # even where nothing caps pinning
    assert sc.wanted(str(tmp_path)) is True
    assert sc.reason(str(tmp_path)) == "FREETOKEN_SAFETENSORS_CPU_LOAD"


def test_an_uncapped_host_reads_straight_onto_the_gpu(monkeypatch, tmp_path):
    monkeypatch.delenv("FREETOKEN_SAFETENSORS_CPU_LOAD", raising=False)
    _capped(monkeypatch, None)
    assert sc.wanted(str(tmp_path)) is False


def test_a_budget_too_small_for_the_checkpoint_stages(monkeypatch, tmp_path):
    monkeypatch.delenv("FREETOKEN_SAFETENSORS_CPU_LOAD", raising=False)
    _capped(monkeypatch, 1 * GiB)  # the reporter's host: 1 GiB against 12 GiB of shards
    assert sc.wanted(str(tmp_path)) is True
    assert "1.00 GiB" in sc.reason(str(tmp_path))
    _capped(monkeypatch, 12 * GiB)  # a host where the whole read fits the budget
    assert sc.wanted(str(tmp_path)) is False


def test_a_checkpoint_that_cannot_be_sized_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.delenv("FREETOKEN_SAFETENSORS_CPU_LOAD", raising=False)
    monkeypatch.setattr(sc, "checkpoint_bytes", lambda path: 0)
    _capped(monkeypatch, 1 * GiB)
    assert sc.wanted(str(tmp_path)) is False


# ------------------------------------------------------------------ the patch itself

class _FakeTensor:
    def __init__(self, name, device="cpu"):
        self.name, self.device = name, device

    def to(self, device):
        return _FakeTensor(self.name, str(device))


class _FakeSlice:
    def __getitem__(self, item):
        return _FakeTensor("sliced")

    def get_shape(self):
        return [2, 2]


class _FakeHandle:
    def __init__(self, path, framework, device):
        self.path, self.framework, self.device = path, framework, device
        self.entered = False

    def get_tensor(self, name):
        return _FakeTensor(name, self.device)

    def get_slice(self, name):
        return _FakeSlice()

    def keys(self):
        return ["a", "b"]

    def metadata(self):
        return {"format": "pt"}

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_safetensors(monkeypatch):
    opened: list[_FakeHandle] = []

    def safe_open(path, framework=None, device=None):
        h = _FakeHandle(path, framework, device)
        opened.append(h)
        return h

    mod = types.ModuleType("safetensors")
    mod.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", mod)
    return mod, opened


def test_a_cuda_read_is_opened_on_the_host_and_handed_over_on_the_device(fake_safetensors):
    mod, opened = fake_safetensors
    with sc.host_staged_reads(True, why="test") as on:
        assert on
        h = mod.safe_open("shard.safetensors", framework="pt", device="cuda:0")
        assert opened[0].device == "cpu"  # the read happened on the host
        t = h.get_tensor("w")
        assert t.device == "cuda:0"  # the consumer still gets its device
        assert h.get_slice("w")[...].device == "cuda:0"
        assert h.keys() == ["a", "b"] and h.metadata() == {"format": "pt"}
        with h as entered:
            assert entered is h and opened[0].entered


def test_a_host_read_is_not_touched(fake_safetensors):
    mod, opened = fake_safetensors
    with sc.host_staged_reads(True):
        h = mod.safe_open("shard.safetensors", framework="pt", device="cpu")
        assert opened[0].device == "cpu"
        assert h.get_tensor("w").device == "cpu"  # no proxy in the way
        assert type(h) is _FakeHandle


def test_a_positional_device_is_handled_too(fake_safetensors):
    mod, opened = fake_safetensors
    with sc.host_staged_reads(True):
        h = mod.safe_open("shard.safetensors", "pt", "cuda:1")
        assert opened[0].device == "cpu"
        assert h.get_tensor("w").device == "cuda:1"


def test_disabled_and_after_the_context_the_original_is_in_place(fake_safetensors):
    mod, opened = fake_safetensors
    original = mod.safe_open
    with sc.host_staged_reads(False) as on:
        assert on is False
        assert mod.safe_open is original
    assert mod.safe_open is original
    with sc.host_staged_reads(True):
        assert mod.safe_open is not original
    assert mod.safe_open is original
