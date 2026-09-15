"""The prefill MoE kernel's e2m1 decode: arithmetic by default up to Ampere, the LUT beyond,
and the environment variable wins everywhere (the two forms are bit-identical)."""

from __future__ import annotations

import pytest

from freetoken.moe import fused_nvfp4 as fm
from freetoken.utils import arch


@pytest.fixture(autouse=True)
def _fresh_cache():
    fm._arith_dequant.cache_clear()
    yield
    fm._arith_dequant.cache_clear()


@pytest.mark.parametrize(
    "capability, arith",
    [((7, 5), True), ((8, 0), True), ((8, 6), True), ((8, 9), False), ((12, 0), False), (None, False)],
)
def test_default_by_capability(monkeypatch, capability, arith):
    monkeypatch.delenv("FREETOKEN_NVFP4_MOE_ARITH", raising=False)
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: capability)
    assert fm._arith_dequant() is arith


@pytest.mark.parametrize("value, arith", [("0", False), ("1", True)])
def test_environment_overrides(monkeypatch, value, arith):
    monkeypatch.setenv("FREETOKEN_NVFP4_MOE_ARITH", value)
    monkeypatch.setattr(arch, "_get_torch_cuda_version", lambda: (8, 6) if value == "0" else (12, 0))
    assert fm._arith_dequant() is arith
