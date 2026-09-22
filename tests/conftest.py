import pytest


@pytest.fixture(autouse=True)
def _default_quant_backend():
    """Tests that install kernel requests must not leak them into the next test."""
    yield
    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend())


@pytest.fixture(autouse=True)
def _own_pin_cap_record(tmp_path_factory, monkeypatch):
    """Keep every test off the developer's real ~/.cache/freetoken/pin_cap.json.

    freetoken.moe.pin_probe reads that file to answer what this host page-locks, so a machine
    that has run ``ft doctor pin`` would otherwise feed a measured cap into tests that mean to
    exercise the fallback estimate."""
    monkeypatch.setenv("FREETOKEN_PIN_CAP_FILE", str(tmp_path_factory.mktemp("pincap") / "pin_cap.json"))
