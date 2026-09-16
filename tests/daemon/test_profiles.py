"""Launch profiles the web console saves in the daemon's state dir."""

from __future__ import annotations

import json

import pytest

from freetoken.daemon.profiles import ProfileStore


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Two model folders, and a HOME of our own: resolution must not see the real machine."""
    root = tmp_path / "models"
    for name in ("flash-next", "gpt-oss-120b"):
        d = root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"model_type": "x"}))
        (d / "model.safetensors").write_bytes(b"0" * 16)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("FREETOKEN_MODELS_DIRS", str(root))
    return root


def test_put_list_delete_roundtrip(tmp_path, models):
    store = ProfileStore(str(tmp_path / "profiles.json"))
    assert store.list() == []
    store.put("3060x2 pp・128k", str(models / "flash-next"), 1919, ["--pp-size", "2", "--kv-cache-dtype", "q8_0"])
    store.put("gpt-oss", "gpt-oss-120b", None, [])  # a bare name resolves to the folder
    names = [p["name"] for p in store.list()]
    assert names == ["gpt-oss", "3060x2 pp・128k"]  # newest first
    reloaded = ProfileStore(str(tmp_path / "profiles.json")).list()
    assert next(p for p in reloaded if p["port"] == 1919)["args"] == ["--pp-size", "2", "--kv-cache-dtype", "q8_0"]
    assert next(p for p in reloaded if p["name"] == "gpt-oss")["model"] == str(models / "gpt-oss-120b")
    assert store.delete("gpt-oss") and not store.delete("gpt-oss")
    assert [p["name"] for p in store.list()] == ["3060x2 pp・128k"]


@pytest.mark.parametrize("name,model,port", [
    ("", "flash-next", None),          # no name
    ("a/b", "flash-next", None),       # a name that cannot be a file
    ("ok", "", None),                  # no model
    ("ok", "flash-next", 70000),       # port out of range
    ("ok", "not-a-model-here", None),  # nothing of that name on this host
])
def test_put_rejects_bad_input(tmp_path, models, name, model, port):
    with pytest.raises(ValueError):
        ProfileStore(str(tmp_path / "p.json")).put(name, model, port, [])


def test_home_is_expanded_in_model_and_flag_values(tmp_path, models):
    home = tmp_path / "home"
    (home / "models" / "m").mkdir(parents=True)
    (home / "models" / "m" / "config.json").write_text("{}")
    (home / "models" / "m" / "model.safetensors").write_bytes(b"0")
    p = ProfileStore(str(tmp_path / "p.json")).put(
        "x", "~/models/m", None, ["--moe-bank-stats", "~/stats.json", "--host", "0.0.0.0"]
    )
    assert p["model"] == str(home / "models" / "m")
    assert p["args"] == ["--moe-bank-stats", str(home / "stats.json"), "--host", "0.0.0.0"]
