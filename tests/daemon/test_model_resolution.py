"""A profile's model must be something ft serve can open.

A bare folder name is read as a Hugging Face repo id, and the engine dies at load with a hub 401 --
which is what a profile saved before this check did. Resolution happens when a profile is saved AND
again on start/switch, so an old profile (or ``ft daemon start``, or curl) cannot get past it either.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.app import build_app
from freetoken.daemon.logring import LogRing
from freetoken.webui.models import resolve_model


def _model_dir(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["X"], "model_type": "x"}))
    (d / "model.safetensors").write_bytes(b"0" * 16)
    return d


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    _model_dir(root, "Ornith-1.5-35B-A3B-NVFP4")
    _model_dir(root, "gpt-oss-20b")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # so ~/models cannot leak the real machine in
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("FREETOKEN_MODELS_DIRS", str(root))
    return root


def test_bare_name_resolves_to_the_local_folder(models):
    assert resolve_model("Ornith-1.5-35B-A3B-NVFP4") == str(models / "Ornith-1.5-35B-A3B-NVFP4")


def test_paths_and_repo_ids_pass_through(models, tmp_path):
    assert resolve_model(str(models / "gpt-oss-20b")) == str(models / "gpt-oss-20b")
    assert resolve_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"  # a repo id is ft serve's problem


@pytest.mark.parametrize("bad", ["", "   ", "no-such-model", "/definitely/not/here"])
def test_unusable_models_are_refused(models, bad):
    with pytest.raises(ValueError):
        resolve_model(bad)


def test_unknown_name_error_lists_what_is_there(models):
    with pytest.raises(ValueError, match="gpt-oss-20b"):
        resolve_model("gpt-oss-20B")  # case matters: it is a folder name


class _Manager:
    def __init__(self):
        self.started = None

    def status(self):
        return {"running": False, "port": None}

    def current_pid(self):
        return None

    def serve_args(self):
        return []

    def start(self, model, port, args):
        self.started = (model, port, args)
        return {"pid": 1, "model": model, "port": port}


@pytest.fixture
def client(models):
    manager = _Manager()
    app = build_app(
        manager=manager, ring=LogRing(), probe=None, footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1), proxy_pool=ThreadPoolExecutor(1), default_serve_port=1919,
    )
    return TestClient(app), manager


def test_start_resolves_a_bare_name_before_spawning(client, models):
    c, manager = client
    r = c.post("/engine/start", json={"model": "gpt-oss-20b", "port": 1919, "args": ["--moe-collect-stats"]})
    assert r.status_code == 200, r.text
    assert manager.started == (str(models / "gpt-oss-20b"), 1919, ["--moe-collect-stats"])


def test_start_refuses_a_model_that_is_not_there(client):
    c, manager = client
    r = c.post("/engine/start", json={"model": "not-here", "port": 1919})
    assert r.status_code == 400
    assert "not-here" in r.json()["detail"]
    assert manager.started is None
