"""ft mgr on the network: anyone may look, only this PC or a token holder may operate."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.app import build_app
from freetoken.daemon.logring import LogRing
from freetoken.daemon.profiles import ProfileStore
from freetoken.webui.auth import is_local, load_or_create_token

from .test_mgr_console_split import _Manager, _Probe

TOKEN = "s3cret-token"
PROFILE = {"model": "Qwen/Qwen3-0.6B", "port": None, "args": []}


def _client(tmp_path, *, client_host, base_url, console=True):
    app = build_app(
        manager=_Manager(), ring=LogRing(), probe=_Probe(), footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1), proxy_pool=ThreadPoolExecutor(1),
        profiles=ProfileStore(str(tmp_path / "profiles.json")) if console else None,
        console=console, console_cache_dir=str(tmp_path / "console"), write_token=TOKEN if console else None,
    )
    return TestClient(app, base_url=base_url, client=(client_host, 50000))


def _local(tmp_path):
    return _client(tmp_path, client_host="127.0.0.1", base_url="http://127.0.0.1:1901")


def _lan(tmp_path):
    return _client(tmp_path, client_host="192.168.1.20", base_url="http://192.168.1.10:1901")


def test_another_pc_can_look(tmp_path):
    c = _lan(tmp_path)
    for path in ("/profiles", "/engine/config", "/engine/health", "/host"):
        assert c.get(path).status_code == 200, path


def test_another_pc_cannot_operate_without_the_token(tmp_path):
    c = _lan(tmp_path)
    r = c.put("/profiles/p", json=PROFILE)
    assert r.status_code == 403 and r.json()["code"] == "write_needs_token"
    assert c.post("/engine/stop", json={}).status_code == 403
    assert c.post("/engine/stop", json={}, headers={"X-FT-Token": "wrong"}).status_code == 403
    assert c.get("/profiles").json()["profiles"] == []


def test_another_pc_operates_with_the_token(tmp_path):
    c = _lan(tmp_path)
    r = c.put("/profiles/p", json=PROFILE, headers={"X-FT-Token": TOKEN})
    assert r.status_code == 200, r.text
    assert [p["name"] for p in c.get("/profiles").json()["profiles"]] == ["p"]


def test_this_pc_operates_without_a_token(tmp_path):
    c = _local(tmp_path)
    assert c.put("/profiles/p", json=PROFILE).status_code == 200


def test_auth_tells_the_page_and_gives_the_token_only_locally(tmp_path):
    local = _local(tmp_path).get("/auth").json()
    assert local["local"] and local["write"] and local["token"] == TOKEN
    lan = _lan(tmp_path).get("/auth").json()
    assert not lan["local"] and not lan["write"] and lan["token"] is None
    good = _lan(tmp_path).get("/auth", headers={"X-FT-Token": TOKEN}).json()
    assert good["write"] and good["token_valid"] and good["token"] is None
    bad = _lan(tmp_path).get("/auth", headers={"X-FT-Token": "nope"}).json()
    assert not bad["write"] and bad["token_given"] and not bad["token_valid"]


def test_a_rebound_name_is_not_this_pc(tmp_path):
    # a page on another site whose name now resolves to 127.0.0.1: loopback address, foreign Host
    c = _client(tmp_path, client_host="127.0.0.1", base_url="http://evil.example:1901")
    assert c.put("/profiles/p", json=PROFILE, headers={"Origin": "http://evil.example:1901"}).status_code == 403


@pytest.mark.parametrize(
    "client, host, local",
    [
        ("127.0.0.1", "127.0.0.1:1901", True),
        ("::1", "[::1]:1901", True),
        ("::ffff:127.0.0.1", "localhost:1901", True),
        ("127.0.0.1", "192.168.1.10:1901", False),
        ("192.168.1.20", "127.0.0.1:1901", False),
        (None, "127.0.0.1", False),
        ("127.0.0.1", None, False),
    ],
)
def test_is_local(client, host, local):
    assert is_local(client, host) is local


def test_plain_daemon_is_untouched(tmp_path):
    c = _client(tmp_path, client_host="192.168.1.20", base_url="http://192.168.1.10:1900", console=False)
    # the check runs before routing: an unknown write reaching 404 means nothing refused it
    assert c.post("/no-such-route", json={}).status_code == 404


def test_token_file_is_kept_and_private(tmp_path):
    first = load_or_create_token(str(tmp_path))
    assert first and load_or_create_token(str(tmp_path)) == first
    mode = stat.S_IMODE(os.stat(tmp_path / "token").st_mode)
    assert mode == 0o600
