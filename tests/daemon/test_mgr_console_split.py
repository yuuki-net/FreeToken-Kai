"""``ft daemon`` is upstream's supervisor; ``ft mgr`` is the same thing plus kai's web console.

The split is the point: a daemon serves none of the console routes, so upstream's contract is
untouched and everything this fork added lives under one command.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.app import build_app
from freetoken.daemon.logring import LogRing
from freetoken.daemon.profiles import ProfileStore

CONSOLE_ROUTES = ["/ui/env.json", "/profiles", "/models", "/host", "/engine/config", "/engine/external", "/engine/view/stats"]
UPSTREAM_ROUTES = ["/health", "/engine/status", "/engine/metrics", "/engine/health"]
# /engine/logs is an endless SSE stream: ask the route table for it rather than opening it
STREAMING_ROUTES = ["/engine/logs"]


class _Probe:
    """The console's read-only views proxy the serve; with none running they still must answer."""

    def health(self, port):
        return {"reachable": False, "status": "unreachable"}

    stats = health

    def get(self, path, port):
        return {"reachable": False, "status": "unreachable"}


class _Manager:
    def status(self):
        return {"running": False, "port": None}

    def current_pid(self):
        return None

    def serve_args(self):
        return []


def _client(tmp_path, console):
    app = build_app(
        manager=_Manager(), ring=LogRing(), probe=_Probe(), footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1), proxy_pool=ThreadPoolExecutor(1),
        profiles=ProfileStore(str(tmp_path / "profiles.json")) if console else None,
        console=console, console_cache_dir=str(tmp_path / "console"),
    )
    return TestClient(app)


@pytest.mark.parametrize("path", CONSOLE_ROUTES)
def test_daemon_does_not_serve_the_console(tmp_path, path):
    assert _client(tmp_path, console=False).get(path).status_code == 404


@pytest.mark.parametrize("path", CONSOLE_ROUTES)
def test_mgr_serves_the_console(tmp_path, path):
    assert _client(tmp_path, console=True).get(path).status_code != 404


@pytest.mark.parametrize("path", UPSTREAM_ROUTES)
def test_both_keep_the_upstream_routes(tmp_path, path):
    for console in (False, True):
        assert _client(tmp_path, console=console).get(path).status_code != 404


@pytest.mark.parametrize("path", STREAMING_ROUTES)
def test_both_keep_the_streaming_routes(tmp_path, path):
    for console in (False, True):
        paths = {r.path for r in _client(tmp_path, console=console).app.routes}
        assert path in paths


def test_the_page_knows_which_command_served_it(tmp_path):
    assert _client(tmp_path, console=True).get("/ui/env.json").json()["mode"] == "mgr"


def test_defaults_do_not_collide():
    from freetoken.daemon import server

    assert server.DEFAULT_PORT == 1900 and server.MGR_PORT == 1901
    assert server._default_state_dir(False) != server._default_state_dir(True)
