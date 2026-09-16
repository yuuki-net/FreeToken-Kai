"""ServeProbe against a port nobody listens on.

Under WSL's mirrored networking a connect to a closed loopback port gets no reset and waits out
the whole timeout, and the probe holds one lock for every serve document meanwhile: a dashboard
polling a stopped serve used to hold up every other console read. The kernel's socket table
answers the same question at once."""

from __future__ import annotations

import socket
import time

from freetoken.daemon.proxy import ServeProbe, local_listener


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_local_listener_sees_listening_and_closed_ports():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert local_listener(port) is True
    assert local_listener(_free_port()) is False


def test_probe_answers_a_closed_port_without_connecting():
    probe = ServeProbe(timeout_s=5.0)
    t = time.perf_counter()
    doc = probe.health(_free_port())
    assert doc == {"reachable": False, "status": "unreachable"}
    assert time.perf_counter() - t < 0.5


def test_an_injected_opener_is_still_asked():
    calls = []
    probe = ServeProbe(opener=lambda url, timeout: calls.append(url) or {"status": "ok"})
    assert probe.health(_free_port())["status"] == "ok" and calls


def test_an_unreadable_socket_table_falls_back_to_connecting():
    calls = []
    probe = ServeProbe(opener=lambda url, timeout: calls.append(url) or {"status": "ok"}, listening=lambda port: None)
    assert probe.health(1234)["reachable"] and calls
