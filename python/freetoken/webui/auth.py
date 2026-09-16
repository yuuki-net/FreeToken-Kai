"""Who may operate ``ft mgr``: anyone on the network may look, only this PC or a token holder may act.

The manager binds 0.0.0.0 so the console can be watched from another PC, and the same port starts
and stops the engine. Reads stay open; a write (start, stop, restart, profile edits, shutdown) is
accepted from this PC or with the token in ``X-FT-Token``. The token lives in the state dir, readable
by this user only, and the console shows it to a browser on this PC so it can be copied elsewhere.

"This PC" is the connection's address being loopback *and* the Host header naming loopback. The
second half keeps a page on another site that rebinds its name to 127.0.0.1 from counting as local.
A port forward that relays LAN connections to 127.0.0.1 (netsh portproxy to localhost) makes them
look local: point such a forward at the WSL address instead.

stdlib only: the daemon imports this, and it must never pull torch."""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from urllib.parse import urlsplit

TOKEN_FILE = "token"
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def load_or_create_token(state_dir: str) -> str:
    path = os.path.join(state_dir, TOKEN_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            value = fh.read().strip()
        if value:
            return value
    except FileNotFoundError:
        pass
    value = secrets.token_urlsafe(18)
    os.makedirs(state_dir, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    return value


def _loopback_address(host: str | None) -> bool:
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return (mapped or addr).is_loopback


def is_local(client_host: str | None, host_header: str | None) -> bool:
    if not _loopback_address(client_host):
        return False
    name = urlsplit("//" + (host_header or "")).hostname
    return bool(name) and (name in _LOOPBACK_NAMES or _loopback_address(name))


def token_matches(given: str | None, expected: str | None) -> bool:
    return bool(given) and bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


def may_write(client_host: str | None, host_header: str | None, given: str | None, expected: str | None) -> bool:
    return is_local(client_host, host_header) or token_matches(given, expected)
