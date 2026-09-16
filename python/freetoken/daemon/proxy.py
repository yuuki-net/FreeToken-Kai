"""Aggregate the per-serve control API (``server/control_api.py``: ``/health``,
``/v1/stats``, ``/v1/admin/prepare-stop``) up to the daemon. The per-serve ``/health`` answers
"how is the model doing?" and dies with the serve; the daemon re-exposes it under
``/engine/health`` alongside
its own reachability, so a client has one control endpoint that OUTLIVES any single serve.

Blocking ``urllib`` (stdlib — no new dependency) run from the daemon's dedicated proxy executor
(kept off the lifecycle executor so a slow/loading serve can never starve
``/engine/stop``). Single-flight + short TTL cache: N concurrent pollers cost one upstream probe.
Serve docs are snake_case; they are transformed to the daemon's camelCase contract."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .accounting import AccountingPrepareError, PrepareStopUnavailable

_SNAKE_RE = re.compile(r"_([a-z0-9])")


def _camel_key(key: str) -> str:
    return _SNAKE_RE.sub(lambda m: m.group(1).upper(), key)


def to_camel(obj: Any) -> Any:
    """Recursively camelCase dict keys (uptime_s→uptimeS, done_bytes→doneBytes, …) so the daemon
    emits one casing convention everywhere."""
    if isinstance(obj, dict):
        return {_camel_key(k): to_camel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_camel(v) for v in obj]
    return obj


def local_listener(port: int) -> bool | None:
    """Whether something LISTENs on ``port`` here, from the kernel's socket tables; None when they
    cannot be read (not Linux). Under WSL's mirrored networking a connect to a closed loopback
    port gets no reset and waits out the whole timeout, and the probe holds its lock meanwhile."""
    readable = False
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as fh:
                rows = fh.read().splitlines()[1:]
        except OSError:
            continue
        readable = True
        for row in rows:
            cols = row.split()
            if len(cols) > 3 and cols[3] == "0A":  # LISTEN
                try:
                    if int(cols[1].rsplit(":", 1)[1], 16) == port:
                        return True
                except ValueError:
                    continue
    return False if readable else None


class ServeProbe:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        ttl_s: float = 0.25,
        timeout_s: float = 1.5,
        prepare_timeout_s: float = 15.0,
        now: Callable[[], float] = time.monotonic,
        opener: Callable[[str, float], dict] | None = None,
        prepare_opener: Callable[[str, float], dict] | None = None,
        listening: Callable[[int], bool | None] | None = None,
    ) -> None:
        self._host = host
        self._ttl = ttl_s
        self._timeout = timeout_s
        self._prepare_timeout = prepare_timeout_s
        self._now = now
        self._opener = opener or self._urlopen
        self._prepare_opener = prepare_opener or self._urlopen_prepare
        # only the real opener talks to sockets; an injected one (tests) is asked as it is
        if listening is None and opener is None and host in ("127.0.0.1", "localhost", "::1"):
            listening = local_listener
        self._listening = listening
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, int], tuple[float, dict]] = {}

    def health(self, port: int) -> dict:
        return self._cached("health", "/health", port)

    def stats(self, port: int) -> dict:
        return self._cached("stats", "/v1/stats", port)

    def get(self, path: str, port: int) -> dict:
        """Any other read-only serve document (the web console's requests / cache / experts)."""
        return self._cached(path, path, port)

    def fresh_stats(self, port: int) -> dict:
        """Fetch uncached stats for a legacy stop receipt."""
        return self._fetch("/v1/stats", port)

    def prepare_stop(self, port: int) -> dict:
        """Quiesce the engine and return its sealed final accounting snapshot."""
        url = f"http://{self._host}:{port}/v1/admin/prepare-stop"
        try:
            doc = self._prepare_opener(url, self._prepare_timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                raise PrepareStopUnavailable("legacy-engine") from exc
            raise AccountingPrepareError(f"prepare-stop returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise AccountingPrepareError(f"prepare-stop request failed: {exc}") from exc
        if not isinstance(doc, dict):
            raise AccountingPrepareError("prepare-stop returned a non-object response")
        return doc

    def _cached(self, kind: str, path: str, port: int) -> dict:
        key = (kind, port)
        # Hold the lock across the fetch so concurrent pollers collapse to a single upstream call
        # (true single-flight); the short timeout + TTL keep the critical section cheap.
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and (self._now() - hit[0]) < self._ttl:
                return hit[1]
            val = self._fetch(path, port)
            self._cache[key] = (self._now(), val)
            return val

    def _fetch(self, path: str, port: int) -> dict:
        if self._listening is not None and self._listening(port) is False:
            return {"reachable": False, "status": "unreachable"}
        url = f"http://{self._host}:{port}{path}"
        try:
            doc = self._opener(url, self._timeout)
        except urllib.error.HTTPError as exc:
            return {"reachable": True, "status": "error", "httpStatus": exc.code}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return {"reachable": False, "status": "unreachable"}
        result = to_camel(doc) if isinstance(doc, dict) else {"value": doc}
        result["reachable"] = True
        return result

    @staticmethod
    def _urlopen(url: str, timeout: float) -> dict:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _urlopen_prepare(url: str, timeout: float) -> dict:
        req = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
