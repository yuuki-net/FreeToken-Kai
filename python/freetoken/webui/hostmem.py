"""Host RAM for the web console. stdlib only: both the serve and the manager import it.

The bar the console draws is used / reclaimable cache / free, straight from /proc/meminfo, so the
three always add up to the total. The engine's own share is reported beside it, not inside it: its
PSS includes file-backed pages (the mapped checkpoint, the expert banks) that /proc/meminfo counts
as cache, so stacking the two would count those pages twice."""

from __future__ import annotations

import os

_FIELDS = {"MemTotal": "total", "MemAvailable": "available", "MemFree": "free", "Cached": "cached",
           "SwapTotal": "swap_total", "SwapFree": "swap_free", "Mlocked": "mlocked"}


def meminfo() -> dict:
    return _read("/proc/meminfo")


def _read(path: str) -> dict:
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                name = _FIELDS.get(key)
                if name:
                    out[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    return out


def host_memory(engine_root_pid: int | None = None) -> dict | None:
    """{total_bytes, used_bytes, reclaimable_bytes, free_bytes, swap_total_bytes, swap_used_bytes,
    mlocked_bytes, engine_bytes}. engine_bytes is the summed PSS of ``engine_root_pid`` and its
    children (None when no root is given)."""
    m = meminfo()
    if not m.get("total"):
        return None
    total, available, free = m["total"], m.get("available", 0), m.get("free", 0)
    doc = {
        "total_bytes": total,
        "used_bytes": max(0, total - available),
        "reclaimable_bytes": max(0, available - free),
        "free_bytes": free,
        "swap_total_bytes": m.get("swap_total", 0),
        "swap_used_bytes": max(0, m.get("swap_total", 0) - m.get("swap_free", 0)),
        "mlocked_bytes": m.get("mlocked", 0),
        "engine_bytes": None,
    }
    if engine_root_pid:
        from freetoken.daemon import osproc  # torch-free

        try:
            doc["engine_bytes"] = sum(osproc.read_pss_bytes(p) for p in osproc.tree_pids(engine_root_pid))
        except Exception:  # noqa: BLE001 -- a vanished child must not break the stats
            doc["engine_bytes"] = None
    return doc


def engine_root() -> int:
    """The serve's process tree is rooted at the API server itself (it spawns the engine ranks)."""
    return os.getpid()
