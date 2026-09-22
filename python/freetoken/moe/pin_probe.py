"""What this host will actually page-lock -- measured, not guessed.

WSL's WDDM-backed CUDA caps ``cudaHostRegister`` far below plain Linux, and nothing derives
the cap. NVIDIA's own answer is that there is no formula for it on any supported OS, that
the limit is Windows' to set and the driver does not control it, and that WSL2 lands lower
still than the native Windows figure.

Measured on two machines (one of them at two ``.wslconfig`` sizes) and reported on a third, no
fraction of anything describes them. On Windows 11 nothing refused at all: 5.50 GiB held in a
7.75 GiB guest (71%), 12 GiB in a 23.5 GiB guest on the same machine (51%, stopped by the
*host's* free RAM), and 93.50 GiB in a 110 GiB guest on the other (85% of the guest, and 78% of
that host's 128 GB). On Windows 10 a 24 GiB guest stops at 1 GiB (4%) and the boot dies mid-load
(reported, not measured here). No fraction of the guest's RAM describes both ends, and neither
does half of the host's -- which a guest cannot read anyway.

So one figure is remembered, and only one: **what a refusal measured**. Either ``ft doctor pin``
reached it, or a boot died in ``cudaHostRegister`` and its registered total is the cap itself, so
the next boot plans against the truth instead of dying in the same place.

A ladder that merely ran out of guest or host RAM -- the usual outcome on Windows 11 -- measures
nothing about this host and is not recorded. It is tempting to treat "held 93.50 GiB once" as a
budget, and that is exactly what must not happen: on the 3060 host the ladder did hold 93.50 GiB,
and starting Flash-Next on one card against a 90 GiB budget then pinned all 63.46 GiB of banks,
reached "Scheduler is idle", served no token in 180 s, and took the guest down with it. Pinned
pages cannot be reclaimed, so what one process holds for a moment with nothing else running is
not what a server can commit and still work.

Where nothing has refused, the estimate stands. It is a guess at the wrong quantity -- it does
not predict the driver -- but on every host measured it has kept the server inside what the host
could spare, which is the quantity that matters.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import NamedTuple

GiB = 1 << 30
STEP_BYTES = 256 << 20  # the ladder's rung, and the granularity a remembered cap is rounded to
_ESTIMATE_FRACTION = 0.4  # of MemTotal; only for a host with no measurement of its own


def is_pin_capped(release: str | None = None) -> bool:
    """Whether the platform caps page-locked host memory at all -- WSL does, plain Linux does not."""
    if not hasattr(os, "uname"):
        return False
    return "microsoft" in (os.uname().release if release is None else release).lower()


def _meminfo_bytes(field: str, proc: str = "/proc") -> int:
    try:
        with open(os.path.join(proc, "meminfo")) as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def estimate(proc: str = "/proc") -> int:
    """The last-resort guess for a host that has never measured: a fraction of the guest's RAM.

    Known to be wrong (see the module docstring); kept so an unmeasured host still plans
    *something* rather than pinning until the driver refuses mid-load."""
    return int(_meminfo_bytes("MemTotal", proc) * _ESTIMATE_FRACTION)


# ------------------------------------------------------------------ the remembered figure

def cache_path() -> str:
    if env := os.environ.get("FREETOKEN_PIN_CAP_FILE"):
        return env
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(root, "freetoken", "pin_cap.json")


def host_key(proc: str = "/proc") -> str:
    """What a remembered figure is valid for.

    The kernel build, and the guest's RAM: the cap does not track MemTotal, but a guest
    smaller than the cap makes the *usable* figure smaller, so a figure measured under one
    ``.wslconfig`` ``memory=`` must not be reused under another."""
    release = os.uname().release if hasattr(os, "uname") else "unknown"
    return f"{release}|memtotal={_meminfo_bytes('MemTotal', proc)}"


class Record(NamedTuple):
    """A cap this host refused at, and how that was learned."""

    cap_bytes: int
    how: str


def remembered(proc: str = "/proc") -> Record | None:
    """The cap on record for this host, or None when nothing has been refused here."""
    try:
        with open(cache_path()) as fh:
            doc = json.load(fh)
        entry = doc[host_key(proc)]
        return Record(int(entry["cap_bytes"]), str(entry.get("how", "unknown")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remember(cap_bytes: int, *, how: str, proc: str = "/proc") -> bool:
    """Record a cap a refusal measured. Returns False when it could not be written.

    Only a refusal belongs here. Last write wins: a boot failure can record less than a
    ``ft doctor pin`` run did, because the cap is shared across processes and something else may
    have been holding part of it -- that lands on the conservative side, which is the right side
    to be wrong on."""
    path = cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as fh:
                doc = json.load(fh)
            if not isinstance(doc, dict):
                doc = {}
        except (OSError, ValueError):
            doc = {}
        doc[host_key(proc)] = {
            "cap_bytes": int(cap_bytes),
            "how": how,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".pin_cap-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
            os.replace(tmp, path)  # atomic: two ranks may record at once
        except BaseException:
            os.unlink(tmp)
            raise
        return True
    except OSError:
        return False


# ------------------------------------------------------------------ the budget consumers use

def budget(reserved: int = 0, proc: str = "/proc") -> int | None:
    """Bytes this process can still safely page-lock, or None where the platform does not cap it.

    ``FREETOKEN_PIN_BUDGET_GB`` overrides anywhere. ``reserved`` subtracts host bytes already
    pinned outside the expert banks (qwen4_exp's PLE table)."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        cap = int(float(env) * GiB)
    elif not is_pin_capped():
        return None
    else:
        known = remembered(proc)
        cap = known.cap_bytes if known else estimate(proc)
    return max(0, cap - reserved)


def source(proc: str = "/proc") -> str:
    """Where :func:`budget` got its figure, for a log line or a doctor report."""
    if os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        return "FREETOKEN_PIN_BUDGET_GB"
    if not is_pin_capped():
        return "uncapped platform"
    known = remembered(proc)
    if known is None:
        return f"estimated ({_ESTIMATE_FRACTION:.0%} of MemTotal, nothing has been refused here)"
    return f"measured ({known.how})"


# ------------------------------------------------------------------ measuring one

@dataclass
class Measurement:
    """What a ladder run learned. ``refused`` is the only outcome that pins down the cap."""

    locked_bytes: int
    refused: bool  # CUDA said no -- locked_bytes is this host's cap
    ceiling_hit: bool  # stopped where the caller said to stop; the cap is higher
    ram_limited: bool  # stopped to stay inside the guest's own free RAM; the cap may be higher
    error: str | None = None

    def summary(self) -> str:
        head = f"{self.locked_bytes / GiB:.2f} GiB page-locked"
        if self.refused:
            return f"{head}; the next rung was refused, so that is this host's cap ({self.error})"
        if self.ram_limited:
            return f"{head}; stopped by the guest's own free RAM before CUDA refused, so the cap is at least this"
        return f"{head}; stopped at the requested ceiling, so the cap is at least this"


def measure(ceiling_bytes: int, *, step: int = STEP_BYTES, proc: str = "/proc", on_step=None) -> Measurement:
    """Page-lock ``step`` bytes at a time up to ``ceiling_bytes`` and report where it stopped.

    Holds every rung until it returns, so this belongs in a short-lived process (``ft doctor
    pin``) and never in a server that is about to need the same budget. It also stops short of
    the guest's own free RAM: past that the kernel's OOM killer arrives before the driver's
    refusal does, and a killed process measures nothing.

    torch's pinned caching allocator does not return a freed rung to the driver, so a second
    ladder in the same process answers for that cache and not for the host -- measure once
    per process."""
    import torch

    # leave the guest room to keep running; the rungs are unswappable once locked
    headroom = max(step, _meminfo_bytes("MemAvailable", proc) // 8)
    ram_ceiling = max(0, _meminfo_bytes("MemAvailable", proc) - headroom)
    held, locked, inited = [], 0, False
    while locked + step <= ceiling_bytes:
        if locked + step > ram_ceiling:
            return Measurement(locked, refused=False, ceiling_hit=False, ram_limited=True)
        if not inited:  # measuring needs a GPU; declining to measure does not
            torch.cuda.init()
            inited = True
        try:
            held.append(torch.empty(step, dtype=torch.uint8).pin_memory())
        except Exception as exc:  # noqa: BLE001 -- any refusal is the answer
            return Measurement(locked, refused=True, ceiling_hit=False, ram_limited=False,
                               error=f"{type(exc).__name__}: {exc}")
        locked += step
        if on_step:
            on_step(locked)
    return Measurement(locked, refused=False, ceiling_hit=True, ram_limited=False)
