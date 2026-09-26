"""Run `ft` with page-locking REFUSED past a cap this tool invents.

**This lies to CUDA. It measures nothing.** The driver here is not asked what it allows; the
bindings are wrapped so that they start answering "no" once a running total passes
FT_FAKE_PIN_CAP_GIB (default 1). That is the opposite of `ft doctor pin`, which asks the host and
records the answer. Never read a number out of a run under this tool as a property of the
machine, and never report one: a host that registers 90 GiB happily will refuse at 1 GiB here.

What it is for: a host whose driver does not refuse cannot otherwise reach the code that runs
when one does -- issue #2's host (Windows 10 + WSL2) refuses past about 1 GiB, and the residency
and registration split (moe/mapped_bank.py) has to be checked against a refusal, not only against
this engine's own page-lock budget. Those two are different paths and a change to registration
wants both: FREETOKEN_PIN_BUDGET_GB makes FreeToken stop before it asks, this makes the answer
come back "no" after it did.

Two ways in are counted against ONE total, as a real capped host counts them:

* ``torch.cuda.cudart().cudaHostRegister`` -- what mapped_bank.py registers the resident rows
  through (and ``cudaHostUnregister`` gives the bytes back, which a give-back has to see);
* ``freetoken.kernel.pinned.alloc_pinned_tensor`` -- cudaHostAlloc, what the server page-locks
  AFTER the banks: the parallel prefill reader, the staging buffers, the CPU executor's I/O.

The first version wrapped only the first, so a registration that spent the whole budget looked
fine: everything pinned afterwards sailed through, and the scheduler death on the first long
prompt (FreeToken-Kai 43c) never showed. Check with a long prompt, too -- a short one never
touches the staging buffers.

The C++ extension's host_register() -- the pinned / PinPipeline path that the non-mapped banks
use -- goes straight to the driver, so this cannot reproduce a refusal there.

    FT_FAKE_PIN_CAP_GIB=1 python tools/fakecap.py serve --model ~/models/gpt-oss-20b \
        --moe-strategy hybrid --moe-bank-ram 4G ...

From the "Issue #2 返信内容" session (2026-09-23), widened the same day for 43c; the __main__
guard is load-bearing (the scheduler is spawned and re-imports this file, which is how the patch
reaches it).
"""
import math, os, sys, threading

CAP = int(float(os.environ.get("FT_FAKE_PIN_CAP_GIB", "1")) * 2**30)
_used = 0
_held: dict[int, int] = {}  # registered address -> bytes
_lock = threading.Lock()


def _take(nbytes: int, what: str) -> bool:
    global _used
    with _lock:
        if _used + nbytes > CAP:
            sys.stderr.write(
                f"[fakecap] refusing {what} {nbytes/2**20:.1f} MiB "
                f"after {_used/2**30:.3f} GiB of {CAP/2**30:.2f} GiB\n"
            )
            sys.stderr.flush()
            return False
        _used += nbytes
        return True


def _give(nbytes: int) -> None:
    global _used
    with _lock:
        _used -= nbytes


import torch
_real_cudart = torch.cuda.cudart


class _Proxy:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cudaHostRegister(self, addr, nbytes, flags):
        if not _take(nbytes, f"cudaHostRegister(flags=0x{flags:02x})"):
            return 2  # cudaErrorMemoryAllocation
        rc = int(self._inner.cudaHostRegister(addr, nbytes, flags))
        if rc != 0:
            _give(nbytes)
        else:
            with _lock:
                _held[addr] = nbytes
        return rc

    def cudaHostUnregister(self, addr):
        rc = self._inner.cudaHostUnregister(addr)
        with _lock:
            n = _held.pop(addr, 0)
        _give(n)
        return rc


torch.cuda.cudart = lambda: _Proxy(_real_cudart())

# Patched before anything imports the name: cpu_executor.py and bank_reader.py take it with
# `from freetoken.kernel.pinned import alloc_pinned_tensor`. What cudaHostAlloc hands out is
# never returned to this total -- nothing in a serving run frees it.
import freetoken.kernel.pinned as _pinned

_real_alloc = _pinned.alloc_pinned_tensor


def _alloc(*shape, dtype=torch.uint8, **kw):
    dims = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
    nbytes = math.prod(int(d) for d in dims) * torch.empty((), dtype=dtype).element_size()
    if not _take(nbytes, "cudaHostAlloc"):
        raise RuntimeError(f"[fakecap] cudaHostAlloc refused {nbytes/2**20:.1f} MiB (fake cap)")
    return _real_alloc(*shape, dtype=dtype, **kw)


_pinned.alloc_pinned_tensor = _alloc

# The scheduler runs in a spawned child, which re-imports this module as __main__ --
# that is how the patches above reach the process that builds the mapped banks. Only the
# CLI call itself must stay behind the guard.
if __name__ == "__main__":
    from freetoken.cli import main

    sys.argv[0] = "ft"
    sys.exit(main())
