"""Run `ft` with cudaHostRegister REFUSING past a cap this tool invents.

**This lies to CUDA. It measures nothing.** The driver here is not asked what it allows; the
binding is wrapped so that it starts answering "no" once a running total passes
FT_FAKE_PIN_CAP_GIB (default 1). That is the opposite of `ft doctor pin`, which asks the host and
records the answer. Never read a number out of a run under this tool as a property of the
machine, and never report one: a host that registers 90 GiB happily will refuse at 1 GiB here.

What it is for: a host whose driver does not refuse cannot otherwise reach the code that runs
when one does -- issue #2's host (Windows 10 + WSL2) refuses past about 1 GiB, and the residency
and registration split (moe/mapped_bank.py) has to be checked against a refusal, not only against
this engine's own page-lock budget. Those two are different paths and a change to registration
wants both: FREETOKEN_PIN_BUDGET_GB makes FreeToken stop before it asks, this makes the answer
come back "no" after it did.

Only the `torch.cuda.cudart()` binding that mapped_bank.py calls is wrapped. The C++ extension's
host_register() -- the pinned / PinPipeline path that the non-mapped banks use -- goes straight to
the driver, so this cannot reproduce a refusal there.

    FT_FAKE_PIN_CAP_GIB=1 python tools/fakecap.py serve --model ~/models/gpt-oss-20b \
        --moe-strategy hybrid --moe-bank-ram 4G ...

From the "Issue #2 返信内容" session (2026-09-23); the __main__ guard is load-bearing (the
scheduler is spawned and re-imports this file, which is how the patch reaches it).
"""
import os, sys, threading

CAP = int(float(os.environ.get("FT_FAKE_PIN_CAP_GIB", "1")) * 2**30)
_used = 0
_lock = threading.Lock()

import torch
_real_cudart = torch.cuda.cudart


class _Proxy:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cudaHostRegister(self, addr, nbytes, flags):
        global _used
        with _lock:
            if _used + nbytes > CAP:
                sys.stderr.write(
                    f"[fakecap] refusing {nbytes/2**20:.0f} MiB (flags=0x{flags:02x}) "
                    f"after {_used/2**30:.2f} GiB of {CAP/2**30:.2f} GiB\n"
                )
                sys.stderr.flush()
                return 2  # cudaErrorMemoryAllocation
            rc = int(self._inner.cudaHostRegister(addr, nbytes, flags))
            if rc == 0:
                _used += nbytes
            return rc


torch.cuda.cudart = lambda: _Proxy(_real_cudart())

# The scheduler runs in a spawned child, which re-imports this module as __main__ --
# that is how the patch above reaches the process that builds the mapped banks. Only the
# CLI call itself must stay behind the guard.
if __name__ == "__main__":
    from freetoken.cli import main

    sys.argv[0] = "ft"
    sys.exit(main())
