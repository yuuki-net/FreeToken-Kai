"""Run `ft` with cudaHostRegister refusing past a fake cap.

Reproduces issue #2's host (Windows 10 WSL2, driver refuses past 1 GiB) on a machine
whose driver does not refuse. Only the cudart binding mapped_bank.py uses is wrapped;
the C++ extension's host_register() (used by the pinned/PinPipeline path) is untouched.

From the "Issue #2 返信内容" session (2026-09-23).
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
