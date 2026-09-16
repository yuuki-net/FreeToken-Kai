"""``ft daemon`` — the FreeToken engine supervisor (persistent, torch-free control plane).

This package MUST NOT import torch / CUDA / flashinfer / sgl_kernel / any model or kernel code,
directly or transitively. That includes ``freetoken.server.*`` (its ``__init__`` pulls
torch) and ``freetoken.utils.*`` (its ``__init__`` pulls transformers). Keep this module's imports
to ``main`` only, which itself imports lazily.
"""

from __future__ import annotations

import sys


def main(argv=None, *, prog: str = "ft daemon", console: bool = False) -> int:
    """Single entry for ``ft daemon`` and, with ``console=True``, for kai's ``ft mgr``. A leading
    client verb (``ft daemon status`` …) controls a running supervisor; anything else (bare, or
    server flags like ``--host``/``--port``, which is what the systemd unit uses) runs the server."""
    from .client import CLIENT_VERBS

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in CLIENT_VERBS:
        from .client import main as _client_main

        return _client_main(args, prog=prog, console=console)

    from .server import main as _server_main

    return _server_main(args, prog=prog, console=console)


__all__ = ["main"]
