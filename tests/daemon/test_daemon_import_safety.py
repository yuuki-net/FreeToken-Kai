"""Guard the daemon's single most important rule: it must never import torch / CUDA /
flashinfer / sgl_kernel / transformers, directly or transitively. A future "small" refactor that
grabs a helper from ``freetoken.server`` or ``freetoken.utils`` would silently pull torch and only
fail on a CUDA-less box at startup. This test makes that a red test instead.

The check runs in a *fresh* interpreter (subprocess): a same-process test would diff sys.modules
against a baseline taken after other test modules already imported torch at collection time, so a
poisoned daemon submodule that imported torch would still "pass". The child starts clean, installs
the meta-path blocker, and only then imports the daemon."""

from __future__ import annotations

import os
import subprocess
import sys

FORBIDDEN = ("torch", "transformers", "flashinfer", "sgl_kernel", "triton")

DAEMON_MODULES = [
    "freetoken.daemon",
    "freetoken.daemon.version",
    "freetoken.daemon.accounting",
    "freetoken.daemon.logfmt",
    "freetoken.daemon.logring",
    "freetoken.daemon.osproc",
    "freetoken.daemon.pidfile",
    "freetoken.daemon.metrics",
    "freetoken.daemon.proxy",
    "freetoken.daemon.tailer",
    "freetoken.daemon.serve_manager",
    "freetoken.daemon.checkpoint",
    "freetoken.daemon.app",
    "freetoken.daemon.client",
    "freetoken.daemon.server",
    "freetoken.daemon.profiles",
    "freetoken.webui",
    "freetoken.webui.stats_path",
    "freetoken.webui.models",
    "freetoken.webui.serve_flags",
    "freetoken.webui.recommend",
    "freetoken.webui.hostmem",
]

# Runs in the child interpreter. FORBIDDEN / DAEMON_MODULES are prepended as literals so the
# blocker is armed *before* the first daemon import — nonzero exit + a clear message on any
# forbidden import attempt or any forbidden module that lands in sys.modules.
_CHILD_BODY = '''
import sys


class _Blocker:
    """Meta-path finder that raises if anything imports a forbidden top-level package."""

    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in FORBIDDEN:
            raise ImportError("forbidden import for the daemon: " + name)
        return None


sys.meta_path.insert(0, _Blocker())

import importlib
from concurrent.futures import ThreadPoolExecutor

try:
    for _mod in DAEMON_MODULES:
        importlib.import_module(_mod)
    # Also exercise app assembly, where a stray pydantic/route import could sneak torch in.
    from freetoken.daemon.app import build_app

    class _Mgr:
        def status(self):
            return {"running": False}

        def current_pid(self):
            return None

    build_app(
        manager=_Mgr(),
        ring=importlib.import_module("freetoken.daemon.logring").LogRing(),
        probe=None,
        footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1),
        proxy_pool=ThreadPoolExecutor(1),
    )
except Exception as exc:  # a forbidden import propagates out of import_module as ImportError
    print("daemon import-safety violated: " + repr(exc), file=sys.stderr)
    sys.exit(1)

leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
if leaked:
    print("forbidden modules landed in sys.modules: " + repr(leaked), file=sys.stderr)
    sys.exit(2)

print("daemon import-safety OK")
'''

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_daemon_package_imports_without_torch():
    pkg = os.path.join(_REPO_ROOT, "python")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg + (os.pathsep + existing if existing else "")

    script = f"FORBIDDEN = {FORBIDDEN!r}\nDAEMON_MODULES = {DAEMON_MODULES!r}\n" + _CHILD_BODY
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"daemon import-safety child exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
