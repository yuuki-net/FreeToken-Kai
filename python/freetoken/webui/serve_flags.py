"""The ``ft serve`` flags as JSON, for the web console's profile editor.

Run as a child process (``python -m freetoken.webui.serve_flags``): building the parser imports
``freetoken.server.args``, which pulls torch, and the daemon must never import that itself. The
parser is created inside ``parse_args``, so it is captured by intercepting its own parse call.
``load`` (torch-free) caches the result per args.py mtime in the daemon's state dir."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def dump() -> list[dict]:
    real = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = capture
    try:
        from freetoken.server.args import parse_args

        parse_args(["--model", "x"])
    except _Captured as exc:
        parser = exc.parser
    finally:
        argparse.ArgumentParser.parse_args = real

    groups = {id(a): g.title for g in parser._action_groups for a in g._group_actions}
    out = []
    for a in parser._actions:
        if not a.option_strings or a.help == argparse.SUPPRESS or isinstance(a, argparse._HelpAction):
            continue
        if type(a).__name__ == "_DeprecatedAlias":
            continue
        kind = "bool" if a.nargs == 0 else "choice" if a.choices else "value"
        default = a.default if a.default is not argparse.SUPPRESS else None
        out.append({
            "flag": max(a.option_strings, key=len),
            "aliases": [s for s in a.option_strings if s != max(a.option_strings, key=len)],
            "kind": kind,
            "negates": isinstance(a, argparse._StoreFalseAction),
            "choices": [str(c) for c in a.choices] if a.choices else None,
            "default": default if isinstance(default, (str, int, float, bool)) or default is None else str(default),
            "multiple": a.nargs in ("+", "*") or (isinstance(a.nargs, int) and a.nargs > 1),
            "required": bool(a.required),
            "metavar": a.metavar if isinstance(a.metavar, str) else None,
            "help": " ".join((a.help or "").split()),
            "group": groups.get(id(a)),
        })
    return out


def load(cache_dir: str, python: str = sys.executable, timeout: float = 180.0) -> list[dict]:
    """The flag list, rebuilt in a child process only when args.py changed. Torch-free."""
    import importlib.util

    spec = importlib.util.find_spec("freetoken")
    root = os.path.dirname(spec.origin) if spec and spec.origin else ""
    try:
        stamp = str(int(os.stat(os.path.join(root, "server", "args.py")).st_mtime))
    except OSError:
        stamp = "unknown"
    path = os.path.join(cache_dir, f"serve-flags-{stamp}.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    proc = subprocess.run(
        [python, "-m", "freetoken.webui.serve_flags"], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "failed")
    flags = json.loads(proc.stdout.strip().splitlines()[-1])
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(flags, fh)
    os.replace(tmp, path)
    return flags


if __name__ == "__main__":
    flags = dump()
    sys.stdout.write("\n" + json.dumps(flags, ensure_ascii=False) + "\n")
