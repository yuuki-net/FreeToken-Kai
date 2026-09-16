"""Saved launch profiles for the web console: named ``{model, port, args}`` in one JSON file.

The console never lets a user type a bare ``ft serve`` line at a restart; they pick a profile, so the
required flags a machine needs (the ones ``serve-3060.sh`` carries) cannot be dropped by accident.
stdlib only — the daemon must stay torch-free."""

from __future__ import annotations

import json
import os
import re
import threading
import time

_NAME = re.compile(r"^[^\x00-\x1f/\\]{1,80}$")


class ProfileStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        try:
            with open(self.path) as fh:
                doc = json.load(fh)
        except (FileNotFoundError, ValueError):
            return {}
        return doc.get("profiles", {}) if isinstance(doc, dict) else {}

    def _save(self, profiles: dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"profiles": profiles}, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def list(self) -> list[dict]:
        with self._lock:
            items = self._load()
        return [{"name": k, **v} for k, v in sorted(items.items(), key=lambda kv: kv[1].get("updated", 0), reverse=True)]

    def put(self, name: str, model: str, port: int | None, args: list[str]) -> dict:
        if not _NAME.match(name or ""):
            raise ValueError("profile name must be 1-80 characters without control characters or slashes")
        from freetoken.webui.models import resolve_model

        model = resolve_model(model)  # ft serve gets argv, not a shell: no ~, and no bare folder names
        if port is not None and not (0 < int(port) < 65536):
            raise ValueError("port out of range")
        if not all(isinstance(a, str) for a in args):
            raise ValueError("args must be strings")
        args = [os.path.expanduser(a) if a.startswith("~/") else a for a in args]
        entry = {"model": model, "port": port, "args": args, "updated": time.time()}
        with self._lock:
            items = self._load()
            items[name] = entry
            self._save(items)
        return {"name": name, **entry}

    def delete(self, name: str) -> bool:
        with self._lock:
            items = self._load()
            if name not in items:
                return False
            del items[name]
            self._save(items)
        return True
