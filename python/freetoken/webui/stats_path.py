"""Where each rank writes its web console snapshot; shared by the engine and the frontend, torch-free."""

from __future__ import annotations

import os
import tempfile


def stats_dir(port: int | None) -> str | None:
    if not port:
        return None
    return os.path.join(tempfile.gettempdir(), f"freetoken-webstats-{port}")
