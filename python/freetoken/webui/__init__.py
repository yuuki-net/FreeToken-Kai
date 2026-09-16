"""Browser console for a serve or a daemon: ``/ui/`` (dashboard) and ``/ui/tuning.html``.

Static files plus one tiny ``/ui/env.json`` telling the page which process served it, so the same
page runs read-only against ``ft serve`` and with lifecycle controls against ``ft daemon``.
stdlib + starlette only: the daemon imports this, and it must never pull torch."""

from __future__ import annotations

import os

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def register_webui(app, mode: str, version: str | None = None) -> None:
    from starlette.responses import JSONResponse, RedirectResponse
    from starlette.staticfiles import StaticFiles

    @app.get("/ui", include_in_schema=False)
    async def _ui_root():
        return RedirectResponse(url="/ui/")

    @app.get("/ui/env.json", include_in_schema=False)
    async def _ui_env():
        return JSONResponse({"mode": mode, "version": version}, headers={"Cache-Control": "no-store"})

    class _Revalidated(StaticFiles):
        # the console changes with the checkout; without this a browser keeps serving the old script
        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache"
            return response

    # after the env route: a mount at /ui would otherwise swallow /ui/env.json
    app.mount("/ui", _Revalidated(directory=STATIC_DIR, html=True), name="webui")
