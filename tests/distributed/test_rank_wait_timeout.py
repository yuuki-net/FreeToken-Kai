"""--pp-size: a long step on one rank does not end the other rank's recv.

Every serving exchange between the pipeline ranks is a gloo send/recv on the group the engine
creates, and gloo ends a recv at the group's timeout. That group used to carry
--distributed-timeout (60 s), so a step on the first rank that ran past a minute took the server
down ("Timed out waiting 60000ms for recv operation", seen once by a fork running Kai, on the first
request after a start; why that step was so long is not known). The group now carries
FREETOKEN_RANK_WAIT_TIMEOUT_SECONDS. Two CPU processes, a 2 s --distributed-timeout, a sender 5 s
late: the old timeout loses the recv, the new one keeps it; and a peer that dies is still seen
at once, so the long timeout hides no crash.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LATE_S = 5.0
DIST_TIMEOUT_S = 2.0


def _worker(rank: int, init_file: str, result_dir: str, group_timeout: float, peer_dies: bool) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=group_timeout),
    )
    outcome, t0 = "ok", time.monotonic()
    try:
        if rank == 0:
            if peer_dies:
                time.sleep(1.0)
                os._exit(0)  # a rank that crashed: its sockets close, it never sends
            time.sleep(LATE_S)  # a long first step
            try:
                dist.send(torch.ones(4), dst=1, tag=3)
            except RuntimeError:
                pass  # the receiver gave up first (the old timeout): its side is what is checked
        else:
            buf = torch.zeros(4)
            try:
                dist.recv(buf, src=0, tag=3)
            except RuntimeError as exc:
                outcome = "timeout" if "imed out" in str(exc) else "closed" if "losed" in str(exc) else f"error: {exc}"
            with open(os.path.join(result_dir, "rank1"), "w") as f:
                f.write(f"{outcome} {time.monotonic() - t0:.1f}")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 -- a group whose peer is gone may not tear down cleanly
            pass


def _run(group_timeout: float, peer_dies: bool = False) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        ctx = mp.start_processes(_worker, args=(init_file, d, group_timeout, peer_dies), nprocs=2,
                                 join=False, start_method="spawn")
        while not ctx.join(timeout=60):
            pass
        path = os.path.join(d, "rank1")
        return open(path).read().split() if os.path.exists(path) else ["missing"]


def _timeout(env_value: float | None) -> float:
    # ENV reads FREETOKEN_RANK_WAIT_TIMEOUT_SECONDS once at import: set the parsed value instead
    from freetoken.engine.engine import _pp_group_timeout
    from freetoken.env import ENV

    var = ENV.RANK_WAIT_TIMEOUT_SECONDS
    old = var.value
    try:
        if env_value is not None:
            var.value = env_value
        return _pp_group_timeout(SimpleNamespace(distributed_timeout=DIST_TIMEOUT_S))
    finally:
        var.value = old


def test_the_pipeline_group_outlasts_a_long_step():
    assert _timeout(None) == 86400.0
    assert _run(_timeout(None))[0] == "ok"


def test_the_old_timeout_loses_the_recv():
    # the failure this is for, so the test above cannot pass for the wrong reason
    assert _run(DIST_TIMEOUT_S)[0] == "timeout"


def test_a_dead_peer_is_still_seen_at_once():
    outcome, took = _run(_timeout(None), peer_dies=True)
    assert outcome == "closed" and float(took) < 10.0


def test_never_below_distributed_timeout():
    assert _timeout(1.0) == DIST_TIMEOUT_S
    assert _timeout(30.0) == 30.0
