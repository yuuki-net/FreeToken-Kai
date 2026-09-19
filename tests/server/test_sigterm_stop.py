"""A scheduler worker stops on SIGTERM the way it does on Ctrl+C (server/launch.py
``_stop_on_sigterm``): the orderly stop runs, and a stop that hangs still ends the process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

_CHILD = textwrap.dedent(
    """
    import sys, time
    from freetoken.server.launch import _stop_on_sigterm

    marker, grace, hang = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1"
    _stop_on_sigterm(grace)
    print("ready", flush=True)
    try:
        while True:
            time.sleep(0.05)          # run_forever
    except KeyboardInterrupt:
        open(marker, "w").write("stopped in order")   # scheduler.shutdown()
        if hang:
            time.sleep(60)            # a shutdown that never returns
    """
)


def _start(tmp_path, grace, hang):
    marker = tmp_path / "marker"
    p = subprocess.Popen([sys.executable, "-c", _CHILD, str(marker), str(grace), "1" if hang else "0"],
                         stdout=subprocess.PIPE, text=True, env={**os.environ})
    assert p.stdout.readline().strip() == "ready"
    return p, marker


def test_sigterm_runs_the_orderly_stop(tmp_path):
    p, marker = _start(tmp_path, grace=30.0, hang=False)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 0
    assert marker.read_text() == "stopped in order"


def test_a_stop_that_hangs_still_ends_the_process(tmp_path):
    p, marker = _start(tmp_path, grace=0.5, hang=True)
    t0 = time.monotonic()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 128 + signal.SIGTERM
    assert time.monotonic() - t0 < 10
    assert marker.read_text() == "stopped in order"


def test_a_second_sigterm_kills(tmp_path):
    p, marker = _start(tmp_path, grace=30.0, hang=True)
    p.send_signal(signal.SIGTERM)
    for _ in range(100):                      # wait for the handler to have run
        if marker.exists():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == -signal.SIGTERM
