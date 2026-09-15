"""--pp-send-ahead: how many residual streams the first pipeline rank may have in flight.

Two processes over gloo, as the transport runs in the server. The receiver sleeps before its
first receive, so the sender's behaviour is visible in wall time: at 1 it waits for the peer, at
N it hands over N chunks and only then waits. The payloads must still arrive in order and
unmixed -- the ring reuses a buffer only once its send completed.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest
import torch

from freetoken.distributed.info import PipelineInfo

SLEEP = 1.0  # the receiver's head start; the asserts leave a wide margin around it
ROWS = [3, 5, 4, 6]
WIDTH = 16


def _expect(step: int, rows: int) -> torch.Tensor:
    return torch.full((rows, WIDTH), float(step + 1), dtype=torch.bfloat16)


def _worker(rank: int, init_file: str, send_ahead: int, out_dir: str) -> None:
    import torch.distributed as dist

    from freetoken.distributed.pipeline import PipelineComm

    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=2)
    device = torch.device("cpu")  # same byte path, unpinned staging
    info = PipelineInfo(rank, 2, 0 if rank == 0 else 5, 5 if rank == 0 else 10, 10)
    comm = PipelineComm(info, dist.group.WORLD, device, send_ahead=send_ahead)
    comm.configure(hidden_width=WIDTH, hidden_dtype=torch.bfloat16)
    if rank == 0:
        marks = []
        t0 = time.perf_counter()
        for step, rows in enumerate(ROWS):
            comm.send_hidden(_expect(step, rows))
            marks.append(time.perf_counter() - t0)
        comm.drain_sends()
        marks.append(time.perf_counter() - t0)
        with open(os.path.join(out_dir, "marks"), "w") as f:
            f.write(" ".join(f"{m:.3f}" for m in marks))
    else:
        time.sleep(SLEEP)
        for step, rows in enumerate(ROWS):
            got = comm.recv_hidden(rows)
            assert torch.equal(got, _expect(step, rows)), (step, got)
    dist.destroy_process_group()


def _run(send_ahead: int) -> list[float]:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(_worker, args=(init_file, send_ahead, d), nprocs=2, join=True)
        with open(os.path.join(d, "marks")) as f:
            return [float(x) for x in f.read().split()]


def test_one_send_at_a_time_waits_for_the_peer():
    marks = _run(1)
    assert marks[0] >= SLEEP * 0.8, marks  # the first send waits out the receiver's head start
    assert marks[-1] >= SLEEP * 0.8


def test_send_ahead_hands_over_a_window_before_waiting():
    """Three slots: the first three sends go out while the receiver still sleeps; the fourth
    waits for the oldest to land."""
    marks = _run(3)
    assert marks[2] < SLEEP * 0.6, marks  # three chunks handed over during the head start
    assert marks[3] >= SLEEP * 0.6, marks  # the fourth needed a slot back
    assert marks[-1] >= marks[3]


def test_a_window_wider_than_the_traffic_never_waits():
    marks = _run(8)
    assert marks[len(ROWS) - 1] < SLEEP * 0.6, marks
    assert marks[-1] >= SLEEP * 0.6, marks  # draining still waits for the receiver


@pytest.mark.parametrize("send_ahead, keys", [(1, 1), (4, 4)])
def test_the_ring_has_one_staging_buffer_per_slot(send_ahead, keys):
    from freetoken.distributed.pipeline import PipelineComm

    comm = PipelineComm(PipelineInfo(0, 2, 0, 5, 10), None, torch.device("cpu"), send_ahead=send_ahead)
    assert comm.send_ahead == send_ahead and len(comm._free_keys) == keys
    assert PipelineComm(PipelineInfo(0, 2, 0, 5, 10), None, torch.device("cpu"), send_ahead=0).send_ahead == 1


def test_the_flag_needs_the_pipeline_engine(monkeypatch):
    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe"}

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _p: _Config())
    base = ["--model", "/models/anon"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--pp-send-ahead", "0"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--pp-send-ahead", "3"])  # needs --pp-size
    args, _ = parse_args(base + ["--pp-size", "2", "--pp-send-ahead", "3"])
    assert args.pp_send_ahead == 3
    assert parse_args(base)[0].pp_send_ahead == 1
