"""Prefill pipeline overlap: which batches carry no tokens back, and the async message-count
channel that replaced the per-step broadcast between the scheduler ranks."""

from __future__ import annotations

import os
import tempfile

import torch

from freetoken.core import Batch, Req, SamplingParams


def _mk(cls, n: int):
    return cls(
        input_ids=torch.arange(n, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def test_batch_default_carries_tokens():
    b = Batch(reqs=[_mk(Req, 4)], phase="prefill")
    assert b.pp_no_tokens is False


def test_only_chunk_only_prefill_batches_skip_tokens():
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.scheduler import _no_tokens_needed

    chunk = _mk(ChunkedReq, 8)
    final = _mk(Req, 8)
    assert _no_tokens_needed(Batch(reqs=[chunk], phase="prefill"))
    assert not _no_tokens_needed(Batch(reqs=[final], phase="prefill"))         # final chunk: sampled
    assert not _no_tokens_needed(Batch(reqs=[chunk, final], phase="prefill"))  # mixed: sampled
    assert not _no_tokens_needed(Batch(reqs=[chunk], phase="decode"))
    verify = Batch(reqs=[chunk], phase="prefill")
    verify.spec_verify = True
    assert not _no_tokens_needed(verify)
    assert not _no_tokens_needed(Batch(reqs=[], phase="prefill"))


def _count_worker(rank: int, init_file: str) -> None:
    import torch.distributed as dist

    from freetoken.scheduler.io import SchedulerIOMixin

    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=2)
    io = SchedulerIOMixin.__new__(SchedulerIOMixin)
    io.tp_cpu_group = dist.group.WORLD
    io._pending_count_sends = []
    counts = [0, 3, 0, 1, 7]
    if rank == 0:
        # rank 0 runs ahead: every count is published before rank 1 asks for any of them
        for c in counts:
            io._publish_msg_count(c)
        assert len(io._pending_count_sends) <= len(counts)
        for _, w in io._pending_count_sends:
            w.wait()
    else:
        got = [io._await_msg_count() for _ in counts]
        assert got == counts, got
    dist.barrier()
    dist.destroy_process_group()


def test_msg_count_channel_lets_rank0_run_ahead():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(_count_worker, args=(init_file,), nprocs=2, join=True)
