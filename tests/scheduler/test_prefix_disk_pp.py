"""--prefix-disk-cache under --pp-size (scheduler/prefix_disk.py): two simulated ranks.

Each rank has its own pools, tree, PrefillManager and store directory, and runs in its own thread;
a "step" hands both ranks the notes rank 0 queued in the previous step (as the relay does) and
then runs one admission pass on each. The property that matters is that the two ranks always make
the same batch at the same step -- restored or not -- whatever each one finds on its own disk.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from freetoken.scheduler.prefix_disk import PrefixDiskCache

from .test_prefix_disk_cache import Rig, _prefill_manager, _user_req


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1))


class _Agree:
    """``all_reduce(MIN)`` over two threads."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=20)
        self.vals = [None, None]
        self.calls = 0

    def for_rank(self, rank):
        def agree(ok: bool) -> bool:
            self.vals[rank] = bool(ok)
            self.barrier.wait()
            out = all(self.vals)
            self.barrier.wait()
            if rank == 0:
                self.calls += 1
            return out

        return agree


class PPRig:
    def __init__(self, tmp_path, *, have=(True, True)) -> None:
        self.ids = torch.arange(1, 65, dtype=torch.int32)
        self.agree = _Agree()
        self.ranks = []
        for r in range(2):
            root = tmp_path / f"r{r}"
            if have[r]:
                w = Rig(root)
                w.fill_random(10 + r)
                w.donate(self.ids)
                w.disk.persist_idle()
                w.disk.close(wait=True)
            rig = Rig(root)
            rig.disk = PrefixDiskCache(rig.store, rig.cm, rig.layout, min_tokens=8, min_gain=4,
                                       rank=r, world=2, agree=self.agree.for_rank(r))
            rig.pm = _prefill_manager(rig)
            self.ranks.append(rig)
        self.notes = []
        self.pool = ThreadPoolExecutor(max_workers=2)

    def submit(self, uid, ids):
        for rig in self.ranks:
            rig.pm.pending_list.append(_user_req(uid, ids.clone()))

    def step(self):
        """One scheduler loop on both ranks: apply last step's notes, then one admission pass."""
        notes, self.notes = self.notes, []

        def run(rig):
            for uid, kind, length in notes:
                req = next((q for q in rig.pm.pending_list if q.uid == uid), None)
                if req is not None:
                    rig.disk.apply_note(req, kind, length)
            batch = rig.pm.schedule_next_batch(1024)
            return None if batch is None else [(r.uid, r.cached_len) for r in batch.reqs]

        outs = list(self.pool.map(run, self.ranks))
        self.notes = self.ranks[0].disk.take_notes()
        assert not self.ranks[1].disk.take_notes(), "only rank 0 decides"
        assert outs[0] == outs[1], f"the ranks diverged: {outs}"
        return outs[0]

    def run_until_admitted(self, max_steps=200):
        for _ in range(max_steps):
            got = self.step()
            if got is not None:
                return got
            for rig in self.ranks:
                rig.disk.wait_for_load(0.05)
        raise AssertionError("never admitted")

    def close(self):
        # an admitted request holds pages and a working GDN slot, so the ranks are compared with
        # each other rather than with an empty pool: the same free pages and slots on both
        a, b = self.ranks
        assert len(a.cm.free_slots) == len(b.cm.free_slots)
        assert a.lin.num_free_slots == b.lin.num_free_slots
        for rig in self.ranks:
            rig.disk.close(wait=True)
        self.pool.shutdown()


def _prompt(ids):
    return torch.cat([ids, torch.tensor([3, 4], dtype=torch.int32)])


def test_both_ranks_restore_their_own_part_at_the_same_step(tmp_path):
    rig = PPRig(tmp_path)
    rig.submit(5, _prompt(rig.ids))
    got = rig.run_until_admitted()
    assert got == [(5, 64)]
    for r in rig.ranks:
        assert r.disk.stats["restored"] == 1 and r.pages_of(rig.ids)[0] == 64
    assert rig.agree.calls == 2, "one agreement on the read, one on the upload"
    rig.close()


def test_an_entry_missing_on_one_rank_prefills_on_both(tmp_path):
    rig = PPRig(tmp_path, have=(True, False))
    rig.submit(5, _prompt(rig.ids))
    assert rig.run_until_admitted() == [(5, 0)]
    for r in rig.ranks:
        assert r.disk.stats["restored"] == 0 and r.pages_of(rig.ids)[0] == 0
    rig.close()


def test_nothing_on_rank_0_means_no_wait_beyond_the_plan(tmp_path):
    rig = PPRig(tmp_path, have=(False, True))
    rig.submit(5, _prompt(rig.ids))
    assert rig.step() is None, "one step while rank 0's plan travels"
    assert rig.step() == [(5, 0)]
    assert rig.agree.calls == 0
    rig.close()


def test_short_prompts_are_not_planned(tmp_path):
    rig = PPRig(tmp_path)
    rig.submit(5, torch.arange(1, 5, dtype=torch.int32))
    assert rig.step() == [(5, 0)]
    assert rig.notes == []
    rig.close()


def test_rank_0_gives_up_on_a_slow_read_for_every_rank(tmp_path):
    rig = PPRig(tmp_path)
    r0 = rig.ranks[0]
    r0.disk.read_deadline_base_s = 0.05
    r0.disk.read_deadline_tokens_per_s = 1e9
    release = threading.Event()
    real = r0.store.load

    def slow(entry, expect_ids=None):
        release.wait(10)
        return real(entry, expect_ids=expect_ids)

    r0.store.load = slow
    rig.submit(5, _prompt(rig.ids))
    rig.step()                      # plan queued
    rig.step()                      # plan applied, reads start
    time.sleep(0.1)
    assert rig.run_until_admitted() == [(5, 0)]
    assert r0.disk.stats["load_abandoned"] == 1
    release.set()
    rig.close()


def test_an_upload_that_fails_on_one_rank_restores_on_neither(tmp_path):
    rig = PPRig(tmp_path)
    r1 = rig.ranks[1]

    def broken(pages, tensors, first):
        raise RuntimeError("simulated upload failure")

    r1.layout.write_pages = broken
    rig.submit(5, _prompt(rig.ids))
    assert rig.run_until_admitted() == [(5, 0)]
    for r in rig.ranks:
        assert r.disk.stats["restored"] == 0 and r.pages_of(rig.ids)[0] == 0
    rig.close()


def test_a_note_for_an_aborted_request_is_dropped(tmp_path):
    rig = PPRig(tmp_path)
    rig.submit(5, _prompt(rig.ids))
    rig.step()
    for r in rig.ranks:                       # aborted on every rank at the same step
        r.pm.pending_list.clear()
    assert rig.step() is None                 # the plan note finds no request
    rig.close()


def test_rank_0_relays_its_notes_behind_the_tokenizer_messages():
    """The relay publishes rank 0's notes after the tokenizer's frames, inside the same count,
    and hands rank 0 the same messages in the same order."""
    from freetoken.message import PrefixDiskBackendMsg, UserMsg  # noqa: F401
    from freetoken.scheduler.io import SchedulerIOMixin

    published, counts = [], []

    class Tok:
        def __init__(self, raws):
            self.raws = list(raws)

        def empty(self):
            return not self.raws

        def get_raw(self):
            return self.raws.pop(0)

        def decode(self, raw):
            import msgpack

            from freetoken.message import BaseBackendMsg

            return BaseBackendMsg.decoder(msgpack.unpackb(raw, raw=False))

    class Pub:
        def put_raw(self, raw):
            published.append(raw)

    import msgpack

    from freetoken.message import AbortBackendMsg

    io = object.__new__(SchedulerIOMixin)
    io._recv_from_tokenizer = Tok([msgpack.packb(AbortBackendMsg(uid=9).encoder(), use_bin_type=True)])
    io._send_into_ranks = Pub()
    io._publish_msg_count = counts.append
    io._rank0_notes = lambda: [PrefixDiskBackendMsg(uid=5, kind="plan", length=64)]
    msgs = io._recv_msg_multi_rank0(blocking=False)
    assert counts == [2] and len(published) == 2
    assert [type(m).__name__ for m in msgs] == ["AbortBackendMsg", "PrefixDiskBackendMsg"]
    assert (msgs[1].uid, msgs[1].kind, msgs[1].length) == (5, "plan", 64)
