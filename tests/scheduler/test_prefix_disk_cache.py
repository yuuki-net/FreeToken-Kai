"""--prefix-disk-cache on the scheduler side (scheduler/prefix_disk.py, CacheManager.
insert_restored_prefix). CPU pools, a real HybridRadixCache, no engine.

The round trip that matters: a prefix the tree held is written while idle, the process
"restarts" (fresh pools, fresh tree, pages and slots numbered differently), and a prompt that
starts with it comes back at the same boundary with the same KV bytes and the same GDN state --
and the page / slot accounting afterwards is exactly what an in-memory donate leaves.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import tempfile
import threading
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.prefix_disk_store import PrefixDiskStore
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefix_disk import (
    PoolLayout,
    PrefixDiskCache,
    PrefixDiskUnsupported,
    build_fingerprint,
)

DEV = torch.device("cpu")
# The round trips also run on a GPU when one is visible: the gathers and scatters are the part
# whose CUDA behaviour (index_select / index_copy_ over the real slab shapes) CPU cannot show.
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no GPU visible"))]


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1))


def _linear(num_slots=12, device=DEV):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 2), num_key_heads=2, num_value_heads=4,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate="silu",
    )
    specs = (SlotStateSpec(name="ple_hist", shape=(3, 5), layer_ids=(0, 2)),)
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device(device), tp_size=1, slot_states=specs)


def _kv(num_pages, page_size, quant=None, qsa=False, device=DEV):
    device = torch.device(device)
    from freetoken.kvcache import kv_quant

    spec = kv_quant.resolve(quant)
    if qsa:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        return QSAKVCache(num_kv_heads=2, num_layers=4, head_dim=32, num_pages=num_pages + 1,
                          page_size=page_size, dtype=torch.bfloat16, device=device,
                          index_head_dim=16, num_index_layers=2, index_ratio=4, num_req_slots=2,
                          layer_ids=(1, 3), kv_quant=spec)
    from freetoken.kvcache.mha_pool import MHAKVCache

    return MHAKVCache(num_kv_heads=2, num_layers=4, head_dim=32, num_pages=num_pages + 1,
                      page_size=page_size, dtype=torch.bfloat16, device=device, layer_ids=(1, 3),
                      kv_quant=spec)


class Rig:
    """One 'process': pools, cache manager, disk cache over a directory."""

    def __init__(self, root, *, page_size=1, num_pages=512, quant=None, qsa=False, slots=12,
                 capacity=1 << 30, fp=None, min_tokens=8, min_gain=4, device="cpu"):
        self.ps = page_size
        self.kv = _kv(num_pages, page_size, quant, qsa, device)
        self.lin = _linear(slots, device)
        self.pt = torch.zeros(4, num_pages * page_size, dtype=torch.int32, device=device)
        self.cm = CacheManager(num_pages, page_size, self.pt, "hybrid_radix", linear_state_pool=self.lin)
        self.layout = PoolLayout(self.kv, self.lin, page_size)
        fp = fp or {"test": True, "layout": self.layout.describe()}
        self.store = PrefixDiskStore(str(root), capacity, fp)
        self.disk = PrefixDiskCache(self.store, self.cm, self.layout, min_tokens=min_tokens,
                                    min_gain=min_gain)

    def fill_random(self, seed):
        g = torch.Generator().manual_seed(seed)
        for buf in (self.kv._kv_buffer, self.kv._scale_buffer, getattr(self.kv, "_cmp_k_buffer", None)):
            if buf is not None:
                buf.copy_(torch.randint(0, 250, buf.shape, generator=g).to(buf.dtype))
        for buf in (self.lin.conv_states, self.lin.recurrent_states, *self.lin.slot_states.values()):
            buf.copy_(torch.randn(buf.shape, generator=g).to(buf.dtype))

    def donate(self, ids):
        """What a finished request leaves: pages for ids, a snapshot slot at their end."""
        n = len(ids) // self.ps
        tokens = self.cm._page_to_token(self.cm._allocate(n))
        slot = self.lin.alloc(1)[0]
        prefix_len, exist = self.cm.prefix_cache.insert(ids, tokens, slot)
        assert not exist
        return tokens, slot

    def pages_of(self, ids):
        m = self.cm.prefix_cache.match_prefix(ids)
        return m.cached_len, m.kv_indices, m.mamba_value

    def check_conservation(self):
        self.cm.check_integrity()  # free pages + tree pages == all pages; GDN slots not over-counted
        pc = self.cm.prefix_cache
        tree_slots = pc.mamba_evictable_size + pc.mamba_protected
        assert self.lin.num_free_slots + tree_slots == self.lin.num_slots - 1


def _page_bytes(kv, tokens, ps):
    pages = (tokens[::ps] // ps).long()
    out = [kv._kv_buffer.index_select(2, pages)]
    if kv._scale_buffer is not None:
        out.append(kv._scale_buffer.index_select(2, pages))
    if hasattr(kv, "_cmp_k_buffer"):
        r = ps // kv.index_ratio
        rows = (pages.unsqueeze(1) * r + torch.arange(r, device=pages.device)).reshape(-1)
        out.append(kv._cmp_k_buffer.index_select(1, rows))
    return out


def _state_bytes(lin, slot):
    return [lin.conv_states[:, slot].clone(), lin.recurrent_states[:, slot].clone(),
            *(t[:, slot].clone() for t in lin.slot_states.values())]


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def _pending(ids, uid=1):
    return SimpleNamespace(uid=uid, input_ids=ids, input_len=len(ids), mm_embeds=None)


def _admit(disk, req):
    while not disk.admit_gate(req):
        disk.wait_for_load(1.0)


def _hold_reads(rig):
    """Keep the rig's disk reads from finishing until the returned event is set. Without it a
    small entry can be read before admit_gate looks at the future (seen on a GPU run: 8 of 40),
    and the first call restores instead of waiting."""
    release = threading.Event()
    real_load = rig.store.load

    def held(entry, expect_ids=None):
        release.wait(10)
        return real_load(entry, expect_ids=expect_ids)

    rig.store.load = held
    return release


class _InlineReader:
    """A reader whose reads are done by the time submit returns."""

    def submit(self, fn, *args):
        fut = cf.Future()
        fut.set_result(fn(*args))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("page_size,quant,qsa", [(1, None, False), (4, "q4_0", False), (8, None, True)])
def test_written_at_idle_and_restored_after_a_restart(tmp_path, page_size, quant, qsa, device):
    ids = torch.arange(100, 100 + 64, dtype=torch.int32)
    a = Rig(tmp_path, page_size=page_size, quant=quant, qsa=qsa, device=device)
    a.fill_random(1)
    tokens, slot = a.donate(ids)
    want_kv, want_state = _page_bytes(a.kv, tokens, page_size), _state_bytes(a.lin, slot)
    assert a.disk.persist_idle() == 1
    a.disk.close(wait=True)
    assert len(a.store) == 1
    assert a.disk.persist_idle() == 0, "an entry is written once"

    b = Rig(tmp_path, page_size=page_size, quant=quant, qsa=qsa, device=device)   # restart
    b.fill_random(2)                                                  # other garbage in the pools
    b.donate(torch.arange(9000, 9000 + 16 * page_size, dtype=torch.int32))  # shift the free lists
    prompt = torch.cat([ids, torch.tensor([7, 7, 7], dtype=torch.int32)])
    req = _pending(prompt)
    release = _hold_reads(b)
    assert b.disk.admit_gate(req) is False, "waits while the entry loads"
    release.set()
    _admit(b.disk, req)
    n, kv_idx, snap = b.pages_of(prompt[:-1])
    assert n == 64 and snap is not None
    _same(_page_bytes(b.kv, kv_idx, page_size), want_kv)
    _same(_state_bytes(b.lin, snap), want_state)
    assert b.disk.stats["restored"] == 1
    b.check_conservation()
    b.disk.close(wait=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("page_size,quant,qsa", [(1, None, False), (4, "q4_0", False)])
def test_a_read_done_before_the_first_look_restores_at_once(tmp_path, page_size, quant, qsa, device):
    """A small entry on a fast disk is read before admit_gate looks at the future: the same
    call restores it, with the same bytes as the waiting path."""
    ids = torch.arange(100, 100 + 64, dtype=torch.int32)
    a = Rig(tmp_path, page_size=page_size, quant=quant, qsa=qsa, device=device)
    a.fill_random(1)
    tokens, slot = a.donate(ids)
    want_kv, want_state = _page_bytes(a.kv, tokens, page_size), _state_bytes(a.lin, slot)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    b = Rig(tmp_path, page_size=page_size, quant=quant, qsa=qsa, device=device)
    b.fill_random(2)
    b.disk._reader = _InlineReader()
    prompt = torch.cat([ids, torch.tensor([7, 7, 7], dtype=torch.int32)])
    assert b.disk.admit_gate(_pending(prompt)) is True
    assert not b.disk.loading
    n, kv_idx, snap = b.pages_of(prompt[:-1])
    assert n == 64 and snap is not None
    _same(_page_bytes(b.kv, kv_idx, page_size), want_kv)
    _same(_state_bytes(b.lin, snap), want_state)
    assert b.disk.stats["restored"] == 1
    b.check_conservation()
    b.disk.close(wait=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("page_size,qsa", [(1, False), (8, True)])
def test_only_the_part_the_tree_lacks_is_uploaded(tmp_path, page_size, qsa, device):
    ps = page_size
    ids = torch.arange(1, 49, dtype=torch.int32)
    a = Rig(tmp_path, page_size=ps, qsa=qsa, device=device)
    a.fill_random(3)
    a_tokens, _ = a.donate(ids)
    want_tail = _page_bytes(a.kv, a_tokens[16:], ps)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    b = Rig(tmp_path, page_size=ps, qsa=qsa, device=device)
    b.fill_random(4)
    head_tokens, _ = b.donate(ids[:16])            # the tree still has the first 16, with a snapshot
    head_before = _page_bytes(b.kv, head_tokens, ps)
    calls = []
    real = b.layout.write_pages

    def spy(pages, src, first):
        calls.append((len(pages), first))
        real(pages, src, first)

    b.layout.write_pages = spy
    req = _pending(torch.cat([ids, torch.tensor([5], dtype=torch.int32)]))
    _admit(b.disk, req)
    assert calls == [(32 // ps, 16 // ps)]
    n, kv_idx, _ = b.pages_of(ids)
    assert n == 48
    assert torch.equal(kv_idx[:16], head_tokens), "the tree's own pages for the head are kept"
    _same(_page_bytes(b.kv, head_tokens, ps), head_before)
    _same(_page_bytes(b.kv, kv_idx[16:], ps), want_tail)
    b.check_conservation()
    b.disk.close(wait=True)


def test_not_read_when_the_tree_already_reaches_about_as_deep(tmp_path):
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)
    b = Rig(tmp_path, min_gain=32)
    b.donate(ids[:48])
    assert b.disk.admit_gate(_pending(torch.cat([ids, ids[:1]]))) is True
    assert b.disk.stats["restored"] == 0
    b.disk.close(wait=True)


def test_image_prompts_and_short_prefixes_are_left_alone(tmp_path):
    rig = Rig(tmp_path, min_tokens=32)
    rig.donate(torch.arange(1, 17, dtype=torch.int32))
    assert rig.disk.persist_idle() == 0
    req = _pending(torch.arange(1, 100, dtype=torch.int32))
    req.mm_embeds = torch.zeros(1)
    assert rig.disk.admit_gate(req) is True
    rig.disk.close(wait=True)


def test_a_different_fingerprint_is_a_miss(tmp_path):
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path, fp={"kv_cache_dtype": "auto"})
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)
    b = Rig(tmp_path, fp={"kv_cache_dtype": "q4_0"})
    assert b.disk.admit_gate(_pending(torch.cat([ids, ids[:1]]))) is True
    assert b.disk.stats["restored"] == 0
    b.disk.close(wait=True)


def test_restore_without_room_changes_nothing(tmp_path):
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    # no GDN slot to spare: every one is held by a "running request"
    b = Rig(tmp_path, slots=12)
    held = b.lin.alloc(b.lin.num_free_slots)
    free_pages = len(b.cm.free_slots)
    _admit(b.disk, _pending(torch.cat([ids, ids[:1]])))
    assert b.disk.stats["restore_skipped"] == 1
    assert len(b.cm.free_slots) == free_pages and b.pages_of(ids)[0] == 0
    b.lin.free(held)
    b.check_conservation()
    b.disk.close(wait=True)

    # no pages to spare: the pool is smaller than the prefix
    c = Rig(tmp_path, num_pages=32)
    _admit(c.disk, _pending(torch.cat([ids, ids[:1]])))
    assert c.disk.stats["restore_skipped"] == 1 and c.pages_of(ids)[0] == 0
    c.check_conservation()
    c.disk.close(wait=True)


def test_a_write_that_fails_halfway_leaves_the_tree_as_it_was(tmp_path):
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)
    b = Rig(tmp_path)

    def boom(pages, src, first):
        raise RuntimeError("CUDA out of memory (simulated)")

    b.layout.write_pages = boom
    _admit(b.disk, _pending(torch.cat([ids, ids[:1]])))
    assert b.pages_of(ids)[0] == 0
    b.check_conservation()
    b.disk.close(wait=True)


def test_intermediate_checkpoints_are_their_own_entries(tmp_path):
    """A long prompt leaves snapshots inside it (chunk commits); each is a resume point, so a
    prompt that diverges after the first one still resumes there from disk."""
    ids = torch.arange(1, 129, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids[:64])
    a.donate(ids)
    assert a.disk.persist_idle() == 2
    a.disk.close(wait=True)
    b = Rig(tmp_path)
    other = torch.cat([ids[:70], torch.tensor([9] * 20, dtype=torch.int32)])
    _admit(b.disk, _pending(other))
    assert b.pages_of(other)[0] == 64
    b.disk.close(wait=True)


def _prefill_manager(rig, max_running=2):
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))
    tm = TableManager(max_running_reqs=max_running, page_table=rig.pt)
    pm = PrefillManager(rig.cm, tm, DecodeManager(rig.ps))
    pm.prefix_disk = rig.disk
    return pm


def _user_req(uid, ids, max_tokens=4):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(uid, ids, SamplingParams(max_tokens=max_tokens))


def test_admission_waits_for_the_read_then_admits_a_hit(tmp_path):
    """Through the real PrefillManager: no batch while the entry loads, then the request is
    admitted with the restored prefix as its cache hit and the snapshot as its restore source."""
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    b = Rig(tmp_path)
    pm = _prefill_manager(b)
    pm.pending_list.append(_user_req(5, torch.cat([ids, torch.tensor([3, 4], dtype=torch.int32)])))
    assert pm.schedule_next_batch(1024) is None
    b.disk.wait_for_load(5.0)
    batch = pm.schedule_next_batch(1024)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    assert req.cached_len == 64 and batch.prompt_admissions == [(5, 66, 64)]
    assert req.mamba_restore_src == b.pages_of(ids)[2]
    b.disk.close(wait=True)


def test_the_restore_waits_until_it_is_first_in_a_pass(tmp_path):
    """Pages taken during a pass would come out of what earlier admissions this pass were
    promised (they are allocated only after the pass), so a restore never runs behind one."""
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    b = Rig(tmp_path)
    req = _pending(torch.cat([ids, ids[:1]]))
    release = _hold_reads(b)
    assert b.disk.admit_gate(req) is False
    release.set()
    b.disk.wait_for_load(5.0)
    assert b.disk.admit_gate(req, can_restore=False) is False
    assert b.pages_of(ids)[0] == 0
    # and it leaves the running requests' reservation alone
    free_tokens = len(b.cm.free_slots) * b.ps
    assert b.disk.admit_gate(req, reserve_tokens=free_tokens - 10) is True
    assert b.pages_of(ids)[0] == 0 and b.disk.stats["restore_skipped"] == 1
    b.check_conservation()
    b.disk.close(wait=True)


def test_an_aborted_wait_does_not_leave_a_load_behind(tmp_path):
    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)
    b = Rig(tmp_path)
    req = _pending(torch.cat([ids, ids[:1]]))
    release = _hold_reads(b)
    assert b.disk.admit_gate(req) is False
    assert b.disk.loading
    del req                                   # aborted: nobody asks again
    release.set()
    b.disk.wait_for_load(5.0)
    b.disk._reader.shutdown(wait=True)
    assert not b.disk.loading
    b.disk.close(wait=True)


def test_a_read_slower_than_recomputing_is_abandoned(tmp_path):
    """Seen on a 2060 under memory pressure: an 83 MiB entry took 7.95 s to read for a prompt
    that prefills in 3 s. Past the deadline the request stops waiting and prefills."""
    import threading

    ids = torch.arange(1, 65, dtype=torch.int32)
    a = Rig(tmp_path)
    a.donate(ids)
    a.disk.persist_idle()
    a.disk.close(wait=True)

    b = Rig(tmp_path)
    b.disk.read_deadline_base_s = 0.05
    b.disk.read_deadline_tokens_per_s = 1e9
    release = threading.Event()
    real_load = b.store.load

    def slow_load(entry, expect_ids=None):
        release.wait(10)
        return real_load(entry, expect_ids=expect_ids)

    b.store.load = slow_load
    req = _pending(torch.cat([ids, ids[:1]]))
    assert b.disk.admit_gate(req) is False
    import time as _t
    _t.sleep(0.1)
    assert b.disk.admit_gate(req) is True, "past the deadline it goes on without the entry"
    assert b.disk.stats["load_abandoned"] == 1 and b.pages_of(ids)[0] == 0
    release.set()
    b.disk.close(wait=True)
    assert b.disk.admit_gate(req) is True and b.pages_of(ids)[0] == 0, "a late result is dropped"
    b.check_conservation()


def test_pool_classes_it_does_not_know_are_refused():
    class Other:
        pass

    with pytest.raises(PrefixDiskUnsupported):
        PoolLayout(Other(), _linear(), 1)
    with pytest.raises(PrefixDiskUnsupported):
        PoolLayout(_kv(8, 1), None, 1)


def test_the_fingerprint_moves_with_what_the_bytes_mean(tmp_path):
    from freetoken.kvcache.prefix_disk_store import fingerprint_digest

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"a": 1}))
    (model / "model.safetensors").write_bytes(b"x" * 10)
    layout = PoolLayout(_kv(8, 1), _linear(), 1)

    def cfg(**over):
        base = dict(model_path=str(model), dtype=torch.bfloat16, kv_cache_dtype=None,
                    dense_quant="none", quant_backend=None, page_size=1, cache_type="hybrid_radix",
                    parallel="tp", tp_info=SimpleNamespace(size=1, rank=0), pp_split=None,
                    model_config=SimpleNamespace(mtp_layer_id=None))
        base.update(over)
        return SimpleNamespace(**base)

    base = fingerprint_digest(build_fingerprint(cfg(), layout))
    assert fingerprint_digest(build_fingerprint(cfg(), layout)) == base
    for over in (dict(kv_cache_dtype="q4_0"), dict(dense_quant="fp8"), dict(page_size=64),
                 dict(dtype=torch.float16), dict(model_config=SimpleNamespace(mtp_layer_id=40))):
        assert fingerprint_digest(build_fingerprint(cfg(**over), layout)) != base, over
    assert fingerprint_digest(build_fingerprint(cfg(), PoolLayout(_kv(8, 1, "q8_0"), _linear(), 1))) != base
    (model / "model.safetensors").write_bytes(b"y" * 11)            # re-downloaded weights
    assert fingerprint_digest(build_fingerprint(cfg(), layout)) != base


def test_an_uncommitted_change_moves_the_code_identity(tmp_path):
    """The first version passed the pathspec "python" from inside python/, which names
    python/python: the diff was always empty and an edited kernel read the old code's entries."""
    import shutil
    import subprocess

    from freetoken.scheduler.prefix_disk import code_identity

    if shutil.which("git") is None:
        pytest.skip("no git")
    repo = tmp_path / "repo"
    src = repo / "python" / "pkg"
    src.mkdir(parents=True)
    (src / "kernel.py").write_text("SCALE = 1")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env, capture_output=True)
    clean = code_identity(str(repo / "python"))
    assert "commit" in clean and "uncommitted" not in clean
    (src / "kernel.py").write_text("SCALE = 2")
    dirty = code_identity(str(repo / "python"))
    assert dirty["commit"] == clean["commit"] and dirty.get("uncommitted")
    (src / "kernel.py").write_text("SCALE = 3")
    assert code_identity(str(repo / "python"))["uncommitted"] != dirty["uncommitted"]


def _parse(argv):
    pytest.importorskip("freetoken.server.args")
    from freetoken.server.args import parse_args

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"],
                       "model_type": "qwen4_exp", "torch_dtype": "bfloat16"}, f)
        args, _ = parse_args(
            ["--model", d, "--dtype", "bfloat16", "--tool-call-parser", "llama3",
             "--reasoning-parser", "off", *argv],
            False,
        )
    return args


def test_flags_off_by_default_and_reach_the_config():
    args = _parse([])
    assert args.prefix_disk_cache is None and args.prefix_disk_cache_size == "32G"
    args = _parse(["--prefix-disk-cache", "/tmp/pfx", "--prefix-disk-cache-size", "8G"])
    assert args.prefix_disk_cache == "/tmp/pfx" and args.prefix_disk_cache_size == "8G"


def test_flags_take_pipeline_ranks_and_refuse_a_bad_size():
    args = _parse(["--prefix-disk-cache", "/tmp/pfx", "--pp-size", "2"])
    assert args.prefix_disk_cache == "/tmp/pfx" and args.parallel == "pp"
    with pytest.raises(SystemExit):
        _parse(["--prefix-disk-cache", "/tmp/pfx", "--prefix-disk-cache-size", "lots"])
