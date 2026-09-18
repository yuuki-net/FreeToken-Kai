"""The host tier of the GDN snapshots (--linear-state-host-slots).

A hybrid GDN model can only resume a prefix from a live GDN snapshot, and the snapshots live in a
handful of VRAM slots, so the number of conversations that stay reusable is the slot count, not
the KV budget (docs/prefix-reuse.md: 8 slots on a 2060 hold two Ornith conversations, and ten
643-token prompts asked twice hit nothing the second time). The host tier moves the least
recently used snapshot to pinned RAM instead of dropping it.

CPU-only: the pool runs on the CPU device, the tree and the CacheManager are the real ones.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager

PROMPT = 128  # tokens per conversation; a multiple of the GDN chunk, so the insert keeps it whole


def _pool(num_slots: int, host_slots: int = 0) -> LinearStatePool:
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(
        group=group, num_slots=num_slots, dtype=torch.bfloat16, device=torch.device("cpu"),
        tp_size=1, slot_states=(SlotStateSpec(name="ple", layer_ids=(), shape=(3,)),),
    )
    pool.enable_host_tier(host_slots)
    return pool


def _fill(pool: LinearStatePool, slot: int, v: float) -> None:
    pool.conv_states[:, slot] = v
    pool.recurrent_states[:, slot] = v + 0.5
    pool.slot_states["ple"][:, slot] = v + 0.25


def _value(pool: LinearStatePool, slot: int) -> tuple[float, float, float]:
    views = dict(pool.state_views(slot))
    return (float(views["conv"].float().mean()), float(views["recurrent"].float().mean()),
            float(views["slot.ple"].float().mean()))


def _manager(pool: LinearStatePool, num_pages: int = 4096) -> CacheManager:
    cm = object.__new__(CacheManager)
    cm.is_hybrid, cm.is_swa = True, False
    cm.page_size = 1
    cm.num_pages = num_pages
    cm.free_slots = torch.arange(num_pages, dtype=torch.int32)
    cm.prefix_cache = HybridRadixCache(torch.device("cpu"), 1)
    cm.linear_state_pool = pool
    return cm


def _ids(conv: int) -> torch.Tensor:
    return torch.full((PROMPT,), 1000 + conv, dtype=torch.int32)


def _donate(cm: CacheManager, conv: int) -> None:
    """What a finished request does: its live slot, filled with the conversation's state, is
    donated to the tree with the conversation's KV."""
    cm.ensure_mamba_slots(1)
    slot = cm.linear_state_pool.alloc(1)[0]
    _fill(cm.linear_state_pool, slot, float(conv))
    kv = cm._allocate(PROMPT)
    _, exist = cm.prefix_cache.insert(_ids(conv), kv, slot)
    assert not exist


def _reusable(cm: CacheManager, n: int) -> list[int]:
    return [c for c in range(n) if cm.prefix_cache.match_prefix(_ids(c)).mamba_value is not None]


# ---------------------------------------------------------------- the pool alone
def test_demote_moves_the_state_down_and_gives_the_device_slot_back():
    pool = _pool(num_slots=4, host_slots=2)
    s = pool.alloc(1)[0]
    _fill(pool, s, 7.0)
    free_before = pool.num_free_slots
    h = pool.demote(s)
    assert pool.is_host(h) and not pool.is_host(s)
    assert pool.num_free_slots == free_before + 1 and pool.num_free_host_slots == 1
    _fill(pool, s, -1.0)                   # the device slot is somebody else's now
    assert _value(pool, h) == (7.0, 7.5, 7.25)


def test_copy_from_a_host_snapshot_restores_it_exactly():
    pool = _pool(num_slots=4, host_slots=1)
    s = pool.alloc(1)[0]
    _fill(pool, s, 3.0)
    ref = [t.clone() for _, t in pool.state_views(s)]
    h = pool.demote(s)
    live = pool.alloc(1)[0]
    _fill(pool, live, 0.0)
    pool.copy_from(h, live)
    assert all(torch.equal(a, b) for a, (_, b) in zip(ref, pool.state_views(live)))


def test_free_and_reclaim_route_host_ids_to_the_host_list():
    pool = _pool(num_slots=4, host_slots=2)
    h = pool.demote(pool.alloc(1)[0])
    assert pool.num_free_host_slots == 1
    pool.free([h])
    assert pool.num_free_host_slots == 2 and pool.num_free_slots == 3
    pool.demote(pool.alloc(1)[0])
    pool.reclaim_all_slots()
    assert pool.num_free_host_slots == 2 and pool.num_free_slots == 3


def test_no_host_tier_is_the_old_pool():
    pool = _pool(num_slots=4, host_slots=0)
    assert pool.host_slots == 0 and pool.num_free_host_slots == 0
    with pytest.raises(AssertionError):
        pool.demote(pool.alloc(1)[0])


# ---------------------------------------------------------------- through the cache manager
def test_without_the_host_tier_old_conversations_lose_their_reuse_point():
    # the ceiling itself, so the next test cannot pass for the wrong reason
    cm = _manager(_pool(num_slots=4))           # 3 usable device slots
    for c in range(6):
        _donate(cm, c)
    assert _reusable(cm, 6) == [3, 4, 5]
    cm.check_integrity()


def test_the_host_tier_keeps_them_and_restores_the_right_state():
    # 3 device + 4 host: the six snapshots and the live slot each restore needs
    pool = _pool(num_slots=4, host_slots=4)
    cm = _manager(pool)
    for c in range(6):
        _donate(cm, c)
    assert _reusable(cm, 6) == list(range(6))
    cm.check_integrity()
    # the oldest went down, the newest stayed up
    slots = {c: cm.prefix_cache.match_prefix(_ids(c)).mamba_value for c in range(6)}
    assert [pool.is_host(slots[c]) for c in range(6)] == [True, True, True, False, False, False]
    # each one comes back as its own state, the way admission does it: match, lock (so taking
    # the live slot cannot move or drop this snapshot), take a live slot, copy up
    for c in range(6):
        m = cm.prefix_cache.match_prefix(_ids(c))
        cm.prefix_cache.inc_lock(m.node)
        cm.ensure_mamba_slots(1)
        live = pool.alloc(1)[0]
        assert cm.prefix_cache.match_prefix(_ids(c)).mamba_value == m.mamba_value
        pool.copy_from(m.mamba_value, live)
        assert _value(pool, live) == (float(c), c + 0.5, c + 0.25)
        pool.free([live])
        cm.prefix_cache.dec_lock(m.node)
    cm.check_integrity()


def test_past_both_tiers_the_least_recently_used_is_dropped():
    pool = _pool(num_slots=4, host_slots=3)
    cm = _manager(pool)
    for c in range(8):
        _donate(cm, c)
    assert _reusable(cm, 8) == [2, 3, 4, 5, 6, 7]
    cm.check_integrity()


def test_locked_host_snapshots_do_not_block_a_device_eviction():
    # A host snapshot gives no device slot back. When the host is full of snapshots somebody is
    # restoring from, a device snapshot must still be evictable -- not every host one freed in
    # a loop while the device stays full.
    pool = _pool(num_slots=4, host_slots=2)
    cm = _manager(pool)
    for c in range(5):
        _donate(cm, c)                               # 0, 1 on the host; 2, 3, 4 on the device
    locks = [cm.prefix_cache.match_prefix(_ids(c)).node for c in (0, 1)]
    for node in locks:
        cm.prefix_cache.inc_lock(node)
    assert pool.num_free_slots == 0
    cm.ensure_mamba_slots(1)
    assert pool.num_free_slots == 1
    assert _reusable(cm, 5) == [0, 1, 3, 4]          # the device's LRU (2) went, the locked stayed
    for node in locks:
        cm.prefix_cache.dec_lock(node)
    cm.check_integrity()


def test_available_size_counts_only_what_frees_a_device_slot():
    pool = _pool(num_slots=4, host_slots=3)
    cm = _manager(pool)
    for c in range(6):
        _donate(cm, c)
    # 3 device slots all held by unlocked snapshots, 3 host snapshots that free nothing up there
    assert cm.mamba_available_size == 3


def test_a_kv_eviction_frees_a_host_snapshot_to_the_host_list():
    pool = _pool(num_slots=4, host_slots=3)
    cm = _manager(pool, num_pages=6 * PROMPT)
    for c in range(6):
        _donate(cm, c)
    cm._allocate(PROMPT)                             # evicts the KV LRU leaf, a host snapshot
    assert pool.num_free_host_slots == 1
    assert _reusable(cm, 6) == [1, 2, 3, 4, 5]


def test_pipeline_ranks_make_the_same_choices_without_talking():
    # --pp-size 2: each rank holds only its own GDN layers, so its pool has different bytes per
    # slot, but the same slot COUNTS and the same tree. Every choice is made from those, so the
    # two replicas must end on the same slot ids with no message between them.
    def rank(layer_ids):
        group = LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=layer_ids, num_key_heads=2, num_value_heads=4,
            key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
        )
        pool = LinearStatePool(group=group, num_slots=4, dtype=torch.bfloat16,
                               device=torch.device("cpu"), tp_size=1)
        pool.enable_host_tier(3)
        return _manager(pool)

    a, b = rank((0, 1, 2)), rank((3,))
    for cm in (a, b):
        for c in range(8):
            cm.ensure_mamba_slots(1)
            slot = cm.linear_state_pool.alloc(1)[0]
            cm.prefix_cache.insert(_ids(c), cm._allocate(PROMPT), slot)
    assert [a.prefix_cache.match_prefix(_ids(c)).mamba_value for c in range(8)] == \
           [b.prefix_cache.match_prefix(_ids(c)).mamba_value for c in range(8)]
