"""Bookkeeping of the chunked-prefill GDN checkpoints (scheduler/cache.commit_chunk_checkpoint).

A commit for a NON-final chunk runs while the next chunk is already in flight over the same
page-table row, so it may hand pages to the tree but must not free or re-point anything. That
splits the prefix into two kinds of span, and getting the split wrong is not a crash at the
commit -- it is a page counted as both free and tree-owned, hours later, in whichever request
happens to free it. These tests pin the split itself.

The first attempt at this treated "the tree already had a longer prefix than my handle" as
"somebody else owns those pages, leave everything alone" and returned without advancing the
handle. But ``insert`` has by then already built a node out of THIS request's pages, and the
handle is what tells the finish path where its own free floor is -- so the finish freed pages
the tree owned. ``test_handle_advances_even_when_the_tree_already_had_a_longer_prefix`` is that
bug: it fails against the version that returns early.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.cache import CacheManager


class FakeTree:
    """Only what the commit calls: insert() reports what was already there, match_prefix()
    answers with the node the insert left at the end boundary."""

    def __init__(self, already: int = 0):
        self.already = already
        self.inserted: list[tuple[int, int]] = []
        self.canonical = torch.arange(10_000, 20_000, dtype=torch.int32)

    def insert(self, input_ids, kv_indices, mamba_value):
        length = len(input_ids)
        self.inserted.append((length, mamba_value))
        prefix_len = min(self.already, length)
        self.already = max(self.already, length)
        return prefix_len, False          # False: the tree took the donated slot

    def match_prefix(self, input_ids):
        n = len(input_ids)
        return SimpleNamespace(cached_len=n, node=f"node@{n}", kv_indices=self.canonical[:n])


def _manager(page_size: int = 1, already: int = 0):
    cm = object.__new__(CacheManager)
    cm.is_hybrid = True
    cm.page_size = page_size
    cm.page_table = torch.arange(4 * 40_000, dtype=torch.int32).reshape(4, 40_000)
    cm.prefix_cache = FakeTree(already)
    cm.linear_state_pool = SimpleNamespace(alloc=lambda n: [777])
    cm.ensure_mamba_slots = lambda n: None
    cm.freed: list[torch.Tensor] = []
    cm._free = cm.freed.append
    cm.locked, cm.unlocked = [], []
    cm.lock = cm.locked.append
    cm.unlock = cm.unlocked.append
    return cm


def _req(*, cached_len=4096, handle_len=0, track=2048, chunk_upto=None, table_idx=1):
    handle = SimpleNamespace(cached_len=handle_len,
                             get_matched_indices=lambda: torch.arange(10_000, 20_000,
                                                                      dtype=torch.int32))
    return SimpleNamespace(
        uid=7, table_idx=table_idx, cached_len=cached_len, input_ids=torch.zeros(cached_len),
        mm_embeds=None, mamba_last_track_seqlen=track, mamba_ping_pong=(11, 12),
        mamba_next_track_idx=0, cache_handle=handle, successor=None,
        chunk_upto=chunk_upto, chunk_dups=[],
    )


def test_a_fresh_prefix_is_handed_over_whole_and_leaves_no_duplicate():
    cm, req = _manager(already=0), _req()
    cm.commit_chunk_checkpoint(req)
    assert req.chunk_dups == [], "nothing was already in the tree, so nothing is a duplicate"
    assert req.chunk_upto == 2048
    assert req.cache_handle.cached_len == 2048
    assert cm.freed == [], "a chunk commit must never free while the next chunk is in flight"


def test_handle_advances_even_when_the_tree_already_had_a_longer_prefix():
    """The first attempt returned early here and left the handle at 0; the finish path then
    used 0 as its free floor and handed back pages the tree owned."""
    cm, req = _manager(already=7), _req()          # 7 tokens of shared chat template
    cm.commit_chunk_checkpoint(req)
    assert req.cache_handle.cached_len == 2048, "the tree owns [7, 2048) of this request's pages"
    assert req.chunk_dups == [(0, 7)], "only the shared 7 tokens are this request's duplicate"
    assert cm.freed == []


def test_the_watermark_carries_across_chunks():
    cm = _manager(already=7)
    req = _req(track=2048)
    cm.commit_chunk_checkpoint(req)
    req.mamba_last_track_seqlen = 4096             # the next chunk's boundary
    cm.commit_chunk_checkpoint(req)
    assert req.chunk_upto == 4096
    assert req.cache_handle.cached_len == 4096
    # [2048, 4096) was ours to give: the only duplicate is still the original shared prefix
    assert req.chunk_dups == [(0, 7)]


def test_the_in_flight_successor_gets_the_new_handle():
    cm, req = _manager(), _req()
    tail = SimpleNamespace(cache_handle=None, successor=None)
    req.successor = SimpleNamespace(cache_handle=None, successor=tail)
    cm.commit_chunk_checkpoint(req)
    assert req.successor.cache_handle is req.cache_handle
    assert tail.cache_handle is req.cache_handle, "the whole chain, not just the next one"


def test_a_boundary_that_is_not_page_aligned_is_skipped():
    cm, req = _manager(page_size=64), _req(track=2000)   # 2000 is not a multiple of 64
    cm.commit_chunk_checkpoint(req)
    assert cm.prefix_cache.inserted == [], "attaching there would date the state to a shorter node"
    assert req.chunk_upto is None and req.cache_handle.cached_len == 0


@pytest.mark.parametrize("track", [0, 6000])
def test_a_boundary_outside_this_chunk_is_skipped(track):
    cm, req = _manager(), _req(cached_len=4096, track=track or None)
    cm.commit_chunk_checkpoint(req)
    assert cm.prefix_cache.inserted == []


def test_settle_repoints_the_row_and_frees_only_our_duplicates():
    cm, req = _manager(), _req(handle_len=4096)
    req.chunk_dups = [(0, 7), (100, 120)]
    ours = [cm.page_table[req.table_idx, 0:7].clone(), cm.page_table[req.table_idx, 100:120].clone()]
    cm._settle_chunk_dups(req)
    assert req.chunk_dups == [] and req.chunk_upto is None
    assert [t.tolist() for t in cm.freed] == [t.tolist() for t in ours]
    canonical = req.cache_handle.get_matched_indices()
    assert cm.page_table[req.table_idx, 0:7].tolist() == canonical[0:7].tolist()
    assert cm.page_table[req.table_idx, 100:120].tolist() == canonical[100:120].tolist()


def test_settle_leaves_a_span_the_handle_never_covered():
    cm, req = _manager(), _req(handle_len=0)
    req.cache_handle.get_matched_indices = lambda: torch.zeros(0, dtype=torch.int32)
    req.chunk_dups = [(0, 7)]
    cm._settle_chunk_dups(req)
    assert cm.freed == [], "no canonical pages to point at -- freeing ours would strand the row"


def test_settle_is_a_no_op_for_a_freed_request():
    cm, req = _manager(), _req(table_idx=-1)
    req.chunk_dups = [(0, 7)]
    cm._settle_chunk_dups(req)
    assert cm.freed == []
