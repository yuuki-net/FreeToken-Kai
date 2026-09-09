"""Mapped expert banks: layout arithmetic, the permuted write, and what the mapping hands back.

The failure to guard against is silent: a block written at the wrong offset, or a row order
that disagrees with the placement, gives the model other experts' weights. Shapes and lengths
all still check out, and the output stays fluent. So the round trip compares bytes.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.bank_disk import plan_placement  # noqa: E402
from freetoken.moe.mapped_bank import (  # noqa: E402
    ALIGN,
    MappedBankLayout,
    MappedBanks,
    MappedBankWriter,
    layout_from,
)


def _sources(num_layers=3, num_experts=6):
    src = {}
    for name, cols, dtype in (("packed", 5, torch.uint8), ("scale", 3, torch.float16)):
        layers = []
        for layer in range(num_layers):
            t = torch.empty((num_experts, cols), dtype=dtype)
            for e in range(num_experts):
                t[e] = (layer * num_experts + e) % 251 + 1
            layers.append(t)
        src[name] = layers
    return src


def _built(tmp_path, num_layers=3, num_experts=6, hot=4, freq=None):
    src = _sources(num_layers, num_experts)
    layers = list(range(num_layers))
    p = plan_placement(layers, num_experts, hot, freq)
    lay = layout_from(src, layers, p)
    path = str(tmp_path / "b.ftmb")
    w = MappedBankWriter(path, lay)
    for layer in layers:
        w.write_layer(layer, {n: src[n][layer] for n in src})
    w.close()
    return src, p, lay, path


# ----- layout ---------------------------------------------------------------------------
def test_blocks_are_aligned_and_do_not_overlap():
    src = _sources()
    p = plan_placement([0, 1, 2], 6, 4, None)
    lay = layout_from(src, [0, 1, 2], p)
    assert lay.data_offset % ALIGN == 0
    spans = []
    for name, _, _, _ in lay.banks:
        for layer in lay.layers:
            off = lay.offset_of(name, layer)
            assert off % ALIGN == 0
            spans.append((off, off + lay.block_bytes(name)))
    spans.sort()
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start
    assert spans[-1][1] <= lay.total_bytes()
    assert spans[0][0] >= lay.data_offset


def test_layout_header_round_trip(tmp_path):
    _, _, lay, path = _built(tmp_path)
    back = MappedBankLayout.read(path)
    assert back.same_as(lay)
    assert back.order == lay.order and back.banks == lay.banks


def test_same_as_rejects_a_different_row_order():
    src = _sources()
    a = layout_from(src, [0, 1, 2], plan_placement([0, 1, 2], 6, 4, None))
    freq = {0: [9, 0, 8, 1, 7, 2], 1: [0, 9, 1, 8, 2, 7], 2: [1, 2, 3, 4, 5, 6]}
    b = layout_from(src, [0, 1, 2], plan_placement([0, 1, 2], 6, 4, freq))
    # same shapes, same counts, different experts in the resident prefix
    assert not a.same_as(b)


def test_same_as_rejects_a_different_resident_count():
    src = _sources()
    a = layout_from(src, [0, 1, 2], plan_placement([0, 1, 2], 6, 4, None))
    b = layout_from(src, [0, 1, 2], plan_placement([0, 1, 2], 6, 3, None))
    assert not a.same_as(b)


def test_bad_magic_is_refused(tmp_path):
    _, _, _, path = _built(tmp_path)
    with open(path, "r+b") as f:
        f.write(b"XXXX")
    with pytest.raises(ValueError, match="not a mapped bank"):
        MappedBankLayout.read(path)


# ----- round trip -----------------------------------------------------------------------
def test_views_hold_the_permuted_rows(tmp_path):
    freq = {0: [0, 9, 8, 0, 7, 0], 1: [5, 0, 0, 6, 0, 7], 2: [1, 2, 3, 4, 5, 6]}
    src, p, _, path = _built(tmp_path, freq=freq)
    banks = MappedBanks(path, register=False)
    try:
        for name in src:
            assert len(banks.sources[name]) == 3
            for layer in (0, 1, 2):
                view = banks.sources[name][layer]
                # shape is unchanged: this is the whole point, the cache never sees a
                # shrunken bank
                assert view.shape == src[name][layer].shape
                assert view.dtype == src[name][layer].dtype
                for physical in range(6):
                    logical = p.order[layer][physical]
                    assert torch.equal(view[physical], src[name][layer][logical])
    finally:
        banks.close()


def test_layers_may_be_written_from_many_threads(tmp_path):
    import threading

    src = _sources(num_layers=6, num_experts=6)
    layers = list(range(6))
    p = plan_placement(layers, 6, 4, None)
    lay = layout_from(src, layers, p)
    path = str(tmp_path / "b.ftmb")
    w = MappedBankWriter(path, lay)
    ts = [
        threading.Thread(target=w.write_layer, args=(layer, {n: src[n][layer] for n in src}))
        for layer in layers
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    w.close()
    banks = MappedBanks(path, register=False)
    try:
        for layer in layers:
            for physical in range(6):
                logical = p.order[layer][physical]
                assert torch.equal(banks.sources["packed"][layer][physical],
                                   src["packed"][layer][logical])
    finally:
        banks.close()


def test_a_missing_layer_removes_the_file(tmp_path):
    src = _sources(num_layers=2, num_experts=4)
    p = plan_placement([0, 1], 4, 2, None)
    lay = layout_from(src, [0, 1], p)
    path = str(tmp_path / "b.ftmb")
    w = MappedBankWriter(path, lay)
    w.write_layer(0, {n: src[n][0] for n in src})
    with pytest.raises(ValueError, match="never written"):
        w.close()
    # zeroed experts do not crash a model, so a half-written file must not survive to be
    # picked up as reusable on the next start
    assert not os.path.exists(path)


def test_everything_resident_is_still_a_valid_file(tmp_path):
    src, p, _, path = _built(tmp_path, hot=6)
    banks = MappedBanks(path, register=False)
    try:
        assert banks.layout.hot_per_layer == 6
        assert torch.equal(banks.sources["packed"][0][0], src["packed"][0][p.order[0][0]])
    finally:
        banks.close()


# ----- madvise ranges -------------------------------------------------------------------
from freetoken.moe.mapped_bank import advise_range  # noqa: E402


def test_advise_range_aligns_the_start_up():
    # madvise rejects an unaligned address, and a resident prefix of 387 rows x 2560 B lands
    # mid-page; rounding up keeps the hint off the region before it
    assert advise_range(ALIGN + 20, 3 * ALIGN, 100 * ALIGN) == (2 * ALIGN, 2 * ALIGN + 20)
    assert advise_range(0, ALIGN, 100 * ALIGN) == (0, ALIGN)


def test_advise_range_is_clamped_to_the_mapping():
    assert advise_range(90 * ALIGN, 50 * ALIGN, 100 * ALIGN) == (90 * ALIGN, 10 * ALIGN)
    assert advise_range(100 * ALIGN, ALIGN, 100 * ALIGN) is None


def test_advise_range_declines_when_nothing_is_left():
    # a range that rounds past its own end has no whole page to advise
    assert advise_range(10, 100, 100 * ALIGN) is None
    assert advise_range(0, 0, 100 * ALIGN) is None


# ----- attach: what the GPU is allowed to touch -------------------------------------------


class _FakeCache:
    """Just the attributes MappedTier.attach writes."""

    def __init__(self):
        self.expert_perm = None
        self.prefix_pinned_rows = None
        self.hybrid_max_fetch = 512
        self.hybrid_fetch_fraction = 0.194


def _tier(tmp_path, registered, hot=4):
    from freetoken.moe.mapped_bank import MappedTier

    _, p, _, path = _built(tmp_path, hot=hot)
    tier = MappedTier(p, path, list(range(3)))
    tier.banks = MappedBanks(path, register=False)
    tier.banks.registered_bytes = registered
    return tier


def test_a_registered_prefix_keeps_the_pcie_fetch_and_bounds_it(tmp_path):
    tier = _tier(tmp_path, registered=1 << 20)
    cache = _FakeCache()
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    # the fetch path stays on -- losing it is what sends every miss to the CPU and costs
    # the VRAM expert cache
    assert cache.hybrid_max_fetch == 512
    assert cache.hybrid_fetch_fraction == pytest.approx(0.194)
    # ... but only rows the device can address are eligible
    assert cache.prefix_pinned_rows == 4


def test_nothing_registered_falls_back_to_the_cpu_for_every_miss(tmp_path):
    tier = _tier(tmp_path, registered=0)
    cache = _FakeCache()
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    assert cache.prefix_pinned_rows is None
    # both knobs, not just the count: --moe-hybrid-max-fetch auto sets the fraction, and
    # the fraction wins
    assert cache.hybrid_max_fetch == 0
    assert cache.hybrid_fetch_fraction == 0.0


def test_readahead_kb_reads_the_devices_window(tmp_path):
    from freetoken.moe.mapped_bank import readahead_kb

    got = readahead_kb(str(tmp_path))
    if got is None:  # no sysfs (Windows, or a filesystem with no block device)
        pytest.skip("no /sys/dev/block for this path")
    kb, where = got
    assert kb > 0
    assert where.endswith("read_ahead_kb")


def test_readahead_kb_is_none_for_a_missing_path(tmp_path):
    from freetoken.moe.mapped_bank import readahead_kb

    assert readahead_kb(str(tmp_path / "nope")) is None


def test_small_blocks_are_kept_whole(tmp_path):
    """A block smaller than the readahead window costs more to fault than to keep."""
    import freetoken.moe.mapped_bank as mb

    _, _, lay, path = _built(tmp_path, num_layers=2, num_experts=8, hot=5)
    small = min(rb for _, _, _, rb in lay.banks) * lay.num_experts
    big = max(rb for _, _, _, rb in lay.banks) * lay.num_experts
    assert small < big, "the fixture needs blocks of different sizes"
    saved = mb._WHOLE_BLOCK_BYTES
    mb._WHOLE_BLOCK_BYTES = small  # whole for the narrow bank, split for the wide one
    try:
        banks = MappedBanks(path, register=False)
    finally:
        mb._WHOLE_BLOCK_BYTES = saved
    try:
        # the narrow bank, for every layer; the wide one keeps its hot/cold split
        assert banks.whole_blocks == len(lay.layers)
    finally:
        banks.close()


def test_a_failed_register_flag_is_cleared_before_the_next_allocation(monkeypatch):
    """cudaHostRegister leaves a sticky cudaErrorMemoryAllocation on the context when it
    refuses a flag, and the next device allocation of any size dies with "CUDA error: out of
    memory" -- a 48 KB tensor in OffloadMoeCache, in practice. The path that mattered was
    flags=0 refused and ReadOnly accepted: the bank reported itself registered and the boot
    died anyway, because the clear only ran when every flag had failed."""
    import torch

    from freetoken.moe.mapped_bank import MappedBanks

    attempts, syncs = [], []

    class _Cudart:
        def cudaHostRegister(self, addr, nbytes, flags):
            attempts.append(flags)
            return 0 if flags == 0x08 else 2  # this driver takes read-only file pages only

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: syncs.append(len(attempts)))

    import mmap as _mmap

    bank = MappedBanks.__new__(MappedBanks)
    bank._base, bank._libc = 0x1000, None
    bank._map = _mmap.mmap(-1, 1 << 20)  # a real mapping: _settle advises it before locking
    bank.locked_bytes = bank.registered_bytes = 0
    bank._registered = []
    bank._settle(offset=0, nbytes=4096, block_bytes=4096, register=True)

    assert bank.registered_bytes == 4096, "the read-only flag registers the prefix"
    assert attempts and attempts[0] == 0x08, f"read-only first, got {attempts}"
    # every refusal is cleared, whether or not a later flag went on to succeed
    assert len(syncs) == sum(1 for f in attempts if f != 0x08), (attempts, syncs)
