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


def _tier(tmp_path, registered, hot=4, blocks=6, registered_blocks=None):
    from freetoken.moe.mapped_bank import MappedTier

    _, p, _, path = _built(tmp_path, hot=hot)
    tier = MappedTier(p, path, list(range(3)))
    tier.banks = MappedBanks(path, register=False)
    tier.banks.registered_bytes = registered
    tier.banks.hot_blocks = blocks
    # every resident block registered unless the caller is asking for the partial case
    tier.banks.registered_blocks = (
        blocks if registered_blocks is None and registered else registered_blocks or 0
    )
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


def test_a_bank_registered_in_part_is_treated_as_not_registered(tmp_path):
    """prefix_pinned_rows は層にもブロックにも 1 つしかない。

    それは「行 [0, hot) はどこでもデバイスから触れる」という主張で、1 ブロックでも登録に
    失敗していれば嘘になる。嘘になった先は decode graph の中の illegal access で、
    そこまで行かせない。いまのところ登録は全部通るか全部断られるかのどちらかだが、
    24 GiB ぶんのブロックの途中で上限に当たればそうではなくなる。
    """
    tier = _tier(tmp_path, registered=1 << 20, blocks=6, registered_blocks=5)
    cache = _FakeCache()
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    assert cache.prefix_pinned_rows is None
    assert cache.hybrid_max_fetch == 0
    assert cache.hybrid_fetch_fraction == 0.0


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


def _bare_bank(libc=None, private=False):
    """A MappedBanks with only what ``_settle`` touches, over an anonymous mapping."""
    import mmap as _mmap

    from freetoken.moe.mapped_bank import MappedBanks

    bank = MappedBanks.__new__(MappedBanks)
    bank._base, bank._libc = 0x1000, libc
    bank._map = _mmap.mmap(-1, 1 << 20)  # a real mapping: _settle advises it before locking
    bank.locked_bytes = bank.requested_bytes = bank.registered_bytes = 0
    bank.lock_errno = 0
    bank._registered = []
    bank.hot_blocks = bank.registered_blocks = 0
    bank.private = private
    bank._buf = torch.frombuffer(bank._map, dtype=torch.uint8) if private else None
    if private:
        bank._base = bank._buf.data_ptr()
    return bank


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_clear_cuda_error_really_clears_it():
    """The mechanism, not the call.

    A refused cudaHostRegister leaves its error in this thread's slot, and the next kernel
    launch reports it -- OffloadMoeCache's 48 KB torch.full, in practice, which is where an
    RTX 3060 pair died with "CUDA error: invalid argument" and the registration nowhere in
    the traceback. torch.cuda.synchronize() does NOT clear that slot, which is exactly what
    the previous version of this test could not see: it asserted that synchronize was called.
    """
    from freetoken.utils.torch_utils import clear_cuda_error

    dev = torch.device("cuda:0")
    torch.full((4, 4), -1, dtype=torch.int32, device=dev)  # make the context

    # 0x1 is not a host allocation: the driver refuses with cudaErrorInvalidValue, and the
    # error stands until something reads it.
    assert int(torch.cuda.cudart().cudaHostRegister(0x1, 4096, 0)) != 0

    torch.cuda.synchronize()  # the old clear: returns success, leaves the slot set
    clear_cuda_error()
    torch.full((24, 128), -1, dtype=torch.int32, device=dev)  # must not raise


def test_every_refused_register_flag_is_cleared(monkeypatch):
    """Clear after each refusal, not only when they all fail.

    Two hosts to survive: one that refuses flags=0 and takes ReadOnly (the clear has to run
    even though the bank ends up registered), and one that refuses both -- an RTX 3060 pair
    whose cudaDevAttrHostRegisterReadOnlySupported is 0, where registering nothing at all is
    a supported state and must not poison the context.
    """
    import freetoken.moe.mapped_bank as mb

    for accepts in (0x08, None):
        attempts, cleared = [], []

        class _Cudart:
            def cudaHostRegister(self, addr, nbytes, flags):
                attempts.append(flags)
                return 0 if flags == accepts else 1

        monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
        monkeypatch.setattr(mb, "clear_cuda_error", lambda: cleared.append(len(attempts)))

        bank = _bare_bank()
        bank._settle(offset=0, nbytes=4096, block_bytes=4096, register=True)

        assert attempts[0] == 0x08, f"read-only first, got {attempts}"
        assert len(cleared) == sum(1 for f in attempts if f != accepts), (attempts, cleared)
        assert bank.registered_bytes == (4096 if accepts is not None else 0)


def test_the_private_form_asks_for_plain_pinning_first(monkeypatch):
    """形が変われば通るフラグも変わる。

    読取専用のファイル写像に要るのは 0x08 で、flags=0 は原理的に通らない。私的写像で
    コピーし終わった行は匿名メモリなので逆になる —— 通るのは flags=0 のほうで、0x08 は
    cudaDevAttrHostRegisterReadOnlySupported が 0 の機械では 801 を返す。その機械のために
    この形があるのだから、最初に投げるのは flags=0 でなければ 1 回ぶん無駄に断られる。
    """
    import freetoken.moe.mapped_bank as mb

    attempts = []

    class _Cudart:
        def cudaHostRegister(self, addr, nbytes, flags):
            attempts.append(flags)
            return 0 if flags == 0 else 1

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)

    bank = _bare_bank(private=True)
    bank._settle(offset=0, nbytes=4096, block_bytes=4096, register=True)

    assert attempts == [0], f"plain first on the private form, got {attempts}"
    assert bank.registered_bytes == 4096
    assert bank.fully_registered


def test_the_shared_form_still_asks_for_read_only_first(monkeypatch):
    import freetoken.moe.mapped_bank as mb

    attempts = []

    class _Cudart:
        def cudaHostRegister(self, addr, nbytes, flags):
            attempts.append(flags)
            return 0 if flags == 0x08 else 1

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)

    bank = _bare_bank()
    bank._settle(offset=0, nbytes=4096, block_bytes=4096, register=True)
    assert attempts == [0x08]


def test_the_form_is_chosen_by_what_the_device_accepts(tmp_path, monkeypatch):
    """写像の形は mmap の前に決めるしかないので、ファイルの 1 ページで先に訊く。

    属性 113 を読むのではなく実際に投げるのは、フラグを持っていると言いながらこのファイルを
    断る機械をその一言で拾えるため。
    """
    import freetoken.moe.mapped_bank as mb

    _, _, _, path = _built(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)

    for accepts, want_private in ((0x08, False), (0, False), (None, True)):
        class _Cudart:
            def cudaHostRegister(self, addr, nbytes, flags):
                return 0 if flags == accepts else 1

            def cudaHostUnregister(self, addr):
                return 0

        monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
        banks = MappedBanks(path, register=False)   # register=False: 形だけ見る
        try:
            assert banks.private is False, "register=False は問い合わせもしない"
        finally:
            banks.close()
        monkeypatch.setenv("FREETOKEN_BANK_MAP", "auto")
        banks = MappedBanks.__new__(MappedBanks)
        banks._fd = os.open(path, os.O_RDONLY)
        banks.map_mode = "auto"
        try:
            assert banks._pick_map_mode(register=True) is want_private, accepts
        finally:
            os.close(banks._fd)


def test_the_private_form_hands_back_the_same_bytes(tmp_path, monkeypatch):
    """コピーしたページの中身が元と 1 バイトも違わないこと。

    _cow は「読んだ値をそのまま書き戻す」でページを私的にする。書き戻しを間違えれば
    ページの先頭 1 バイトだけが壊れた重みになり、形も長さも合ったまま出力だけが濁る。
    """
    import freetoken.moe.mapped_bank as mb

    class _Cudart:
        def cudaHostRegister(self, addr, nbytes, flags):
            return 0 if flags == 0 else 1

        def cudaHostUnregister(self, addr):
            return 0

    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart())
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)
    monkeypatch.setenv("FREETOKEN_BANK_MAP", "private")
    freq = {0: [0, 9, 8, 0, 7, 0], 1: [5, 0, 0, 6, 0, 7], 2: [1, 2, 3, 4, 5, 6]}
    src, p, _, path = _built(tmp_path, freq=freq)
    # register=True: copying without registering would be cost for nothing, so the form is
    # only taken when something is going to be registered
    banks = MappedBanks(path)
    try:
        assert banks.private
        assert banks.fully_registered
        for name in src:
            for layer in (0, 1, 2):
                view = banks.sources[name][layer]
                for physical in range(6):
                    logical = p.order[layer][physical]
                    assert torch.equal(view[physical], src[name][layer][logical])
    finally:
        banks.close()


def test_a_short_mlock_is_reported_not_swallowed():
    """RLIMIT_MEMLOCK stops the lock partway and the run keeps going on evictable pages.

    The report this came from asked for 24 GiB per rank and got 7.8 (systemd's default limit
    is MAX(64M, RAM/8)); the summary line said "7.8 GiB locked resident" and explained
    nothing, so two thirds of the resident half was quietly page cache.
    """
    import ctypes as _ctypes

    class _Libc:
        def mlock(self, addr, span):
            _ctypes.set_errno(12)  # ENOMEM, what the limit gives
            return -1

    bank = _bare_bank(libc=_Libc())
    bank._settle(offset=0, nbytes=4096, block_bytes=4096, register=False)

    assert bank.requested_bytes == 4096, "what the placement wanted"
    assert bank.locked_bytes == 0, "what the OS gave"
    assert bank.lock_errno == 12
