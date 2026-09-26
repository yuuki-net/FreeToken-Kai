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
from freetoken.moe.bank_file import BankFile, BankFileError, layout_from_sample  # noqa: E402
from freetoken.moe.mapped_bank import (  # noqa: E402
    ALIGN,
    MappedBankLayout,
    MappedBanks,
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


def _layout(src, layers):
    return layout_from_sample({n: src[n][0] for n in src}, layers, src["packed"][0].shape[0])


def _built(tmp_path, num_layers=3, num_experts=6, hot=4, freq=None):
    src = _sources(num_layers, num_experts)
    layers = list(range(num_layers))
    p = plan_placement(layers, num_experts, hot, freq)
    lay = _layout(src, layers)
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        for layer in layers:
            w.write_layer(layer, {n: src[n][layer] for n in src}, p.order[layer])
    return src, p, lay, path


# ----- layout ---------------------------------------------------------------------------
def test_blocks_are_aligned_and_do_not_overlap():
    src = _sources()
    lay = _layout(src, [0, 1, 2])
    assert lay.data_offset % ALIGN == 0 and lay.slot_offset % ALIGN == 0
    assert lay.data_offset == lay.slot_offset + 2 * lay.slot_bytes
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
    assert back.mismatch(lay) is None
    assert back.banks == lay.banks and back.layers == lay.layers


def test_the_order_is_not_part_of_the_geometry(tmp_path):
    """A new histogram reorders layers in place; it does not make the file another file."""
    src = _sources()
    a, b = _layout(src, [0, 1, 2]), _layout(src, [0, 1, 2])
    assert a.mismatch(b) is None and a.data_offset == b.data_offset


def test_mismatch_names_what_differs():
    src = _sources()
    a = layout_from_sample({n: src[n][0] for n in src}, [0, 1, 2], 6, {"kernel": "triton"})
    b = layout_from_sample({n: src[n][0] for n in src}, [0, 1, 2], 6, {"kernel": "marlin"})
    assert "kernel" in a.mismatch(b)
    c = _layout(src, [0, 1])
    assert "MoE layers" in _layout(src, [0, 1, 2]).mismatch(c)


def test_bad_magic_is_refused(tmp_path):
    _, _, _, path = _built(tmp_path)
    with open(path, "r+b") as f:
        f.write(b"XXXX")
    with pytest.raises(ValueError, match="not a mapped bank"):
        MappedBankLayout.read(path)


def test_a_per_rank_file_from_an_older_build_is_named_as_such(tmp_path):
    import struct

    path = tmp_path / "bank.rank0of2.ftmb"
    body = b'{"version":1}'
    path.write_bytes(struct.pack("<4sIQ", b"FTMB", 1, len(body)) + body)
    with pytest.raises(ValueError, match="older build"):
        MappedBankLayout.read(str(path))


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
    p = plan_placement(layers, 6, 4, {l: [l, 5, 4, 3, 2, 1] for l in layers})
    lay = _layout(src, layers)
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        ts = [
            threading.Thread(target=w.write_layer, args=(layer, {n: src[n][layer] for n in src}, p.order[layer]))
            for layer in layers
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert w.present_layers() == layers
    banks = MappedBanks(path, register=False)
    try:
        for layer in layers:
            for physical in range(6):
                logical = p.order[layer][physical]
                assert torch.equal(banks.sources["packed"][layer][physical],
                                   src["packed"][layer][logical])
    finally:
        banks.close()


def test_a_layer_never_committed_is_not_handed_out(tmp_path):
    src = _sources(num_layers=2, num_experts=4)
    lay = _layout(src, [0, 1])
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        w.write_layer(0, {n: src[n][0] for n in src}, [0, 1, 2, 3])
    # zeroed experts do not crash a model, so a layer that was never written must not be
    # mapped as if it had been
    with pytest.raises(BankFileError, match="not committed"):
        MappedBanks(path, register=False, layers=[0, 1])
    banks = MappedBanks(path, register=False)
    try:
        assert banks.layers == [0]
    finally:
        banks.close()


def test_everything_resident_is_still_a_valid_file(tmp_path):
    src, p, _, path = _built(tmp_path, hot=6)
    banks = MappedBanks(path, register=False, hot_per_layer=6)
    try:
        assert banks.hot_per_layer == 6
        assert torch.equal(banks.sources["packed"][0][0], src["packed"][0][p.order[0][0]])
    finally:
        banks.close()


def test_a_rank_maps_only_its_own_layers(tmp_path):
    """--pp-size: each rank hands out its window of the one file, from its first layer."""
    src, p, _, path = _built(tmp_path, num_layers=4)
    banks = MappedBanks(path, register=False, layers=[2, 3], hot_per_layer=2)
    try:
        assert banks.mapped_bytes == 2 * banks.layout.layer_bytes()
        # the mapping itself is only those layers' blocks: a private mapping is charged for its
        # whole length, and the whole file is twice a rank's share
        start, end = banks.layout.range_of([2, 3])
        assert len(banks._map) == end - start + ALIGN < banks.layout.total_bytes() - banks.layout.data_offset
        assert len(banks.sources["packed"]) == 2
        assert torch.equal(banks.sources["packed"][0][0], src["packed"][2][p.order[2][0]])
        assert torch.equal(banks.sources["scale"][1][5], src["scale"][3][p.order[3][5]])
    finally:
        banks.close()


def test_the_mapping_does_not_begin_inside_a_block(tmp_path):
    """Every view shares the mapping's storage, and Tensor.is_pinned() asks about the storage's
    first byte. A mapping that began at a registered resident prefix made every view read as
    pinned, and the whole-layer prefill copy sent the unregistered rows to an async copy: CUDA
    "invalid argument" on the first prefill of the 2060."""
    _, _, lay, path = _built(tmp_path, num_layers=4)
    for layers in ([0, 1, 2, 3], [2, 3]):
        banks = MappedBanks(path, register=False, layers=layers, hot_per_layer=2)
        try:
            # the storage begins one page before the rank's first block, on a page it never registers
            assert banks._map_offset + ALIGN == lay.range_of(layers)[0]
            assert banks._buf.data_ptr() + ALIGN == banks.sources["packed"][0].data_ptr()
        finally:
            banks.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_view_of_a_prefix_registered_block_is_not_pinned(tmp_path, monkeypatch):
    """The real thing: register the resident prefix and ask torch what the staged copy asks."""
    monkeypatch.setenv("FREETOKEN_BANK_MAP", "shared")
    src = {"packed": [torch.full((64, 131072), layer + 1, dtype=torch.uint8) for layer in range(2)]}
    lay = layout_from_sample({"packed": src["packed"][0]}, [0, 1], 64)
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        for layer in (0, 1):
            w.write_layer(layer, {"packed": src["packed"][layer]}, list(range(64)))
    banks = MappedBanks(path, layers=[0, 1], hot_per_layer=16)
    try:
        if not banks.fully_registered:
            pytest.skip("this device registers nothing")
        assert not banks.sources["packed"][0].is_pinned()
        from freetoken.moe.offload_cache import OffloadMoeCache

        class _Stage:  # just what the staged copy uses
            _staging = OffloadMoeCache._staging
            bank_reader = None  # the fault-in path: this test is about the registered prefix, not the parallel reads

        dst = torch.empty((64, 131072), dtype=torch.uint8, device="cuda")
        # what copy_missing does with a whole layer of a prefix-registered bank
        OffloadMoeCache._staged_h2d(_Stage(), dst, banks.sources["packed"][0])
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), src["packed"][0])
    finally:
        banks.close()


def test_staging_buffers_the_host_refuses_fall_back_to_the_plain_copy(monkeypatch):
    """43c: 登録が枠を使い切った後、最初の長いプロンプトでステージングの cudaHostAlloc が
    断られ、例外がスケジューラまで上がってサーバが落ちた。遅くても正しい普通のコピーに落とし、
    断られたことは覚えて 2 度目は頼まない。"""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    import freetoken.kernel.pinned as pinned
    from freetoken.moe.offload_cache import OffloadMoeCache

    asked = []

    def refuse(*shape, dtype):
        asked.append(shape)
        raise RuntimeError("cudaHostAlloc: out of memory")

    monkeypatch.setattr(pinned, "alloc_pinned_tensor", refuse)
    monkeypatch.delenv("FREETOKEN_STAGED_COPY", raising=False)

    class _Stage:
        _staging = OffloadMoeCache._staging
        bank_reader = None

    stage = _Stage()
    src = torch.arange(1 << 20, dtype=torch.int32).view(256, -1)
    for _ in range(2):
        dst = torch.empty_like(src, device="cuda")
        OffloadMoeCache._staged_h2d(stage, dst, src)
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), src)
    assert len(asked) == 1  # 1 本目で断られたら、それきり


def test_refused_staging_still_copies_a_prefix_registered_block(tmp_path, monkeypatch):
    """上の縮退を dst.copy_(src) で書いたら、実機で `CUDA error: invalid argument` で落ちた。
    登録済みの先頭と未登録の残りにまたがる 1 回のコピーをドライバが断る。自前の
    ページ可能メモリを経由して片ごとに運ぶ。"""
    import freetoken.kernel.pinned as pinned
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setenv("FREETOKEN_BANK_MAP", "shared")
    monkeypatch.setenv("FREETOKEN_STAGED_COPY_MB", "1")  # 1 ブロックを複数の片に分ける
    src = {"packed": [torch.full((64, 131072), layer + 1, dtype=torch.uint8) for layer in range(2)]}
    lay = layout_from_sample({"packed": src["packed"][0]}, [0, 1], 64)
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        for layer in (0, 1):
            w.write_layer(layer, {"packed": src["packed"][layer]}, list(range(64)))
    banks = MappedBanks(path, layers=[0, 1], hot_per_layer=16)
    try:
        if not banks.fully_registered:
            pytest.skip("this device registers nothing")

        def refuse(*shape, dtype):
            raise RuntimeError("cudaHostAlloc: out of memory")

        monkeypatch.setattr(pinned, "alloc_pinned_tensor", refuse)

        class _Stage:
            _staging = OffloadMoeCache._staging
            bank_reader = None

        dst = torch.empty((64, 131072), dtype=torch.uint8, device="cuda")
        OffloadMoeCache._staged_h2d(_Stage(), dst, banks.sources["packed"][0])
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), src["packed"][0])
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


def _tier(tmp_path, registered, hot=4, blocks=6, registered_blocks=None, layers=(0, 1, 2)):
    from freetoken.moe.mapped_bank import MappedTier

    _, p, _, path = _built(tmp_path, hot=hot)
    tier = MappedTier(path, list(range(3)), num_experts=6, hot_per_layer=hot)
    tier._orders = dict(p.order)
    tier.banks = MappedBanks(path, register=False, hot_per_layer=hot)
    tier.banks.registered_bytes = registered
    tier.banks.hot_blocks = blocks
    # every resident block registered unless the caller is asking for the partial case
    tier.banks.registered_blocks = (
        blocks if registered_blocks is None and registered else registered_blocks or 0
    )
    # the layers whose every block registered: what the GPU may address (attach keys on this)
    tier.banks.registered_layers = set(layers) if registered else set()
    return tier


def test_a_registered_prefix_keeps_the_pcie_fetch_and_bounds_it(tmp_path):
    tier = _tier(tmp_path, registered=1 << 20)  # every layer covered
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


def test_the_layers_the_budget_covered_keep_the_fetch_and_the_rest_do_not(tmp_path):
    """A pin budget that reaches two of three layers: those two fetch, the third is the engine's
    to route to the CPU (engine._mapped_cpu_layers reads the same set)."""
    tier = _tier(tmp_path, registered=1 << 20, registered_blocks=4, layers=(0, 1))
    cache = _FakeCache()
    said = []
    tier.log = said.append
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    assert cache.prefix_pinned_rows == 4  # the resident prefix is still the bound
    assert cache.hybrid_max_fetch == 512  # and the fetch path stays on for the covered layers
    assert any("2/3" in m for m in said), said


def test_no_layer_covered_sends_every_miss_to_the_cpu(tmp_path):
    tier = _tier(tmp_path, registered=0)
    cache = _FakeCache()
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    assert cache.prefix_pinned_rows is None and cache.hybrid_max_fetch == 0


def test_the_draft_head_layer_gets_the_identity_permutation(tmp_path):
    """--spec-mtp はキャッシュのバンクをもう 1 層ぶん伸ばす（engine._append_mtp_bank）。

    その層は配置が解かれたあとに足されるので再番号付けされておらず、行はチェックポイント
    そのままの順。つまり identity が正しい写像で、attach_offload_moe_cache にとっての
    identity は None である。

    埋めないと、ドラフトヘッドの層がこのリストの終端を越えて索き、
    **ヘッドを載せているランクだけが起動に失敗する**（`--pp-size 2` なら rank 1 だけ）。
    2060 と 3060 の両方で `--moe-bank-ram` + `--spec-mtp` が
    `IndexError: list index out of range` で死んでいた。
    """
    tier = _tier(tmp_path, registered=1 << 20)
    cache = _FakeCache()
    # 配置は 3 層、バンクは 4 層（末尾がドラフトヘッドの層）
    cache.bank_sources = {"packed": [None] * 4, "scale": [None] * 4}
    try:
        tier.attach(cache, device="cpu")
    finally:
        tier.banks.close()
    assert len(cache.expert_perm) == 4
    assert all(p is not None for p in cache.expert_perm[:3])
    assert cache.expert_perm[3] is None


class _FakeCudart:
    """cudaHostRegister that takes the first ``ok`` calls and refuses the rest, like a host whose
    page-lock budget runs out partway through the blocks."""

    def __init__(self, ok):
        self.ok = ok
        self.calls = 0
        self.refusals = 0
        self.unregistered = []

    def cudaHostRegister(self, addr, nbytes, flags):
        self.calls += 1
        if self.calls <= self.ok:
            return 0
        self.refusals += 1
        return 2  # any non-zero is a refusal

    def cudaHostUnregister(self, addr):  # close() gives the registrations back
        self.unregistered.append(addr)
        return 0


def _registered_bank(monkeypatch, tmp_path, *, ok=10**9, budget=None, hot=4, keep_free=0):
    import torch

    from freetoken.moe import mapped_bank as mb

    fake = _FakeCudart(ok)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    monkeypatch.setattr(mb, "clear_cuda_error", lambda: None)
    _, _placement, _, path = _built(tmp_path, hot=hot)
    return mb.MappedBanks(path, hot_per_layer=hot, register_budget=budget, keep_free=keep_free), fake


def test_registration_counts_whole_layers_and_stops_at_the_first_refusal(monkeypatch, tmp_path):
    """`prefix_pinned_rows` は「行 [0, hot) はデバイスから触れる」という主張で、その層の
    ブロックが 1 つでも登録に失敗していれば嘘になる。嘘の先は decode graph の中の illegal
    access なので、**層単位で**全ブロック揃ったものだけを数える。

    登録は層ごとに進む（種類ごとではない）。種類ごとだと途中で止まった時点でどの層も
    ブロックが揃わず、使える層がゼロになる（FreeToken-Kai#2）。
    """
    banks, _ = _registered_bank(monkeypatch, tmp_path)
    try:
        per_layer = len(banks.layout.banks)  # このジオメトリの 1 層あたりのブロック数
        assert banks.registered_layers == {0, 1, 2} and banks.fully_registered
    finally:
        banks.close()

    # 層 0 は揃い、その次の層で断られ、以降は試されない。
    # 「試されない」は拒否の回数で見る: 打ち切らなければ残りのブロックぶん拒否が並ぶ。
    banks, fake = _registered_bank(monkeypatch, tmp_path, ok=per_layer + 1)
    try:
        assert banks.registered_layers == {0}  # 揃ったのは層 0 だけ
        assert banks.registered_blocks >= per_layer  # 層 0 のぶんは登録できている
        assert fake.refusals <= 2  # 断られたブロック 1 つ（フラグ 2 通りまで）で止まる
        assert not banks.fully_registered
    finally:
        banks.close()


def test_a_budget_stops_at_a_layer_boundary_without_asking(monkeypatch, tmp_path):
    """予算が 2 層ぶんなら 2 層で止める。断られてから気づくのではなく**先に数える**
    （断られた試行はどれも CUDA のエラースロットを掃除しないといけないし、
    使えない層の中に登録済みブロックを残すだけになる）。"""
    banks, _ = _registered_bank(monkeypatch, tmp_path)
    try:
        per_layer_blocks = len(banks.layout.banks)
        per_layer_bytes = banks.registered_bytes // 3
    finally:
        banks.close()

    banks, fake = _registered_bank(monkeypatch, tmp_path, budget=2 * per_layer_bytes)
    try:
        assert banks.registered_layers == {0, 1}
        assert banks.registered_bytes == 2 * per_layer_bytes
        assert banks.registered_blocks == 2 * per_layer_blocks
        assert fake.refusals == 0  # 断られてから気づくのではなく、先に数えて止める
    finally:
        banks.close()


def _per_layer(monkeypatch, tmp_path):
    """(1 層あたりのブロック数, 1 層あたりの登録バイト数) -- 全部登録できる状態で測る。

    shared 形式に固定する: auto は最初に 1 回登録して外す探り（_pick_map_mode）を入れるので、
    登録と解除の回数が 1 ずつずれ、端数の層を狙って作れない。"""
    monkeypatch.setenv("FREETOKEN_BANK_MAP", "shared")
    banks, _ = _registered_bank(monkeypatch, tmp_path)
    try:
        return len(banks.layout.banks), banks.registered_bytes // 3
    finally:
        banks.close()


def test_a_refusal_unregisters_the_layer_it_left_short(monkeypatch, tmp_path):
    """43c: 断られた層の登録済みブロックは GPU が使わない（層単位でしか数えない）のに枠だけ
    食っていた。返せば、その分は後から固定するもの（読み込み・ステージング）に回る。"""
    blocks, _ = _per_layer(monkeypatch, tmp_path)
    banks, fake = _registered_bank(monkeypatch, tmp_path, ok=2 * blocks + 1)
    try:
        assert banks.registered_layers == {0, 1}
        assert banks.registered_blocks == 2 * blocks  # 層 2 の 1 ブロックは外した
        assert len(fake.unregistered) == 1
        assert banks.given_back_layers == 0  # keep_free=0 なので揃った層は返さない
    finally:
        banks.close()


def test_a_refusal_gives_back_whole_layers_until_keep_free_is_free(monkeypatch, tmp_path):
    """見積もりの予算より手前で断られたら、枠は使い切られている。そのままだと最初の長い
    プロンプトでステージングの cudaHostAlloc が断られてスケジューラが落ちた（偽の 1 GiB 上限で
    実測）。後ろの層から keep_free 分を返す。返した層は CPU でデコードされる。"""
    blocks, layer_bytes = _per_layer(monkeypatch, tmp_path)
    banks, fake = _registered_bank(monkeypatch, tmp_path, ok=2 * blocks + 1, keep_free=layer_bytes)
    try:
        # 層 2 の端数（1 ブロック）では足りないので層 1 も返す
        assert banks.registered_layers == {0}
        assert banks.registered_bytes == layer_bytes
        assert banks.given_back_layers == 1
        assert banks.given_back_bytes >= layer_bytes
        assert len(fake.unregistered) == 1 + blocks
        assert banks.refused
    finally:
        banks.close()


def test_the_budget_stopping_first_gives_nothing_back(monkeypatch, tmp_path):
    """予算で止まったのなら断られていない: 予約は予算の側で既に引いてあるので返さない。"""
    blocks, layer_bytes = _per_layer(monkeypatch, tmp_path)
    banks, fake = _registered_bank(monkeypatch, tmp_path, budget=2 * layer_bytes,
                                   keep_free=10 * layer_bytes)
    try:
        assert banks.registered_layers == {0, 1}
        assert not banks.refused and banks.given_back_bytes == 0
        assert fake.unregistered == []
    finally:
        banks.close()


def test_the_registrations_count_in_the_process_total(monkeypatch, tmp_path):
    """43c: host_banks.registered_bytes() はプロセス全体の固定量として PinFailed の文面と
    記憶する上限（remember）に使われるのに、マッピングの登録が入っていなかった。後から
    バンクが断られると、その分だけ小さい上限が記憶された。"""
    from freetoken.moe import host_banks

    base = host_banks.registered_bytes()
    blocks, _ = _per_layer(monkeypatch, tmp_path)
    assert host_banks.registered_bytes() == base  # close() で戻る
    banks, _ = _registered_bank(monkeypatch, tmp_path, ok=2 * blocks + 1)
    try:
        assert host_banks.registered_bytes() - base == banks.registered_bytes  # 返した分も引く
    finally:
        banks.close()
    assert host_banks.registered_bytes() == base


def test_what_is_left_for_after_the_banks(monkeypatch):
    """登録予算から外す量 = 並列読み込み（2 組 x スレッド x 片）+ ステージング 2 本 + 余裕。
    どちらも切れば余裕だけ。"""
    from freetoken.moe import mapped_bank as mb
    from freetoken.moe.bank_reader import ALIGN

    for name in ("FREETOKEN_BANK_PREAD", "FREETOKEN_BANK_READ_THREADS",
                 "FREETOKEN_BANK_READ_PIECE_MB", "FREETOKEN_STAGED_COPY", "FREETOKEN_STAGED_COPY_MB"):
        monkeypatch.delenv(name, raising=False)
    assert mb.pinned_after_banks() == (
        2 * 8 * ((16 << 20) + 3 * ALIGN) + 2 * (32 << 20) + mb.PIN_AFTER_BANKS_MARGIN
    )
    monkeypatch.setenv("FREETOKEN_BANK_READ_THREADS", "4")
    monkeypatch.setenv("FREETOKEN_STAGED_COPY_MB", "8")
    assert mb.pinned_after_banks() == (
        2 * 4 * ((16 << 20) + 3 * ALIGN) + 2 * (8 << 20) + mb.PIN_AFTER_BANKS_MARGIN
    )
    monkeypatch.setenv("FREETOKEN_BANK_PREAD", "0")
    monkeypatch.setenv("FREETOKEN_STAGED_COPY", "0")
    assert mb.pinned_after_banks() == mb.PIN_AFTER_BANKS_MARGIN


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
        banks = MappedBanks(path, register=False, hot_per_layer=5)
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
    bank._base, bank._libc, bank._map_offset = 0x1000, libc, 0
    bank._map = _mmap.mmap(-1, 1 << 20)  # a real mapping: _settle advises it before locking
    bank.locked_bytes = bank.requested_bytes = bank.registered_bytes = 0
    bank.lock_errno = 0
    bank._layer_regs, bank._pos = {}, 0
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
    # only taken when something is going to be registered. Layers 1-2: the mapping starts
    # mid-file, so the copy and the page-cache release must agree on where a block is.
    banks = MappedBanks(path, layers=[1, 2], hot_per_layer=3)
    try:
        assert banks.private
        assert banks.fully_registered
        for name in src:
            for i, layer in enumerate((1, 2)):
                view = banks.sources[name][i]
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


# ----- --moe-bank-readahead -------------------------------------------------------------
def _tier_with_window(tmp_path, monkeypatch, mode, kb=8192, writable=True, report=True):
    """A MappedTier whose bank's widest row block is 1600 kB (Flash-Next), on a device whose
    read_ahead_kb is a plain file under tmp_path."""
    import freetoken.moe.mapped_bank as mb

    ra = tmp_path / "queue" / "read_ahead_kb"
    if writable:
        ra.parent.mkdir()
        ra.write_text(f"{kb}\n")
    where = str(ra)
    monkeypatch.setattr(mb, "readahead_kb", lambda path: (kb, where))
    logs, warns = [], []
    layout = MappedBankLayout(8, [0], [("gate_up", (1,), "uint8", 1600 * 1024),
                                       ("scale", (1,), "uint8", 200 * 1024)], {})
    tier = mb.MappedTier(str(tmp_path / "bank.ftmb"), [0], num_experts=8, hot_per_layer=4, layout=layout,
                         log=logs.append, warn=warns.append, readahead=mode, report_readahead=report)
    return tier, ra, logs, warns


def test_readahead_off_only_reports_and_names_the_recommended_window(tmp_path, monkeypatch):
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "off")
    tier._apply_readahead()
    assert ra.read_text().strip() == "8192"
    (msg,) = warns
    assert "widest expert-row block (1600 kB)" in msg
    assert f"echo 256 | sudo tee {ra}" in msg
    assert "--moe-bank-readahead auto" in msg


def test_readahead_auto_writes_the_recommended_window(tmp_path, monkeypatch):
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "auto")
    tier._apply_readahead()
    assert ra.read_text().strip() == "256"
    assert not warns
    assert any("8192 -> 256 kB" in m and "left set after exit" in m for m in logs)


def test_readahead_value_is_written_as_given(tmp_path, monkeypatch):
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "2048")
    tier._apply_readahead()
    assert ra.read_text().strip() == "2048"


def test_readahead_that_cannot_be_written_logs_the_command_once(tmp_path, monkeypatch):
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "auto", writable=False)
    tier._apply_readahead()
    (msg,) = warns
    assert "could not write" in msg and "echo 256 | sudo tee" in msg


def test_a_window_inside_the_widest_block_is_one_info_line(tmp_path, monkeypatch):
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "off", kb=256)
    tier._apply_readahead()
    assert not warns
    assert logs == [f"--moe-bank-ram: device readahead 256 kB ({ra})"]


def test_a_later_rank_sets_the_window_for_its_own_mapping_and_says_nothing(tmp_path, monkeypatch):
    """One bank file for every rank, so one device: each rank opens its own mapping, which takes
    the window at open, so each sets it -- and only the first reports."""
    tier, ra, logs, warns = _tier_with_window(tmp_path, monkeypatch, "auto", report=False)
    tier._apply_readahead()
    assert ra.read_text().strip() == "256"
    assert logs == [] and warns == []
    (tmp_path / "later").mkdir()
    tier, ra, logs, warns = _tier_with_window(tmp_path / "later", monkeypatch, "off", report=False)
    tier._apply_readahead()  # 8192 is too wide, but the first rank has already said so
    assert ra.read_text().strip() == "8192" and logs == [] and warns == []
