"""What a start decides before the loader runs: read the checkpoint's experts, or not at all.

Items 3 (a), (c), (d) of guides/39: a bank file that already holds this rank's layers means the
checkpoint's expert tensors are never opened; a new placement is applied by reordering the file,
not by re-reading the checkpoint; a different layer split maps the same file; and a checkpoint
without expert tensors that cannot be served from the file says what is missing before anything
is loaded.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe.bank_file import BankFile, BankFileError, layout_from_sample  # noqa: E402
from freetoken.moe.mapped_bank import MappedTier  # noqa: E402

E, L = 6, 4


def _banks(layer):
    packed = torch.stack([torch.full((10,), layer * 16 + e, dtype=torch.uint8) for e in range(E)])
    scale = torch.stack([torch.full((2,), float(layer) + e / 8, dtype=torch.float16) for e in range(E)])
    return {"packed": packed, "scale": scale}


def _layout(meta=None):
    return layout_from_sample(_banks(0), range(L), E, meta or {"kind": "nvfp4", "kernel": "triton"})


def _tier(path, layers, **kw):
    kw.setdefault("layout", _layout())
    kw.setdefault("hot_per_layer", 2)
    return MappedTier(str(path), layers, all_layers=range(L), num_experts=E, **kw)


def _serve(tier):
    """What the engine does: prepare, stream the checkpoint only when asked, finish."""
    loaded = []
    if not tier.prepare():
        for i, layer in enumerate(tier.layers):
            loaded.append(layer)
            tier.sink(i, _banks(layer))
    tier.finish()
    return loaded


def _check_views(tier):
    for i, layer in enumerate(tier.layers):
        order = tier.placement.order[layer]
        src = _banks(layer)
        for name in src:
            view = tier.sources[name][i]
            for physical, logical in enumerate(order):
                assert torch.equal(view[physical], src[name][logical]), (layer, name, physical)


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    monkeypatch.setenv("FREETOKEN_BANK_REGISTER", "none")


# ----- (a) the checkpoint is not read once the file holds the layers ----------------------------
def test_the_second_start_reads_no_expert_tensor(tmp_path):
    path = tmp_path / "bank.ftmb"
    wanted = {l: list(reversed(range(E))) for l in range(L)}
    first = _tier(path, range(L), wanted=wanted)
    assert _serve(first) == [0, 1, 2, 3]
    first.banks.close()

    again = _tier(path, range(L), wanted=wanted)
    assert _serve(again) == []  # prepare() said the loader need not run
    assert again.ready and again.written == [] and again.reordered == []
    _check_views(again)
    again.banks.close()


def test_a_start_without_a_histogram_keeps_the_order_in_the_file(tmp_path):
    path = tmp_path / "bank.ftmb"
    wanted = {l: [5, 4, 3, 2, 1, 0] for l in range(L)}
    _serve(_tier(path, range(L), wanted=wanted))  # the first start writes it
    tier = _tier(path, range(L))  # no --moe-bank-stats this time
    assert _serve(tier) == []
    assert tier.placement.order == wanted and tier.reordered == []
    _check_views(tier)
    tier.banks.close()


def test_a_different_budget_rewrites_nothing(tmp_path):
    """The resident count is a property of the run, not of the file."""
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, range(L), hot_per_layer=2))
    tier = _tier(path, range(L), hot_per_layer=5)
    assert _serve(tier) == [] and tier.reordered == []
    assert tier.banks.hot_per_layer == 5
    tier.banks.close()


# ----- (c) a new placement reorders the file --------------------------------------------------
def test_a_new_histogram_reorders_in_place_without_the_checkpoint(tmp_path):
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, range(L)))
    new = {l: [(e + l + 1) % E for e in range(E)] for l in range(L)}
    tier = _tier(path, range(L), wanted=new, can_write=False)  # as if the checkpoint were packed
    assert _serve(tier) == []
    assert sorted(tier.reordered) == [0, 1, 2, 3]
    _check_views(tier)
    tier.banks.close()
    with BankFile.open(str(path)) as bank:
        for layer in range(L):
            assert bank.layer_state(layer)[0] == new[layer]
            bank.check_digest(layer)


# ----- (d) the layer split ------------------------------------------------------------------
def test_ranks_of_one_split_fill_one_file_and_another_split_reuses_it(tmp_path):
    path = tmp_path / "bank.ftmb"
    wanted = {l: [(5 * e + l) % E for e in range(E)] for l in range(L)}
    r0, r1 = _tier(path, [0, 1], wanted=wanted), _tier(path, [2, 3], wanted=wanted)
    assert _serve(r0) == [0, 1]
    assert _serve(r1) == [2, 3]
    r0.banks.close(), r1.banks.close()

    # --pp-layers moved the boundary: 3 + 1 layers, nothing is read or written
    a, b = _tier(path, [0, 1, 2], wanted=wanted), _tier(path, [3], wanted=wanted)
    assert _serve(a) == [] and _serve(b) == []
    _check_views(a), _check_views(b)
    a.banks.close(), b.banks.close()

    # and one GPU
    one = _tier(path, range(L), wanted=wanted)
    assert _serve(one) == []
    one.banks.close()


def test_only_the_missing_layers_are_written(tmp_path):
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, [0, 1]))
    tier = _tier(path, range(L))
    assert _serve(tier) == [0, 1, 2, 3]  # the loader streams the rank's layers ...
    assert sorted(tier.written) == [2, 3]  # ... and the file only takes the ones it lacked
    tier.banks.close()


# ----- a checkpoint without expert tensors ------------------------------------------------------
def test_no_file_says_so(tmp_path):
    with pytest.raises(BankFileError, match="no bank file"):
        _tier(tmp_path / "bank.ftmb", range(L), can_write=False).prepare()


def test_missing_layers_are_named(tmp_path):
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, [0, 1]))
    with pytest.raises(BankFileError, match=r"no layers 2-3 of this rank's 0-3"):
        _tier(path, range(L), can_write=False).prepare()


def test_a_file_for_another_kernel_is_not_started_over_when_it_is_the_only_copy(tmp_path):
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, range(L)))
    other = _layout({"kind": "nvfp4", "kernel": "marlin"})
    with pytest.raises(BankFileError, match="nvfp4 / triton experts; this run binds nvfp4 / marlin"):
        _tier(path, range(L), layout=other, can_write=False).prepare()
    with BankFile.open(str(path)) as bank:
        bank.set_canonical_for("/models/slim")
    with pytest.raises(BankFileError, match="only copy of the experts of /models/slim"):
        _tier(path, range(L), layout=other).prepare()
    with BankFile.open(str(path)) as bank:
        assert bank.present_layers() == [0, 1, 2, 3]  # still there


def test_an_ordinary_checkpoint_starts_a_mismatched_file_over(tmp_path):
    path = tmp_path / "bank.ftmb"
    _serve(_tier(path, range(L)))
    other = _layout({"kind": "nvfp4", "kernel": "triton", "fingerprint": "new"})
    tier = _tier(path, range(L), layout=other)
    assert tier.prepare() is False
    with BankFile.open(str(path)) as bank:
        assert bank.present_layers() == [] and bank.layout.meta["fingerprint"] == "new"


def test_a_checkpoint_downloaded_again_is_not_served_the_old_experts(tmp_path):
    """Same names and shapes, new shard mtimes: an ordinary checkpoint rewrites; a packed one has no shards to stamp."""
    path = tmp_path / "bank.ftmb"
    old = _layout({"kind": "nvfp4", "kernel": "triton", "fingerprint": "f", "source_stamp": "old"})
    _serve(_tier(path, range(L), layout=old))
    new = _layout({"kind": "nvfp4", "kernel": "triton", "fingerprint": "f", "source_stamp": "new"})
    assert _tier(path, range(L), layout=new).prepare() is False  # started over
    _serve(_tier(path, range(L), layout=old))
    packed = _layout({"kind": "nvfp4", "kernel": "triton", "fingerprint": "f"})
    tier = _tier(path, range(L), layout=packed, can_write=False)
    assert tier.prepare() is True


def test_not_enough_disk_is_said_before_writing(tmp_path, monkeypatch):
    import freetoken.moe.mapped_bank as mb

    monkeypatch.setattr(mb, "free_bytes", lambda _p: 10)
    with pytest.raises(BankFileError, match="needs .* GiB and the filesystem has"):
        _tier(tmp_path / "bank.ftmb", range(L)).prepare()


def test_a_write_that_fails_while_loading_names_the_bank_file(tmp_path, monkeypatch):
    """The engine reports anything but BankFileError from the loader as an unreadable checkpoint."""
    import errno

    tier = _tier(tmp_path / "bank.ftmb", range(L))
    assert tier.prepare() is False

    def full(*_a, **_kw):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(BankFile, "write_layer", full)
    with pytest.raises(BankFileError, match=r"writing .*bank\.ftmb failed: .*No space left") as exc:
        tier.sink(0, _banks(0))
    assert isinstance(exc.value.__cause__, OSError)


# ----- build_tier: flags to tier ----------------------------------------------------------------
class _Method:
    """The parts of a bound expert method build_tier reads."""

    def __init__(self, resident=False):
        from freetoken.layers.quantization import QuantKind
        from freetoken.layers.quantization.moe.base import BankSpec, MoEConfig

        self.kind, self.kernel = QuantKind.NVFP4, SimpleNamespace(name="triton")
        self.cfg = MoEConfig(num_experts=E, hidden=16, intermediate=8, top_k=2)
        self._layout = {"packed": BankSpec((10,), torch.uint8), "scale": BankSpec((2,), torch.float16)}
        if resident:
            self._layout["gate_up_alpha"] = BankSpec((), torch.bfloat16, resident=True)

    def layout(self):
        return self._layout


def _config(tmp_path, **kw):
    mc = SimpleNamespace(num_experts=E, num_moe_layers=kw.pop("local_layers", L), first_k_dense_replace=0)
    full = SimpleNamespace(num_experts=E, num_moe_layers=L)
    base = dict(
        model_path=str(tmp_path / "model"), model_config=mc, full_model_config=full,
        moe_bank_ram="120", moe_bank_stats=None, moe_bank_dir=str(tmp_path / "bankmap"),
        tp_info=SimpleNamespace(rank=0, size=1),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_tier_takes_this_ranks_window_and_global_histograms(tmp_path):
    from freetoken.moe.bank_tier import build_tier

    os.makedirs(tmp_path / "model")
    stats = tmp_path / "s.rank1.json"
    stats.write_text(json.dumps({"layer_range": [2, 4], "decode_freq": [[0, 0, 0, 0, 0, 9], [0, 9, 0, 0, 0, 0]]}), encoding="utf-8")
    pp = SimpleNamespace(bank_window=lambda fkd: (2, 4))
    # 12 B per expert; 120 B for 2 ranks -> 60 B per rank over 2 layers -> 2 experts resident
    cfg = _config(tmp_path, local_layers=2, moe_bank_stats=[str(stats)], tp_info=SimpleNamespace(rank=1, size=2))
    lines = []
    tier = build_tier(cfg, _Method(), pp=pp, log=lines.append)
    assert tier.layers == [2, 3] and tier.all_layers == [0, 1, 2, 3]
    assert tier.hot_per_layer == 2
    assert tier.wanted[2][0] == 5 and tier.wanted[3][0] == 1
    assert tier.layout.layers == [0, 1, 2, 3] and tier.layout.meta["kernel"] == "triton"
    assert tier.path == str(tmp_path / "bankmap" / "bank.ftmb") and tier.can_write


def test_build_tier_steps_aside_when_everything_fits(tmp_path):
    from freetoken.moe.bank_tier import build_tier

    assert build_tier(_config(tmp_path, moe_bank_ram="1G"), _Method(), log=lambda _m: None) is None


def test_a_packed_checkpoint_needs_the_flag_and_keeps_every_row_when_it_fits(tmp_path):
    from freetoken.moe.bank_pack import PACK_MANIFEST
    from freetoken.moe.bank_tier import build_tier

    model = tmp_path / "model"
    os.makedirs(model)
    (model / PACK_MANIFEST).write_text(json.dumps({"format": "freetoken-bank-pack", "bank": "bank.ftmb", "fingerprint": "f", "kind": "nvfp4", "kernel": "triton"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Serve it with --moe-bank-ram"):
        build_tier(_config(tmp_path, moe_bank_ram=None), _Method(), log=lambda _m: None)
    tier = build_tier(_config(tmp_path, moe_bank_ram="1G", moe_bank_dir=None), _Method(), log=lambda _m: None)
    assert tier is not None and tier.hot_per_layer == E and not tier.can_write
    assert tier.path == str(model / "bank.ftmb")
    assert tier.layout.meta["fingerprint"] == "f"
    with pytest.raises(ValueError, match="GPU-resident"):
        build_tier(_config(tmp_path, moe_bank_ram="1G"), _Method(resident=True), log=lambda _m: None)


def test_old_per_rank_files_are_pointed_out(tmp_path):
    from freetoken.moe.bank_tier import build_tier

    os.makedirs(tmp_path / "bankmap")
    (tmp_path / "bankmap" / "bank.rank0of2.ftmb").write_bytes(b"x" * 1024)
    lines = []
    build_tier(_config(tmp_path), _Method(), log=lines.append)
    assert any("no longer read" in line and "bank.rank0of2.ftmb" in line for line in lines)


def test_build_tier_resolves_auto_when_parse_args_did_not(tmp_path, monkeypatch):
    from freetoken.moe import disk_probe
    from freetoken.moe.bank_tier import build_tier

    # 14 B per expert x 6 x 4 layers = 336 B of banks; auto takes 120 B -> 2 resident per layer
    monkeypatch.setattr(disk_probe, "auto_bank_ram", lambda mem, ranks: SimpleNamespace(
        total_bytes=120, as_flag=lambda: "120", reason=lambda: "--moe-bank-ram auto: 120 B"))
    monkeypatch.setattr(disk_probe, "meminfo", lambda proc="/proc": {})
    lines = []
    cfg = _config(tmp_path, moe_bank_ram="auto")
    tier = build_tier(cfg, _Method(), log=lines.append)
    assert cfg.moe_bank_ram == "120" and "--moe-bank-ram auto: 120 B" in lines
    assert tier.hot_per_layer == 2


def test_build_tier_warns_about_the_bank_directory_once_and_passes_the_readahead_flag(tmp_path, monkeypatch):
    from freetoken.moe import disk_probe
    from freetoken.moe.bank_tier import build_tier

    seen = []
    monkeypatch.setattr(disk_probe, "storage_warnings", lambda d: seen.append(d) or [f"slow: {d}"])
    lines = []
    tier = build_tier(_config(tmp_path, moe_bank_readahead="auto"), _Method(), log=lines.append)
    assert seen == [str(tmp_path / "bankmap")] and f"slow: {tmp_path / 'bankmap'}" in lines
    assert tier.readahead == "auto" and tier.report_readahead

    seen.clear()
    tier = build_tier(_config(tmp_path, tp_info=SimpleNamespace(rank=1, size=2)), _Method(), log=lines.append)
    assert seen == [] and tier.readahead == "off" and not tier.report_readahead


def test_build_tier_refuses_to_create_a_bank_on_a_filesystem_that_cannot_hold_one(tmp_path, monkeypatch):
    from freetoken.moe import disk_probe
    from freetoken.moe.bank_tier import build_tier

    asked = []
    monkeypatch.setattr(disk_probe, "refuses_new_bank", lambda d: asked.append(d) or "not creating a bank file: drvfs")
    with pytest.raises(ValueError, match="not creating a bank file"):
        build_tier(_config(tmp_path, tp_info=SimpleNamespace(rank=1, size=2)), _Method(), log=lambda _m: None)
    assert asked == [str(tmp_path / "bankmap")] and not (tmp_path / "bankmap").exists()
    # a file somebody already put there is served (with the warning)
    os.makedirs(tmp_path / "bankmap")
    (tmp_path / "bankmap" / "bank.ftmb").write_bytes(b"")
    asked.clear()
    build_tier(_config(tmp_path), _Method(), log=lambda _m: None)
    assert asked == []


def test_without_a_histogram_only_layers_the_file_lacks_are_warned_about(tmp_path, _no_register):
    """Seen on the 3060s: a file holding all 48 layers in a histogram's order still drew "an arbitrary
    resident slice" on every start without the flag, which the docs tell you to leave off by then."""
    from freetoken.moe.bank_tier import build_tier

    os.makedirs(tmp_path / "model")
    infos, warns = [], []
    build_tier(_config(tmp_path), _Method(), log=infos.append, warn=warns.append)
    assert any("not in the bank file yet" in w for w in warns)

    os.makedirs(tmp_path / "bankmap", exist_ok=True)
    tier = _tier(tmp_path / "bankmap" / "bank.ftmb", range(L))
    _serve(tier)
    tier.banks.close()
    infos, warns = [], []
    build_tier(_config(tmp_path), _Method(), log=infos.append, warn=warns.append)
    assert not any("moe-bank-stats" in w for w in warns), warns
    assert any("keep the order already in" in line for line in infos), infos
