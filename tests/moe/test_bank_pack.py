"""``ft bank pack`` / ``verify`` on small synthetic checkpoints: the bank file as the only copy.

The claim ``pack`` prints -- the original can be deleted -- rests on two checks it makes before
it says so, and these tests check the checks: every removed tensor comes back from the bank file
byte for byte, and the bank's rows are what the loader would pack from the original. Then the
packed checkpoint is served from the bank (``MappedTier`` with no checkpoint experts) and its
views are compared with rows built from the ORIGINAL files, independently of ``pack``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

import freetoken.moe.expert_pieces as expert_pieces  # noqa: E402
from freetoken.layers.quantization.moe.base import MoEConfig  # noqa: E402
from freetoken.layers.quantization.moe.mxfp4 import TritonGptossMxfp4MoEKernel  # noqa: E402
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel  # noqa: E402
from freetoken.models.nvfp4_banks import (  # noqa: E402
    Nvfp4ExpertSourceSpec,
    iter_nvfp4_expert_pieces,
    nvfp4_expert_sources,
)
from freetoken.moe.bank_file import BankFile, BankFileError, layout_from_specs  # noqa: E402
from freetoken.moe.bank_pack import (  # noqa: E402
    PackError,
    ShardHeader,
    bank_path_for,
    fingerprint,
    pack_checkpoint,
    read_pack_manifest,
    unpack_checkpoint,
    verify_packed,
)
from freetoken.moe.mapped_bank import MappedTier  # noqa: E402

L, E, H, I = 3, 4, 32, 16
FP8 = torch.float8_e4m3fn

SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="test NVFP4 experts",
)
CONFIG = SimpleNamespace(num_moe_layers=L, num_experts=E, num_layers=L, first_k_dense_replace=0)
CFG = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_BANK_REGISTER", "none")

    def sources(model_path, config, kind, *, weight_map=None):
        from freetoken.layers.quantization import QuantKind

        if kind is QuantKind.MXFP4:
            from freetoken.models.gpt_oss.weight import expert_sources as gptoss

            return gptoss(model_path, GPTOSS_CONFIG, kind, weight_map=weight_map)
        return nvfp4_expert_sources(model_path, config, SPEC, weight_map=weight_map)

    monkeypatch.setattr(expert_pieces, "expert_sources", sources)


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _nvfp4_checkpoint(path):
    """Three shards: dense only (hard-linked), dense + experts (rewritten), experts only (left out)."""
    os.makedirs(path)
    dense = {
        "model.embed_tokens.weight": torch.randn(50, H, generator=_g(1)).to(torch.bfloat16),
        "model.norm.weight": torch.randn(H, generator=_g(2)),
    }
    mixed, experts_only = {}, {}
    seed = 10
    for layer in range(L):
        mixed[f"model.layers.{layer}.self_attn.qkv.weight"] = torch.randn(H, H, generator=_g(seed))
        # an expert tensor the reader does not use: it must stay
        mixed[f"model.layers.{layer}.mlp.experts.0.gate_proj.input_scale"] = torch.tensor(0.5)
        for e in range(E):
            for proj, (rows, cols) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                seed += 1
                base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                target = mixed if layer < 2 else experts_only
                target[base + ".weight"] = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=_g(seed))
                target[base + ".weight_scale"] = torch.randint(0, 120, (rows, cols // 16), dtype=torch.uint8, generator=_g(seed + 1000)).view(FP8)
                # the global scales stay in the checkpoint, so they go where a shard survives
                mixed[base + ".weight_scale_2"] = torch.rand((), generator=_g(seed + 2000)) + 0.1
    mixed["mtp.layers.0.mlp.experts.0.gate_proj.weight"] = torch.randn(I, H, generator=_g(3))
    shards = {"model-1.safetensors": dense, "model-2.safetensors": mixed, "model-3.safetensors": experts_only}
    weight_map = {}
    for shard, tensors in shards.items():
        safetensors_torch.save_file(tensors, os.path.join(path, shard), metadata={"format": "pt"})
        weight_map.update({n: shard for n in tensors})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1234567}, "weight_map": weight_map}, f)
    for name, text in (("config.json", '{"architectures": ["Test"]}'), ("tokenizer.json", "{}")):
        with open(os.path.join(path, name), "w") as f:
            f.write(text)
    os.makedirs(os.path.join(path, "assets"))
    with open(os.path.join(path, "assets", "logo.txt"), "w") as f:
        f.write("x")
    return weight_map


def _headers(path, weight_map):
    heads = {s: ShardHeader.read(os.path.join(path, s)) for s in set(weight_map.values())}
    return lambda n: (heads[weight_map[n]].entries[n]["dtype"], heads[weight_map[n]].entries[n]["shape"])


def _meta(kind, kernel, cfg, fp):
    return {
        "kind": kind, "kernel": kernel, "fingerprint": fp,
        "kernel_cfg": {"num_experts": cfg.num_experts, "hidden": cfg.hidden, "intermediate": cfg.intermediate, "tp_rank": 0, "tp_size": 1},
    }


def _nvfp4_banks_from_reader(model):
    """The layers as the real loader packs them (iter_nvfp4_expert_pieces + the triton kernel)."""
    kernel = TritonNvfp4MoEKernel()
    specs = kernel.layout(CFG)
    banks = {l: {n: torch.zeros((E, *s.shape), dtype=s.dtype) for n, s in specs.items()} for l in range(L)}
    for layer, e0, e1, piece in iter_nvfp4_expert_pieces(model, CONFIG, SPEC):
        kernel.pack(piece, CFG, {n: banks[layer][n][e0:e1] for n in specs})
    return kernel, specs, banks


def _served_nvfp4_bank(tmp_path, order=None, fp=None):
    """An original checkpoint and the bank file a --moe-bank-ram start would have written."""
    model = str(tmp_path / "orig")
    weight_map = _nvfp4_checkpoint(model)
    kernel, specs, banks = _nvfp4_banks_from_reader(model)
    fp = fp or fingerprint(nvfp4_expert_sources(model, CONFIG, SPEC), _headers(model, weight_map))
    layout = layout_from_specs(specs, range(L), E, _meta("nvfp4", "triton", CFG, fp))
    bank_path = str(tmp_path / "cache" / "bank.ftmb")
    order = order or {l: [(3 * e + l) % E for e in range(E)] for l in range(L)}
    tier = MappedTier(bank_path, range(L), num_experts=E, hot_per_layer=2, wanted=order, layout=layout)
    assert tier.prepare() is False
    for layer in range(L):
        tier.sink(layer, banks[layer])
    tier.finish()
    tier.banks.close()
    return model, weight_map, bank_path, banks, order


def _snapshot(folder):
    out = {}
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, folder)] = hashlib.sha256(fh.read()).hexdigest()
    return out


# ----- pack ----------------------------------------------------------------------------------
def test_pack_removes_only_what_the_bank_gives_back_and_touches_nothing_else(tmp_path):
    model, weight_map, bank_path, banks, order = _served_nvfp4_bank(tmp_path)
    before = _snapshot(model)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)

    assert _snapshot(model) == before, "the original checkpoint must not change"
    removed = {n for n in weight_map if SPEC.key_pattern.match(n) and not n.endswith("weight_scale_2")}
    assert report["removed_tensors"] == len(removed) == L * E * 6

    # shards: dense-only linked, mixed rewritten, experts-only left out
    assert os.stat(os.path.join(slim, "model-1.safetensors")).st_ino == os.stat(os.path.join(model, "model-1.safetensors")).st_ino
    assert not os.path.exists(os.path.join(slim, "model-3.safetensors"))
    kept = safetensors_torch.load_file(os.path.join(slim, "model-2.safetensors"))
    original = safetensors_torch.load_file(os.path.join(model, "model-2.safetensors"))
    assert set(kept) == set(original) - removed
    for name in kept:
        assert torch.equal(kept[name].reshape(-1).view(torch.uint8), original[name].reshape(-1).view(torch.uint8))
    assert "model.layers.0.mlp.experts.0.gate_proj.input_scale" in kept  # not a bank source
    assert "model.layers.0.mlp.experts.1.down_proj.weight_scale_2" in kept  # lossy in the bank

    with open(os.path.join(slim, "model.safetensors.index.json")) as f:
        index = json.load(f)
    assert set(index["weight_map"]) == set(weight_map) - removed
    assert "model-3.safetensors" not in index["weight_map"].values()
    for meta_file in ("config.json", "tokenizer.json", os.path.join("assets", "logo.txt")):
        assert os.path.isfile(os.path.join(slim, meta_file))

    # the bank moved in and knows it is the only copy
    assert not os.path.exists(bank_path)
    manifest = read_pack_manifest(slim)
    assert bank_path_for(slim, manifest) == os.path.join(slim, "bank.ftmb") == report["bank"]
    with BankFile.open(report["bank"]) as bank:
        assert bank.canonical_for() == slim

    # whole-file hashes of what was rewritten or left out: enough to put the originals back
    for shard in ("model-2.safetensors", "model-3.safetensors"):
        with open(os.path.join(model, shard), "rb") as f:
            assert manifest["shards"][shard]["sha256"] == hashlib.sha256(f.read()).hexdigest()


def test_a_mark_whose_checkpoint_is_still_there_blocks_a_second_pack(tmp_path):
    """One bank backs one packed checkpoint: packing again would leave the first unservable."""
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    pack_checkpoint(model, str(tmp_path / "slim"), config=CONFIG, bank_path=bank_path,
                    keep_bank=True, log=lambda _m: None)
    with pytest.raises(PackError, match="already the only copy"):
        pack_checkpoint(model, str(tmp_path / "slim2"), config=CONFIG, bank_path=bank_path,
                        keep_bank=True, log=lambda _m: None)


def test_a_mark_whose_checkpoint_is_gone_is_dropped_instead_of_blocking_pack(tmp_path):
    """A --keep-bank pack whose packed checkpoint is later deleted used to refuse every later
    pack, with no command to clear the mark (hit twice on the same host, guides/39 8.4)."""
    import shutil

    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, keep_bank=True, log=lambda _m: None)
    shutil.rmtree(slim)

    said = []
    second = str(tmp_path / "slim2")
    pack_checkpoint(model, second, config=CONFIG, bank_path=bank_path, keep_bank=True, log=said.append)
    assert any("no longer exists" in m and "dropping that mark" in m for m in said)
    with BankFile.open(bank_path) as bank:
        assert bank.canonical_for() == second  # and the mark now points at the pack that exists


def test_a_dry_run_reads_no_tensor_and_writes_nothing(tmp_path, monkeypatch):
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    import freetoken.moe.bank_pack as bp

    monkeypatch.setattr(bp, "_SlimShard", None)  # would fail if anything were written
    slim = str(tmp_path / "new" / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, dry_run=True, log=lambda _m: None)
    assert report["plan"] == {"model-1.safetensors": "link", "model-2.safetensors": "rewrite", "model-3.safetensors": "drop"}
    assert report["removed_tensors"] == L * E * 6
    assert not os.path.exists(tmp_path / "new") and os.path.exists(bank_path)


def test_the_bank_regenerates_every_removed_tensor_byte_for_byte(tmp_path):
    """Independent of pack's own check: unpack the moved bank and compare with the original files."""
    model, weight_map, bank_path, _, order = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    kernel = TritonNvfp4MoEKernel()
    originals = {}
    for shard in ("model-2.safetensors", "model-3.safetensors"):
        originals.update(safetensors_torch.load_file(os.path.join(model, shard)))
    with BankFile.open(report["bank"]) as bank:
        checked = 0
        for layer in range(L):
            got_order, _ = bank.layer_state(layer)
            assert got_order == order[layer]
            rows = {}
            for name, shape, dtype, row_bytes in bank.layout.banks:
                physical = bank.read_block(name, layer).view(E, row_bytes)
                logical = torch.empty_like(physical)
                logical[torch.as_tensor(got_order)] = physical  # physical p holds logical order[p]
                rows[name] = logical.view(getattr(torch, dtype)).view(E, *shape)
            unpacked = kernel.unpack(rows, CFG)
            for e in range(E):
                for proj, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                    for kind, suffix in (("weight", ""), ("weight_scale", "_scale")):
                        name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.{kind}"
                        want = originals[name].contiguous().reshape(-1).view(torch.uint8)
                        got = unpacked[role + suffix][e].contiguous().reshape(-1).view(torch.uint8)
                        assert torch.equal(got, want), name
                        checked += 1
    assert checked == report["removed_tensors"]


def test_the_packed_checkpoint_is_served_from_the_bank_alone(tmp_path):
    model, _, bank_path, banks, order = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    manifest = read_pack_manifest(slim)
    layout = layout_from_specs(TritonNvfp4MoEKernel().layout(CFG), range(L), E,
                               _meta("nvfp4", "triton", CFG, manifest["fingerprint"]))
    for layers in ([0, 1, 2], [0], [1, 2]):  # one GPU, then a two-rank split
        tier = MappedTier(report["bank"], layers, all_layers=range(L), num_experts=E, hot_per_layer=3,
                          layout=layout, can_write=False)
        assert tier.prepare() is True
        tier.finish()
        for i, layer in enumerate(layers):
            for name, per_layer in tier.sources.items():
                for physical, logical in enumerate(order[layer]):
                    assert torch.equal(per_layer[i][physical].reshape(-1).view(torch.uint8),
                                       banks[layer][name][logical].reshape(-1).view(torch.uint8))
        tier.banks.close()


def test_verify_passes_and_then_catches_a_flipped_byte(tmp_path):
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    assert verify_packed(slim, config=CONFIG, log=lambda _m: None)["removed_tensors"] == L * E * 6
    with BankFile.open(report["bank"]) as bank:
        off = bank.layout.offset_of("down", 1) + 3
        os.pwrite(bank._fd, bytes([os.pread(bank._fd, 1, off)[0] ^ 1]), off)
    with pytest.raises(BankFileError, match="damaged"):
        verify_packed(slim, config=CONFIG, log=lambda _m: None)


def test_verify_catches_a_bank_whose_hashes_were_recommitted_over_other_bytes(tmp_path):
    """A block that changed AND got a fresh hash still has to give back the original bytes."""
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    with BankFile.open(report["bank"]) as bank:
        order, digest = bank.layer_state(2)
        block = bank.read_block("gate_up", 2)
        block[7] ^= 0xFF
        os.pwrite(bank._fd, block.numpy().tobytes(), bank.layout.offset_of("gate_up", 2))
        digest["gate_up"] = hashlib.sha256(block.numpy().tobytes()).hexdigest()
        bank._commit_layer(2, order, digest)
    with pytest.raises(PackError, match="layer 2: the bank no longer gives back"):
        verify_packed(slim, config=CONFIG, log=lambda _m: None)


def test_pack_refuses_a_bank_that_does_not_match_the_checkpoint(tmp_path):
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    # same geometry, same fingerprint, one expert row different: only the content check sees it
    with BankFile.open(bank_path) as bank:
        order, digest = bank.layer_state(0)
        block = bank.read_block("down_global", 0)
        block[0] ^= 0x10
        os.pwrite(bank._fd, block.numpy().tobytes(), bank.layout.offset_of("down_global", 0))
        digest["down_global"] = hashlib.sha256(block.numpy().tobytes()).hexdigest()
        bank._commit_layer(0, order, digest)
    with pytest.raises(PackError, match="layer 0 bank 'down_global': expert 0 in the bank file is not what the original checkpoint packs to"):
        pack_checkpoint(model, str(tmp_path / "slim"), config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    assert os.path.exists(bank_path)  # not moved
    assert read_pack_manifest(str(tmp_path / "slim")) is None


def test_pack_refuses_an_incomplete_bank_and_a_foreign_fingerprint(tmp_path):
    model, weight_map, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    with BankFile.open(bank_path) as bank:
        bank._forget_layer(1)
    with pytest.raises(PackError, match="lacks layers 1"):
        pack_checkpoint(model, str(tmp_path / "slim"), config=CONFIG, bank_path=bank_path, log=lambda _m: None)

    other = tmp_path / "other"
    other.mkdir()
    model2, _, bank2, _, _ = _served_nvfp4_bank(other, fp="written-from-some-other-checkpoint")
    with pytest.raises(PackError, match="was not written from the expert tensors"):
        pack_checkpoint(model2, str(other / "slim"), config=CONFIG, bank_path=bank2, log=lambda _m: None)


def test_pack_will_not_write_into_the_checkpoint_or_a_full_directory(tmp_path):
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    with pytest.raises(PackError, match="outside the checkpoint"):
        pack_checkpoint(model, os.path.join(model, "slim"), config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    with pytest.raises(PackError, match="not an empty directory"):
        pack_checkpoint(model, str(full), config=CONFIG, bank_path=bank_path, log=lambda _m: None)


def test_a_packed_checkpoint_refuses_to_start_without_the_bank_flag(tmp_path):
    from freetoken.moe.bank_pack import check_served_packed

    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    with pytest.raises(PackError, match="--moe-bank-ram"):
        check_served_packed(slim, moe_bank_ram=None, offload=True)
    with pytest.raises(PackError, match="--moe-bank-ram"):
        check_served_packed(slim, moe_bank_ram="8G", offload=False)
    assert check_served_packed(slim, moe_bank_ram="8G", offload=True) is not None
    assert check_served_packed(model, moe_bank_ram=None, offload=False) is None


def test_reordering_a_packed_bank_keeps_it_verifiable(tmp_path):
    """The pack record is about bytes per expert, not about where the rows sit."""
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    with BankFile.open(report["bank"]) as bank:
        for layer in range(L):
            bank.reorder_layer(layer, list(reversed(range(E))))
    verify_packed(slim, config=CONFIG, log=lambda _m: None)


def test_unpack_gives_back_the_original_files_byte_for_byte(tmp_path):
    """The way back: a packed checkpoint plus its bank is the whole original, not an approximation."""
    order = {l: [(e + 2 * l + 1) % E for e in range(E)] for l in range(L)}
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path, order=order)
    original = _snapshot(model)
    slim = str(tmp_path / "slim")
    pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    with BankFile.open(os.path.join(slim, "bank.ftmb")) as bank:  # a new placement in between
        for layer in range(L):
            bank.reorder_layer(layer, list(range(E)))
    restored = str(tmp_path / "restored")
    report = unpack_checkpoint(slim, restored, config=CONFIG, log=lambda _m: None)
    assert report["shards"] == {"link": 1, "rewrite": 1, "drop": 1}
    assert _snapshot(restored) == original


def test_unpack_refuses_when_the_bank_cannot_give_the_bytes_back(tmp_path):
    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=CONFIG, bank_path=bank_path, log=lambda _m: None)
    with BankFile.open(report["bank"]) as bank:
        order, digest = bank.layer_state(2)
        block = bank.read_block("down", 2)
        block[1] ^= 0x01
        os.pwrite(bank._fd, block.numpy().tobytes(), bank.layout.offset_of("down", 2))
        digest["down"] = hashlib.sha256(block.numpy().tobytes()).hexdigest()
        bank._commit_layer(2, order, digest)
    with pytest.raises(PackError, match="model-3.safetensors: rebuilt, but not the original bytes"):
        unpack_checkpoint(slim, str(tmp_path / "restored"), config=CONFIG, log=lambda _m: None)


def test_the_ft_bank_commands_end_to_end(tmp_path, monkeypatch, capsys):
    from freetoken.moe import bank_cli

    model, _, bank_path, _, _ = _served_nvfp4_bank(tmp_path)
    original = _snapshot(model)
    monkeypatch.setattr(bank_cli, "_model_config", lambda _p: CONFIG)
    slim, back = str(tmp_path / "slim"), str(tmp_path / "back")
    cache = os.path.dirname(bank_path)
    assert bank_cli.main(["pack", "--model", model, "--out", slim, "--moe-bank-dir", cache, "--dry-run"]) == 0
    assert "dry run: nothing read or written" in capsys.readouterr().out
    assert bank_cli.main(["pack", "--model", model, "--out", slim, "--moe-bank-dir", cache]) == 0
    out = capsys.readouterr().out
    assert "Nothing was deleted" in out and "can be deleted" in out
    assert bank_cli.main(["info", "--model", slim]) == 0
    assert "the only copy of the experts of" in capsys.readouterr().out
    assert bank_cli.main(["verify", "--model", slim]) == 0
    assert bank_cli.main(["unpack", "--model", slim, "--out", back]) == 0
    assert _snapshot(back) == original
    # a second pack of the same original has no bank left to pack from
    assert bank_cli.main(["pack", "--model", model, "--out", str(tmp_path / "again"), "--moe-bank-dir", cache]) == 1
    assert "no bank file" in capsys.readouterr().err


# ----- gpt-oss (MXFP4, stacked experts with biases) -------------------------------------------
GH, GI = 64, 32
GPTOSS_CONFIG = SimpleNamespace(num_layers=2, num_experts=E, moe_intermediate_size=GI, num_moe_layers=2)


def test_pack_gpt_oss_stacked_experts(tmp_path):
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.gpt_oss.weight import expert_sources as gptoss_sources

    model = str(tmp_path / "gptoss")
    os.makedirs(model)
    tensors, seed = {"model.embed_tokens.weight": torch.randn(10, GH, generator=_g(5))}, 100
    for layer in range(2):
        p = f"model.layers.{layer}.mlp.experts."
        seed += 10
        tensors[p + "gate_up_proj_blocks"] = torch.randint(0, 256, (E, 2 * GI, GH // 32, 16), dtype=torch.uint8, generator=_g(seed))
        tensors[p + "gate_up_proj_scales"] = torch.randint(0, 256, (E, 2 * GI, GH // 32), dtype=torch.uint8, generator=_g(seed + 1))
        tensors[p + "gate_up_proj_bias"] = torch.randn(E, 2 * GI, generator=_g(seed + 2)).to(torch.bfloat16)
        tensors[p + "down_proj_blocks"] = torch.randint(0, 256, (E, GH, GI // 32, 16), dtype=torch.uint8, generator=_g(seed + 3))
        tensors[p + "down_proj_scales"] = torch.randint(0, 256, (E, GH, GI // 32), dtype=torch.uint8, generator=_g(seed + 4))
        tensors[p + "down_proj_bias"] = torch.randn(E, GH, generator=_g(seed + 5)).to(torch.bfloat16)
        tensors[f"model.layers.{layer}.mlp.router.weight"] = torch.randn(E, GH, generator=_g(seed + 6))
    safetensors_torch.save_file(tensors, os.path.join(model, "model.safetensors"))
    weight_map = {n: "model.safetensors" for n in tensors}
    with open(os.path.join(model, "config.json"), "w") as f:
        f.write("{}")

    kernel = TritonGptossMxfp4MoEKernel()
    cfg = MoEConfig(num_experts=E, hidden=GH, intermediate=GI, top_k=2, has_bias=True)
    specs = kernel.layout(cfg)
    sources = gptoss_sources(model, GPTOSS_CONFIG, QuantKind.MXFP4, weight_map=weight_map)
    fp = fingerprint(sources, _headers(model, weight_map))
    layout = layout_from_specs(specs, range(2), E, _meta("mxfp4", "triton_gptoss", cfg, fp))
    bank_path = str(tmp_path / "cache" / "bank.ftmb")
    tier = MappedTier(bank_path, range(2), num_experts=E, hot_per_layer=1, layout=layout,
                      wanted={0: [1, 0, 3, 2], 1: [3, 2, 1, 0]})
    assert tier.prepare() is False
    role_of = {"gate_up_proj_blocks": "gate_up", "gate_up_proj_scales": "gate_up_scale", "gate_up_proj_bias": "gate_up_bias",
               "down_proj_blocks": "down", "down_proj_scales": "down_scale", "down_proj_bias": "down_bias"}
    for layer in range(2):
        pieces = {role: tensors[f"model.layers.{layer}.mlp.experts.{src}"] for src, role in role_of.items()}
        out = {n: torch.empty((E, *s.shape), dtype=s.dtype) for n, s in specs.items()}
        kernel.pack(pieces, cfg, out)
        tier.sink(layer, out)
    tier.finish()
    tier.banks.close()

    slim = str(tmp_path / "slim")
    report = pack_checkpoint(model, slim, config=GPTOSS_CONFIG, bank_path=bank_path, log=lambda _m: None)
    assert report["removed_tensors"] == 12 and report["shards"]["rewrite"] == 1
    left = safetensors_torch.load_file(os.path.join(slim, "model.safetensors"))
    assert set(left) == {"model.embed_tokens.weight", "model.layers.0.mlp.router.weight", "model.layers.1.mlp.router.weight"}
    verify_packed(slim, config=GPTOSS_CONFIG, log=lambda _m: None)
