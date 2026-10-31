"""``--moe-bank-ram`` for GGUF experts, whose layers keep their own ggml types (FreeToken-Kai#5: unsloth's
Qwen3.8-Flash-Next UD-IQ4_XS keeps IQ3_S / IQ4_NL experts in most layers and IQ4_XS in a few, so one
bank file holds several row sizes; the DeepSeek-V4.1 GSQ-RCO builds mix IQ1_M / IQ2_XXS / IQ2_XS the same way).

The geometry is read off the GGUF's tensor table before a byte is read (``gguf_bank_geometry``), the
file lays each layer out in its own row size (``MappedBankLayout.layer_banks``), the loader streams each
layer into it (``load_gguf_expert_banks(layer_sink=...)``), and the mapping hands every layer back in
its own shape -- the second start without reading the GGUF at all. No GPU needed."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.layers.quantization.ggml_types import row_bytes  # noqa: E402
from freetoken.moe.bank_file import BankFile, MappedBankLayout, layout_from_layer_specs  # noqa: E402

from ._gguf_writer import UD_IQ4_XS_TYPES as TYPES  # noqa: E402
from ._gguf_writer import write_experts_gguf  # noqa: E402

E, H, I = 4, 256, 256
L = len(TYPES)


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    monkeypatch.setenv("FREETOKEN_BANK_REGISTER", "none")


def _write_experts_gguf(path: str, seed: int = 0) -> None:
    write_experts_gguf(path, TYPES, experts=E, hidden=H, inter=I, seed=seed)


def _expected(path: str, layer: int) -> dict:
    from freetoken.models.gguf.experts import _expert_rows
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(path)
    gate, up, down = (_expert_rows(index[f"blk.{layer}.{n}.weight"]) for n in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"))
    return {"gate_up": torch.cat([gate, up], dim=1), "down": down}


def test_a_layout_keeps_each_layers_own_rows_and_reads_back(tmp_path):
    from freetoken.models.gguf.experts import gguf_bank_geometry

    path = str(tmp_path / "experts.gguf")
    _write_experts_gguf(path)
    geometry = gguf_bank_geometry(path, L)
    assert [t[1:] for t in geometry.tags] == [(a.name, b.name) for a, b in TYPES]
    layout = layout_from_layer_specs(dict(enumerate(geometry.specs)), range(L), E, {"kind": "gguf"})
    # the most common geometry (IQ1_M / IQ1_M) is the file's, the others are overrides
    assert sorted(layout.layer_banks) == [1, 2]
    for layer, (gate_up, down) in enumerate(TYPES):
        assert layout.row_bytes("gate_up", layer) == 2 * I * row_bytes(int(gate_up), H)
        assert layout.row_bytes("down", layer) == H * row_bytes(int(down), I)
    # blocks follow each other in file order, each 4096-aligned, sized by its own layer's rows
    pos = layout.data_offset
    for layer in range(L):
        for name in ("gate_up", "down"):
            assert layout.offset_of(name, layer) == pos
            pos += layout.block_bytes(name, layer)
    assert layout.total_bytes() == pos
    f = BankFile.create(str(tmp_path / "bank.ftmb"), layout)
    f.close()
    back = MappedBankLayout.read(str(tmp_path / "bank.ftmb"))
    assert back.mismatch(layout) is None and back.layer_banks == layout.layer_banks
    other = layout_from_layer_specs({l: s for l, s in enumerate(gguf_bank_geometry(path, L).specs[::-1])}, range(L), E, {"kind": "gguf"})
    assert "per-layer" in (layout.mismatch(other) or "")


def _gguf_method():
    from freetoken.layers.quantization import QuantKind
    from freetoken.layers.quantization.moe.base import BankSpec, MoEConfig

    method = SimpleNamespace(kind=QuantKind.GGUF, kernel=SimpleNamespace(name="ggml"), cfg=MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2))
    # what the bound method of layer 0 lays out: one layer's geometry, which build_tier must not take for all
    method.layout = lambda: {"gate_up": BankSpec((2 * I, row_bytes(int(TYPES[0][0]), H)), torch.uint8), "down": BankSpec((H, row_bytes(int(TYPES[0][1]), I)), torch.uint8)}
    return method


def _config(tmp_path, model_path, budget):
    mc = SimpleNamespace(num_experts=E, num_moe_layers=L, first_k_dense_replace=0)
    return SimpleNamespace(
        model_path=model_path, model_config=mc, full_model_config=mc, moe_bank_ram=str(budget), moe_bank_stats=None,
        moe_bank_dir=str(tmp_path / "bankmap"), tp_info=SimpleNamespace(rank=0, size=1),
    )


def test_a_gguf_streams_into_its_bank_file_once_and_maps_back_layer_by_layer(tmp_path, monkeypatch):
    from freetoken.models.gguf import experts as gx
    from freetoken.models.gguf.experts import gguf_bank_geometry, load_gguf_expert_banks
    from freetoken.moe import bank_disk
    from freetoken.moe.bank_tier import build_tier

    path = str(tmp_path / "experts.gguf")
    _write_experts_gguf(path)
    cells = [bank_disk.cell_bytes_of_specs(s) for s in gguf_bank_geometry(path, L).specs]
    budget = 2 * sum(cells)  # two experts of every layer
    tier = build_tier(_config(tmp_path, path, budget), _gguf_method(), log=lambda _m: None)
    assert tier is not None and tier.layout.layer_banks and tier.hot_per_layer == 2
    assert tier.layout.meta["fingerprint"] and tier.layout.meta["source_stamp"]

    assert not tier.prepare()
    banks = load_gguf_expert_banks(path, L, layer_sink=tier.sink)
    assert banks.streamed and not any(banks.sources.values()) and banks.layer_tags == gguf_bank_geometry(path, L).tags
    tier.finish()
    for layer in range(L):
        want = _expected(path, layer)
        for role in ("gate_up", "down"):
            view = tier.sources[role][layer]
            assert tuple(view.shape) == tuple(want[role].shape) and torch.equal(view, want[role]), (layer, role)
    tier.banks.close()

    # the second start finds every layer in the file: the GGUF's expert tensors are not read
    def no_read(*_a, **_k):
        raise AssertionError("the GGUF's experts were read on the second start")

    monkeypatch.setattr(gx, "_expert_rows", no_read)
    again = build_tier(_config(tmp_path, path, budget), _gguf_method(), log=lambda _m: None)
    assert again.prepare()
    again.finish()
    monkeypatch.undo()
    for layer in range(L):
        assert torch.equal(again.sources["down"][layer], _expected(path, layer)["down"])
    again.banks.close()


def test_a_reorder_moves_each_layers_own_rows(tmp_path):
    from freetoken.models.gguf.experts import gguf_bank_geometry

    path = str(tmp_path / "experts.gguf")
    _write_experts_gguf(path, seed=1)
    layout = layout_from_layer_specs(dict(enumerate(gguf_bank_geometry(path, L).specs)), range(L), E, {"kind": "gguf"})
    bank = BankFile.create(str(tmp_path / "bank.ftmb"), layout)
    for layer in range(L):
        bank.write_layer(layer, _expected(path, layer), list(range(E)))
    order = [2, 0, 3, 1]
    assert bank.reorder_layer(2, order)
    rows = bank.read_block("gate_up", 2).view(E, -1)
    want = _expected(path, 2)["gate_up"].reshape(E, -1)
    for physical, logical in enumerate(order):
        assert torch.equal(rows[physical], want[logical])
    # the neighbours of the reordered layer are untouched
    assert torch.equal(bank.read_block("down", 3).view(E, -1), _expected(path, 3)["down"].reshape(E, -1))
    bank.check_digest(1)
    bank.close()
    os.remove(str(tmp_path / "bank.ftmb"))
