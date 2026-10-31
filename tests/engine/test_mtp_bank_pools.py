"""--spec-mtp appends the draft head's expert layer to the offload cache's banks
(``Engine._append_mtp_bank``), after the model's MoE layers were loaded. Upstream #689 keys the
slot pools on per-layer tables (``layer_tags``, ``pool_of_layer``), so those must grow by the
same layer, or the cache asserts at boot on the rank that holds the head. CPU-only."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.cache_budget import layer_slot_bytes
from freetoken.engine.engine import Engine
from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache, bank_layouts
from freetoken.moe.pool_policy import group_layers, make_pools, pool_of_layer, split_slots

L, E = 3, 4


def _rows(n_layers):
    return {
        "gate_up": [torch.zeros(E, 8, 18, dtype=torch.uint8) for _ in range(n_layers)],
        "down": [torch.zeros(E, 4, 18, dtype=torch.uint8) for _ in range(n_layers)],
    }


def _append(tags):
    banks = SimpleNamespace(
        quant_format="q2_0",
        sources=_rows(L),
        layer_residency=[HostResidency.PINNED.value] * L,
        layer_tags=tags,
    )
    head = {k: v[0].clone() for k, v in _rows(1).items()}
    engine = SimpleNamespace(_mtp_bank_host=head)
    assert Engine._append_mtp_bank(engine, banks) == 1
    return banks


def test_the_head_layer_joins_the_model_layers_pool():
    banks = _append(["q2_0"] * L)
    assert [len(v) for v in banks.sources.values()] == [L + 1, L + 1]
    assert banks.layer_tags == ["q2_0"] * (L + 1)
    layouts = bank_layouts(banks.sources, banks.layer_tags)
    groups = group_layers("unified", layouts, layer_slot_bytes(banks.sources))
    assert groups == [[0, 1, 2, 3]]
    pools = make_pools(groups, E, min_num_slots=E, slot_limit=None)
    owner = pool_of_layer(pools, L + 1)
    assert owner == [0] * (L + 1)
    cache = OffloadMoeCache(
        num_layers=L + 1, num_experts=E, cache_size=3 * E, device=torch.device("cpu"),
        quant_format="q2_0", pool_of_layer=owner, pool_sizes=split_slots(3 * E, pools),
    )
    cache.set_bank_sources(banks.sources, layer_tags=banks.layer_tags)
    assert cache.pool_slots(L) == 3 * E


def test_untagged_banks_stay_untagged():
    banks = _append(None)
    assert banks.layer_tags is None
    # bank_layouts reads the layer count off the sources, which already hold the head's layer
    assert len(bank_layouts(banks.sources, banks.layer_tags)) == L + 1


def test_per_layer_pools_give_the_head_its_own():
    banks = _append(["q2_0"] * L)
    groups = group_layers("per-layer", bank_layouts(banks.sources, banks.layer_tags), layer_slot_bytes(banks.sources))
    owner = pool_of_layer(make_pools(groups, E, min_num_slots=E, slot_limit=None), L + 1)
    assert sorted(owner) == [0, 1, 2, 3] and len(set(owner)) == L + 1


def test_an_mtp_ggufs_experts_join_a_gguf_bank_in_their_own_types():
    """--spec-mtp-gguf on a GGUF model (FreeToken-Kai#6): the MTP file's Q8_0 rows are wider than the
    model's IQ3_S ones; they keep their tag and MMQ tails, so the cache gives them a pool of their own."""
    from freetoken.layers.quantization import QuantKind

    iq3 = ("ggml", "IQ3_S", "IQ4_NL")
    banks = SimpleNamespace(
        quant_format="gguf", kind=QuantKind.GGUF, sources=_rows(L),
        layer_residency=[HostResidency.PINNED.value] * L, layer_tags=[iq3] * L,
        row_tail_bytes={"gate_up": [0] * L, "down": [0] * L},
    )
    head = {"gate_up": torch.zeros(E, 8, 34, dtype=torch.uint8), "down": torch.zeros(E, 4, 34, dtype=torch.uint8)}
    q8 = ("ggml", "Q8_0", "Q8_0")
    engine = SimpleNamespace(_mtp_bank_host=head, _mtp_bank_tag=q8, _mtp_bank_tails={"gate_up": 64, "down": 32})
    assert Engine._append_mtp_bank(engine, banks) == 1
    assert banks.layer_tags == [iq3] * L + [q8]
    assert banks.row_tail_bytes == {"gate_up": [0] * L + [64], "down": [0] * L + [32]}
    assert [t.shape[-1] for t in banks.sources["gate_up"]] == [18] * L + [34]
    groups = group_layers("unified", bank_layouts(banks.sources, banks.layer_tags), layer_slot_bytes(banks.sources))
    assert sorted(groups) == [[0, 1, 2], [3]]
