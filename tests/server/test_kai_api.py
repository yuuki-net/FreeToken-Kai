"""The web console's aggregation of per-rank snapshots (scheduler/webstats.py -> server/kai_api.py)."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

from freetoken.server import kai_api
from freetoken.webui.stats_path import stats_dir

GiB = 2**30


def _rank(rank, active, miss, faults, layer_range):
    return {
        "v": 1, "rank": rank, "size": 2, "time": time.time(), "spec_k": 0,
        "layer_range": layer_range,
        "gpu": {"index": rank, "name": "RTX 3060", "total_bytes": 12 * GiB, "free_bytes": 1 * GiB, "reserved_bytes": 10 * GiB},
        "moe": {"decode_target": "gpu", "cache_size": 700, "num_layers": 2, "num_experts": 64, "collect_stats": True},
        "prefill_chunk": 4096,
        "windows": {"60": {"seconds": 60.0, "layer_active": active, "layer_miss": miss, "major_faults": faults,
                           "decode_tokens": 600, "prefill_new_tokens": 100, "prefill_cached_tokens": 300}},
    }


def _state(tmp_path, monkeypatch, docs):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    d = stats_dir(19999)
    os.makedirs(d)
    for doc in docs:
        with open(os.path.join(d, f"rank{doc['rank']}.json"), "w") as fh:
            json.dump(doc, fh)
    return SimpleNamespace(config=SimpleNamespace(server_port=19999, kv_cache_dtype="q8_0", prefill_chunk_budget=0.55))


def test_kai_block_sums_ranks(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, [
        _rank(0, [100, 100], [5, 5], 30, [0, 24]),
        _rank(1, [100, 100], [15, 15], 90, [24, 48]),
    ])
    k = kai_api.kai_block(state)
    assert abs(k["moe"]["gpu_hit_rate"] - (1 - 40 / 400)) < 1e-9
    assert k["moe"]["major_faults_per_min"] == 120
    assert k["prefix_reuse_rate"] == 0.75
    assert [g["layers"] for g in k["gpus"]] == [[0, 23], [24, 47]]
    assert [g["used_bytes"] for g in k["gpus"]] == [11 * GiB, 11 * GiB]


def test_experts_doc_orders_layers_by_rank(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, [_rank(1, [10, 20], [1, 2], 0, [24, 48]), _rank(0, [30, 40], [3, 4], 0, [0, 24])])
    doc = kai_api.experts_doc(state, "60")
    assert [(l["rank"], l["active"]) for l in doc["layers"]] == [(0, 30), (0, 40), (1, 10), (1, 20)]


def test_no_snapshots_means_no_block(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    assert kai_api.kai_block(SimpleNamespace(config=SimpleNamespace(server_port=19998))) is None
