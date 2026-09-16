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


def _write_freq(rank, windows, cumulative):
    d = stats_dir(19999)
    with open(os.path.join(d, f"rank{rank}.experts.json"), "w") as fh:
        json.dump({"v": 1, "rank": rank, "time": time.time(), "layer_range": None,
                   "cumulative": cumulative, "windows": windows}, fh)


def test_expert_counts_ride_along_only_when_asked(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, [_rank(0, [10, 20], [1, 2], 0, [0, 24])])
    _write_freq(0, {"300": {"seconds": 290.0, "freq": [[3, 0, 1], [0, 2, 2]]}},
                {"seconds": 900.0, "freq": [[9, 1, 1], [1, 5, 5]]})
    assert kai_api.experts_doc(state, "300")["expert_freq"] is None
    doc = kai_api.experts_doc(state, "300", freq=True)
    assert doc["expert_freq"] == [{"rank": 0, "layer_range": None, "seconds": 290.0, "freq": [[3, 0, 1], [0, 2, 2]]}]
    # a window the file has not filled yet falls back to everything since the start
    assert kai_api.experts_doc(state, "60", freq=True)["expert_freq"][0]["seconds"] == 900.0


def test_expert_counts_file_is_not_a_rank(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, [_rank(0, [10, 20], [1, 2], 0, [0, 24])])
    _write_freq(0, {}, {"seconds": 10.0, "freq": [[1]]})
    assert [r["rank"] for r in kai_api.read_ranks(19999)] == [0]
    assert len(kai_api.experts_doc(state, "60")["layers"]) == 2


def test_publisher_writes_windowed_expert_counts(tmp_path, monkeypatch):
    import torch

    from freetoken.scheduler.webstats import WebStatsPublisher

    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    cfg = SimpleNamespace(tp_info=SimpleNamespace(rank=0, size=1), server_port=19997, pp_layer_range=None)
    pub = WebStatsPublisher(SimpleNamespace(config=cfg))
    # one routing per 10 s tick: layer 0 always picks expert 1, layer 1 expert 2
    for i in range(40):
        pub._write_freq(1000.0 + 10 * i, torch.tensor([[0, i, 0], [0, 0, 2 * i]]))
    state = SimpleNamespace(config=SimpleNamespace(server_port=19997))
    five = kai_api.read_expert_freq(19997, "300")[0]
    assert five["freq"] == [[0, 30, 0], [0, 0, 60]] and five["seconds"] == 300.0
    one = kai_api.read_expert_freq(19997, "60")[0]
    assert one["freq"] == [[0, 6, 0], [0, 0, 12]]
    assert kai_api.experts_doc(state, "300", freq=True)["expert_freq"][0]["freq"] == five["freq"]


def test_no_expert_counts_without_the_flag(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch, [_rank(0, [10, 20], [1, 2], 0, [0, 24])])
    assert kai_api.experts_doc(state, "300", freq=True)["expert_freq"] is None


def test_no_snapshots_means_no_block(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    assert kai_api.kai_block(SimpleNamespace(config=SimpleNamespace(server_port=19998))) is None
