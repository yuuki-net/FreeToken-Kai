"""--moe-collect-stats / --moe-stats-out / --disable-cuda-graph.

The dump is the input to a disk-backed expert bank's hot/cold placement table, so the two
things worth pinning are that asking for the dump actually turns the counters on, and that
the written payload carries the per-rank layer window (without it, two pipeline ranks'
histograms cannot be stitched back into one model-wide table).
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest


def _parse(argv):
    pytest.importorskip("freetoken.server.args")
    from freetoken.server.args import parse_args

    with tempfile.TemporaryDirectory() as d:
        # parse_args reads config.json for the dtype / parser inference
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"],
                       "model_type": "qwen4_exp", "torch_dtype": "bfloat16"}, f)
        args, _ = parse_args(
            ["--model", d, "--dtype", "bfloat16", "--tool-call-parser", "llama3",
             "--reasoning-parser", "off", *argv],
            False,
        )
    return args


def test_defaults_leave_instrumentation_off():
    args = _parse([])
    assert args.moe_collect_stats is False
    assert args.moe_stats_out is None
    # None (not []) so CudaGraphRunner still auto-sizes the capture list
    assert args.cuda_graph_bs is None


def test_stats_out_implies_collect_stats():
    args = _parse(["--moe-stats-out", "/tmp/moe.json"])
    assert args.moe_stats_out == "/tmp/moe.json"
    assert args.moe_collect_stats is True


def test_collect_stats_alone_does_not_set_an_output_path():
    args = _parse(["--moe-collect-stats"])
    assert args.moe_collect_stats is True
    assert args.moe_stats_out is None


def test_disable_cuda_graph_empties_the_bs_list():
    # [] is the existing "graphs off" contract: CudaGraphRunner's max_graph_bs becomes 0
    # and _capture_graphs returns early.
    args = _parse(["--disable-cuda-graph"])
    assert args.cuda_graph_bs == []


class _FakeFreq:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _FakeCache:
    num_layers = 2
    num_experts = 3
    cache_size = 4
    decode_target = "hybrid"
    collect_decode_freq = True

    def __init__(self):
        self.decode_freq = _FakeFreq([[5, 1, 0], [2, 2, 2]])

    def decode_miss_stats(self):
        return {"miss_rate": 0.25, "layer_calls": 8}

    def decode_miss_stats_per_layer(self):
        return {"per_layer": [{"layer": 0, "miss_rate": 0.5}, {"layer": 1, "miss_rate": 0.0}]}

    def decode_routing_stats(self):
        return {"experts_for_90pct": 2.0, "oracle_hit_at_slots": 0.75}


def _write(path, rank=(0, 1), window=None, cache=None):
    from freetoken.engine.moe_stats import write_moe_stats

    return write_moe_stats(cache or _FakeCache(), path, rank[0], rank[1], window)


def test_write_payload_carries_histogram_and_layer_window(tmp_path):
    out = tmp_path / "moe.json"
    _write(str(out), rank=(0, 1), window=(0, 24))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decode_freq"] == [[5, 1, 0], [2, 2, 2]]
    assert payload["layer_range"] == [0, 24]
    assert payload["num_experts"] == 3
    assert payload["routing"]["experts_for_90pct"] == 2.0
    assert payload["miss_stats"]["miss_rate"] == 0.25
    assert len(payload["per_layer"]) == 2


def test_pipeline_ranks_write_separate_files(tmp_path):
    out = tmp_path / "moe.json"
    _write(str(out), rank=(1, 2), window=(24, 48))
    # rank 1 of 2 -> suffixed; the bare path must stay untouched so a later rank 0 (or a
    # single-rank run) cannot be clobbered by whichever process exits last
    assert not out.exists()
    payload = json.loads((tmp_path / "moe.rank1.json").read_text(encoding="utf-8"))
    assert payload["rank"] == 1
    assert payload["world_size"] == 2
    assert payload["layer_range"] == [24, 48]


def test_no_output_path_writes_nothing(tmp_path):
    assert _write(None) is None
    assert list(tmp_path.iterdir()) == []


def test_missing_cache_is_not_an_error(tmp_path):
    from freetoken.engine.moe_stats import write_moe_stats

    out = tmp_path / "moe.json"
    # a dense model has no offload cache
    assert write_moe_stats(None, str(out), 0, 1, None) is None
    assert not out.exists()


def test_unwritable_path_does_not_raise(tmp_path):
    # shutdown must stay clean even when the dump cannot be written
    assert _write(str(tmp_path / "no-such-dir" / "moe.json")) is None


def test_histogram_omitted_when_not_collected(tmp_path):
    out = tmp_path / "moe.json"
    cache = _FakeCache()
    cache.collect_decode_freq = False
    _write(str(out), cache=cache)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decode_freq"] is None
    assert payload["miss_stats"]["miss_rate"] == 0.25


def test_bank_flags_default_off():
    args = _parse([])
    assert args.moe_bank_ram is None
    assert args.moe_bank_stats is None
    assert args.moe_bank_dir is None


def test_bank_flags_parse():
    args = _parse(["--moe-bank-ram", "50G",
                   "--moe-bank-stats", "a.json", "b.json",
                   "--moe-bank-dir", "/tmp/cold"])
    assert args.moe_bank_ram == "50G"
    assert args.moe_bank_stats == ["a.json", "b.json"]
    assert args.moe_bank_dir == "/tmp/cold"


def test_bad_bank_ram_is_rejected_at_parse_time():
    # better here than deep in the loader, after the weights have been read
    with pytest.raises(ValueError, match="could not parse"):
        _parse(["--moe-bank-ram", "lots"])
