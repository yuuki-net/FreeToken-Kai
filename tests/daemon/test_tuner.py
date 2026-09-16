"""The console's benchmark: the search over flags (webui/search.py), the job that runs it
(webui/tuner.py) and the decisions in the hardware bench (webui/hwbench.py).

The job runs against fakes: a manager that records lifecycle calls, a child process that prints
canned JSON lines, and a serve whose speed follows the flags it was started with. What is pinned:
the previous engine comes back, each candidate is one change on the best so far and is kept only on
the measurement, candidates that undo each other are not both tried, a failure is a run not kept,
generation on code is measured where MTP is in play and the chosen use decides."""

from __future__ import annotations

import io
import json

import pytest

from freetoken.webui import hwbench, search, tuner

GiB = 1 << 30


# ------------------------------------------------------------------ flags and decisions
def test_set_flags_overrides_adds_and_drops_in_place():
    args = ["--dtype", "float16", "--moe-cache-auto", "--moe-cpu-threads", "8", "--host", "0.0.0.0"]
    out = tuner.set_flags(args, {"--moe-cpu-threads": "6", "--kv-reserve-tokens": "65536", "--moe-collect-stats": None},
                          drop=("--dtype",))
    assert out == ["--moe-cache-auto", "--moe-cpu-threads", "6", "--host", "0.0.0.0",
                   "--kv-reserve-tokens", "65536", "--moe-collect-stats"]
    assert tuner.parse_flags(["--moe-hybrid-max-fetch", "-1", "--x=3"]) == [("--moe-hybrid-max-fetch", "-1"), ("--x", "3")]


def test_quant_backend_entries_join_instead_of_replacing():
    args = ["--quant-backend", "moe.nvfp4=marlin"]
    assert search.merge_changes(args, {"--quant-backend": "linear.nvfp4=triton"}) == {"--quant-backend": "linear.nvfp4=triton,moe.nvfp4=marlin"}
    assert search.merge_changes(args, {"--quant-backend": "moe.nvfp4=b12x"}) == {"--quant-backend": "moe.nvfp4=b12x"}
    assert search.merge_changes(args, {"--spec-mtp": "3"}) == {"--spec-mtp": "3"}


def _t(prefill, decode, code=None, ok=True):
    return {"ok": ok, "prefill_tps": prefill, "decode_tps": decode, "decode_code_tps": code}


def test_keep_by_goal_and_use():
    base = _t(600, 30.0)
    gen = search.Candidate("x", {"--a": "1"})
    assert tuner.keep(base, _t(560, 31.0), gen, "prose")          # +3.3% generation, prefill holds 90%
    assert not tuner.keep(base, _t(500, 31.0), gen, "prose")      # prefill fell below 90%
    pre = search.Candidate("y", {"--b": "1"}, goal="prefill", gain=1.05)
    assert tuner.keep(base, _t(640, 29.2), pre, "prose") and not tuner.keep(base, _t(640, 28.0), pre, "prose")
    either = search.Candidate("z", {"--c": "1"}, goal="either")
    assert tuner.keep(base, _t(580, 31.2), either, "prose") and tuner.keep(base, _t(700, 29.5), either, "prose")
    assert not tuner.keep(base, _t(600, 30.0, ok=False), either, "prose")
    # MTP: faster on code, slower on prose -- the use decides
    mtp = search.Candidate("m", {"--spec-mtp": "3"}, code=True, hold_prefill=0.85)
    drafted = _t(560, 26.0, code=45.0)
    assert tuner.keep(base, drafted, mtp, "code") and not tuner.keep(base, drafted, mtp, "prose")
    assert tuner.keep(base, drafted, mtp, "both")  # sqrt(26*45)=34.2 > 30.9


def _geo(now=16384, kv=10880, slots=1104, per_expert=1775616, kv_max=249909):
    return {"num_pages": now, "page_size": 1, "moe_cache_size": slots,
            "unit_bytes": {"kv_per_token": kv, "moe_per_expert": per_expert},
            "limits": {"kv_tokens": {"max": kv_max}}}


def test_context_candidates_stop_at_the_model_and_the_budget():
    assert tuner.context_candidates(_geo(), 262144) == [32768, 65536, 131072]  # 262144 > budget max
    assert tuner.context_candidates(_geo(), 32768) == [32768]
    assert tuner.context_candidates(_geo(now=65536), 262144) == [131072]


def test_longer_context_is_still_answered_for_old_callers():
    c = tuner.longer_context(_geo(), [{"gpu": {"free_bytes": int(2.5 * GiB)}}], cache_auto=False, model_max=262144)
    assert c["tokens"] == 131072


def test_corpus_is_this_repository():
    corpus = tuner.load_corpus()
    assert "FreeToken" in corpus["prose"] and "def " in corpus["code"]
    import random
    part = tuner.excerpt(corpus["code"], 1000, random.Random(1))
    assert len(part) == 1000
    assert tuner.load_corpus("/nonexistent")["prose"]  # a wheel install still has something to read


# ------------------------------------------------------------------ the hardware bench's decisions
def test_threads_are_the_knee_not_the_maximum():
    assert hwbench.pick_threads({1: 3.0, 2: 5.8, 4: 9.9, 6: 11.6, 8: 11.9, 12: 12.1}) == 6
    assert hwbench.thread_candidates(12) == [1, 2, 4, 6, 8, 10, 12]
    assert hwbench.thread_candidates(3) == [1, 2, 3]


def test_expert_formats():
    assert hwbench.expert_format("modelopt") == "nvfp4"
    assert hwbench.expert_format("mxfp4") == "mxfp4_triton"
    assert hwbench.expert_format("fp8") == "fp8_block"
    assert hwbench.expert_format(None) == "bf16"


def _hw(cpu_best, gather):
    return {"cpu_moe": {"best_gbs": cpu_best, "threads": 6}, "gather": {"0": {"gbs": gather}},
            "overlap": {"cpu_gbs": 6.0, "pcie_gbs": 3.0, "fetch_fraction": 0.333}}


def test_derive_hybrid_when_the_cpu_is_twice_the_link():
    notes = {n["flag"]: n for n in hwbench.derive(_hw(12.0, 5.0))}
    assert notes["--moe-strategy"]["value"] == "hybrid" and notes["--moe-cpu-threads"]["value"] == "6"
    assert all(n["why"] and n["why_en"] for n in notes.values())


def test_derive_offload_when_the_link_keeps_up():
    notes = {n["flag"]: n for n in hwbench.derive(_hw(12.0, 7.0))}
    assert notes["--moe-strategy"]["value"] == "offload" and "--moe-cpu-threads" not in notes


def test_bench_profile_is_merged_not_replaced(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
    from freetoken.moe.benchbw import Workload

    gpu = {"index": 0, "name": "NVIDIA GeForce RTX 2060", "uuid": "GPU-test"}
    path = default_profile_path(gpu["uuid"])
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump({"version": 4, "gpu": gpu, "dtypes": {"bf16": "offload"}, "dtype_kernels": {"bf16": {"recommended": "offload"}}}, fh)
    wl = Workload("Ornith", 2048, 512, 256, 8, ("nvfp4",))
    m = {"cpu_moe": {"best_gbs": 40.8, "expert_bytes": 1775616}, "gather": {"0": {"gbs": 9.2}},
         "overlap": {"cpu_gbs": 31.6, "pcie_gbs": 5.6}, "ram": {"read_gbs": 53.8}, "pcie": {"0": {"h2d_gbs": 11.1, "d2h_gbs": 13.0}}}
    assert hwbench.update_bench_profile(gpu, "nvfp4", wl, m) == path
    prof = json.load(open(path))
    assert prof["dtypes"] == {"bf16": "offload", "nvfp4": "hybrid"}
    assert load_backend_recommendation("nvfp4", gpu["name"], gpu_uuid=gpu["uuid"]) == "hybrid"
    assert load_hybrid_fetch_fraction("nvfp4", gpu["name"], gpu_uuid=gpu["uuid"]) == pytest.approx(5.6 / 37.2)


# ------------------------------------------------------------------ what gets tried
ORNITH = dict(model_type="qwen3_5_moe", num_experts=256, num_layers=40, quant="modelopt", mtp=True, max_context=262144,
              layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
              ram_total=24 * GiB, weight_bytes=21 * GiB)
ORNITH_ARGS = ["--dtype", "float16", "--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8",
               "--moe-cache-auto", "--disable-moe-prefill-overlap", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "16384",
               "--max-seq-len-override", "16384", "--host-embedding", "--prefill-mixer-pieces", "2", "--max-running-req", "1"]
HW_2060 = {"measurements": {"cpu_moe": {"best_gbs": 36.4, "threads": 8, "cores": 12}, "gather": {"0": {"gbs": 10.1}},
                            "pcie": {"0": {"h2d_gbs": 11.2}}}}
FLASH = dict(model_type="qwen4_exp", num_experts=512, num_layers=48, quant="modelopt", mtp=True, max_context=262144,
             layer_types=["linear_attention"] * 36 + ["full_attention"] * 12, ram_total=108 * GiB, weight_bytes=135 * GiB)
FLASH_ARGS = ["--pp-size", "2", "--gpu", "0,1", "--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8",
              "--moe-cache-auto", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "65536", "--dense-quant", "fp8",
              "--prefill-mixer-pieces", "2", "--max-running-req", "1"]
HW_3060 = {"measurements": {"cpu_moe": {"best_gbs": 30.3, "threads": 8, "cores": 8}, "gather": {"0": {"gbs": 11.5}, "1": {"gbs": 6.3}},
                            "pcie": {"0": {"h2d_gbs": 23.1}, "1": {"h2d_gbs": 6.3}}}}


def _keys(args, facts, hw, mode):
    return [c.key for c in search.plan(args, facts, hw, mode)]


def test_standard_plan_on_a_2060_with_ornith():
    keys = _keys(ORNITH_ARGS, ORNITH, HW_2060, "standard")
    assert keys == ["strategy_offload", "threads_all", "threads_fewer", "kernel_moe_triton", "kernel_moe_marlin", "kernel_moe_b12x",
                    "kv_16bit", "overlap_on", "budget_075", "pieces_4", "mtp_3", "mtp_5"]


def test_thorough_plan_adds_the_rarely_winning_ones():
    standard = set(_keys(ORNITH_ARGS, ORNITH, HW_2060, "standard"))
    thorough = set(_keys(ORNITH_ARGS, ORNITH, HW_2060, "thorough"))
    assert standard < thorough
    assert {"fetch_none", "kv_q4", "hit_d2d", "budget_090", "pieces_1", "max_prefill_16k", "no_host_embedding",
            "kernel_linear_marlin"} <= thorough - standard


def test_two_3060s_with_flash_next():
    keys = _keys(FLASH_ARGS, FLASH, HW_3060, "standard")
    assert "pp_layers_25" in keys  # rank 1 sits on the x4 link: rank 0 takes one more layer
    assert "threads_fewer" in keys and "threads_all" not in keys  # already at every core
    assert "mtp_3" in keys and "mtp_5" in keys
    assert "ple_pinned" not in keys  # 135 GiB of weights do not leave RAM for the PLE table
    thorough = _keys(FLASH_ARGS, FLASH, HW_3060, "thorough")
    assert {"pp_layers_26", "send_ahead_3", "prefill_group_2", "dense_bf16"} <= set(thorough)


def test_no_moe_candidates_for_a_dense_model_and_no_mtp_without_a_head():
    dense = dict(model_type="llama", num_layers=32, quant=None, mtp=False)
    keys = _keys(["--max-running-req", "1"], dense, {}, "thorough")
    assert keys and all(not k.startswith(("strategy", "kernel", "kv_", "mtp", "threads")) for k in keys)


def test_every_skip_has_a_reason_in_both_languages():
    assert all(flag and ja and en for flag, ja, en in search.SKIPPED)


# ------------------------------------------------------------------ the job, with fakes
class FakeManager:
    def __init__(self, running=None):
        self.calls = []
        self.cfg = running  # (model, port, args) or None
        self.fail_with = set()  # arg tokens that make a start exit

    def status(self):
        running = self.cfg is not None
        return {"running": running, "starting": False, "model": self.cfg[0] if running else None,
                "port": self.cfg[1] if running else None, "lastExitCode": None if running else 1}

    def serve_args(self):
        return list(self.cfg[2]) if self.cfg else []

    def stop(self, *a, **k):
        self.calls.append(("stop",))
        self.cfg = None
        return {"stopped": True}

    def switch(self, model, port, args, force=False):
        self.calls.append(("switch", model, port, list(args)))
        self.cfg = None if self.fail_with & set(args) else (model, port, list(args))
        return {"pid": 1}


class FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(json.dumps(x) + "\n" for x in lines))
        self.returncode = 0

    def wait(self):
        return 0

    def terminate(self):
        pass


HW_LINES = [
    {"type": "plan", "steps": [{"id": "ram"}]},
    {"type": "step", "id": "ram"}, {"type": "sample", "id": "ram", "value": 20.0}, {"type": "done", "id": "ram", "value": 20.0},
    {"type": "result", "measurements": {"ram": {"read_gbs": 20.0}}, "notes": [
        {"flag": "--moe-strategy", "value": "hybrid", "why": "w", "why_en": "w"},
        {"flag": "--moe-cpu-threads", "value": "6", "why": "w", "why_en": "w"},
        {"flag": "--moe-hybrid-max-fetch", "value": "-1", "why": "w", "why_en": "w"},
    ]},
]

# a dense model keeps the plan to the prefill candidates; a MoE facts dict brings the rest
DENSE_REC = {
    "flags": ["--dtype", "float16", "--kv-reserve-tokens", "16384", "--max-seq-len-override", "16384", "--max-running-req", "1"],
    "notes": [{"flag": f, "value": v, "why": "r", "why_en": "r"} for f, v in
              [("--dtype", "float16"), ("--kv-reserve-tokens", "16384"), ("--max-seq-len-override", "16384"), ("--max-running-req", "1")]],
    "host": {"model": {"model_type": "llama", "num_layers": 32, "max_context": 262144}, "memory": {"total": 24 * GiB}, "weight_bytes": 8 * GiB},
}


def _moe_rec(mtp=False):
    rec = json.loads(json.dumps(DENSE_REC))
    rec["flags"] += ["--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8", "--moe-cache-auto"]
    rec["host"]["model"] = dict(model_type="qwen3_5_moe", num_experts=256, num_layers=40, quant=None, mtp=mtp, max_context=262144)
    return rec


class FakeServe:
    """A serve whose speed follows its flags. ``base`` is (prose, prefill) at 16k context; ``effects``
    maps "--flag=value" (or "--flag" for a switch, "-flag" for a dropped one) to factors
    (prose, prefill, code); ``fail`` is a predicate on the flags that errors the prompt request."""

    def __init__(self, manager, base=(30.0, 600.0), effects=None, fail=None, free=2.5 * GiB):
        self.m, self.base, self.effects, self.fail, self.free = manager, base, effects or {}, fail, free
        self.t = 0.0
        self.code_seen = False

    def clock(self):
        return self.t

    def flags(self):
        return dict(tuner.parse_flags(self.m.cfg[2]))

    def ctx(self):
        return int(self.flags().get("--kv-reserve-tokens") or 16384)

    def rates(self):
        f = self.flags()
        prose, prefill = self.base
        code = prose
        for key, (dp, pp, dc) in self.effects.items():
            if key.startswith("-") and not key.startswith("--"):
                hit = ("-" + key) not in f
            elif "=" in key:
                k, v = key.split("=", 1)
                hit = f.get(k) == v
            else:
                hit = key in f
            if hit:
                prose, prefill, code = prose * dp, prefill * pp, code * dc
        return prose, prefill, code

    def http(self, port, path, body=None, timeout=0):
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": "m"}]}
        if path == "/v1/cache/status":
            return {"geometry": _geo(now=self.ctx())}
        if path.startswith("/v1/kai/experts"):
            return {"layers": [{"active": 100, "miss": 10}], "ranks": [{"gpu": {"free_bytes": int(self.free)}}]}
        if path == "/v1/completions":
            assert isinstance(body["prompt"], str)  # token-id prompts are refused by ft serve
            n = int(len(body["prompt"]) / 4)
            if body["max_tokens"] == 1:
                if self.fail and self.fail(self.flags()):
                    raise RuntimeError("HTTP Error 500: Internal Server Error")
                self.t += n / self.rates()[1]
            return {"usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]}}
        raise AssertionError(path)

    def stream(self, port, path, body):
        prose, _, code = self.rates()
        is_code = "Python module" in body["prompt"]
        self.code_seen |= is_code
        rate = code if is_code else prose
        for _ in range(body["max_tokens"]):
            self.t += 1 / rate
            yield "data: " + json.dumps({"choices": [{"text": "x"}]}) + "\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}) + "\n"
        yield "data: [DONE]\n"


CORPUS = {"prose": "FreeToken docs. " * 4000, "code": "def f():\n    return 1\n" * 3000}


def _job(tmp_path, manager, serve, rec=DENSE_REC, lines=HW_LINES):
    return tuner.TuneJob(manager, state_dir=str(tmp_path), python="python", default_port=1919,
                         recommend=lambda m: rec, spawn=lambda argv: FakeProc(lines), corpus=CORPUS,
                         http=serve.http, stream=serve.stream, clock=serve.clock, sleep=lambda s: None)


def _run(job, model="/models/M", trials=True, mode="standard", use="both"):
    job.start(model, trials, mode=mode, use=use)
    job._thread.join(30)
    return job.status()


def _tried(r):
    return [(t["label"], t.get("key"), t.get("decision")) for t in r["trials"]]


def test_full_run_tries_each_candidate_once_and_brings_the_old_engine_back(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    # the budget does nothing; 32k context costs 2%, 65k costs 10%
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.98, 1.0, 0.98), "--kv-reserve-tokens=65536": (0.90, 1.0, 0.90)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    assert _tried(r) == [("A", None, "base"), ("B", "budget_075", "rejected"), ("C", "context_32768", "kept"),
                         ("D", "context_65536", "rejected")]
    assert r["chosen"] == "C"
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--kv-reserve-tokens"] == flags["--max-seq-len-override"] == "32768"
    assert "--prefill-chunk-budget" not in flags and "--moe-collect-stats" not in flags
    assert [n for n in r["notes"] if n.get("key") == "budget_075"][0]["rejected"]
    assert m.calls[0] == ("stop",) and m.calls[-1] == ("switch", *prev) and m.cfg == prev
    assert json.load(open(tmp_path / "tune" / "M.json"))["chosen"] == "C"
    assert r["skipped"] and all(s["why"] and s["why_en"] for s in r["skipped"])
    kinds = {e["k"] for e in st["events"]}
    assert {"phase", "hw", "trial", "sample", "plan", "decision"} <= kinds


def test_a_kept_change_is_carried_into_every_later_run(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert _tried(r)[1] == ("B", "budget_075", "kept")
    assert all(dict(tuner.parse_flags(t["args"]))["--prefill-chunk-budget"] == "0.75" for t in r["trials"][2:])
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"
    assert m.cfg is None  # nothing was running before: nothing is left running


def test_a_run_that_fails_mid_measurement_is_not_kept_and_the_job_goes_on(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)}, fail=lambda f: f.get("--prefill-chunk-budget") == "0.75")
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    assert _tried(r)[1] == ("B", "budget_075", "failed") and "500" in r["trials"][1]["error"]
    assert "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))


def test_undoing_candidates_are_not_both_tried(tmp_path):
    m = FakeManager()
    # offload wins; from offload args the plan offers hybrid again, which must not be tried
    serve = FakeServe(m, effects={"--moe-strategy=offload": (1.2, 1.0, 1.2), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve, rec=_moe_rec()))["result"]
    keys = [k for _, k, _ in _tried(r)]
    assert "strategy_offload" in keys and "strategy_hybrid" not in keys
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--moe-strategy"] == "offload" and "--moe-cpu-threads" not in flags
    assert "threads_fewer" not in keys[keys.index("strategy_offload"):]  # hybrid-only candidates close after the switch


def test_mtp_is_measured_on_code_and_the_use_decides(tmp_path):
    effects = {"--spec-mtp=3": (0.85, 0.95, 1.6), "--spec-mtp=5": (0.80, 0.95, 1.7), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)}
    m = FakeManager()
    serve = FakeServe(m, effects=effects)
    r = _run(_job(tmp_path, m, serve, rec=_moe_rec(mtp=True)), use="code")["result"]
    assert serve.code_seen
    decisions = {k: d for _, k, d in _tried(r)}
    assert decisions["mtp_3"] == "kept" and decisions["mtp_5"] == "kept"
    assert dict(tuner.parse_flags(r["args"]))["--spec-mtp"] == "5"
    mtp_runs = [t for t in r["trials"] if (t.get("key") or "").startswith("mtp_")]
    assert all(t["decode_code_tps"] > t["decode_tps"] for t in mtp_runs)

    m2 = FakeManager()
    r2 = _run(_job(tmp_path / "prose", m2, FakeServe(m2, effects=effects), rec=_moe_rec(mtp=True)), use="prose")["result"]
    assert "--spec-mtp" not in dict(tuner.parse_flags(r2["args"]))


def test_float16_that_does_not_load_is_dropped(tmp_path):
    m = FakeManager()
    m.fail_with = {"float16"}
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert st["state"] == "done", st["error"]
    assert "--dtype" not in dict(tuner.parse_flags(r["args"]))
    assert [n for n in r["notes"] if n["flag"] == "--dtype"][0].get("removed")
    assert r["trials"][0]["ok"] is False


def test_hardware_only_does_not_start_a_server(tmp_path):
    m = FakeManager()
    st = _run(_job(tmp_path, m, FakeServe(m), rec=_moe_rec()), trials=False)
    assert st["state"] == "done"
    assert not [c for c in m.calls if c[0] == "switch"]
    assert dict(tuner.parse_flags(st["result"]["args"]))["--moe-strategy"] == "hybrid"


def test_offload_from_the_bench_drops_the_cpu_flags(tmp_path):
    lines = [dict(x) for x in HW_LINES]
    lines[-1] = {"type": "result", "measurements": {}, "notes": [{"flag": "--moe-strategy", "value": "offload", "why": "w", "why_en": "w"}]}
    m = FakeManager()
    st = _run(_job(tmp_path, m, FakeServe(m), rec=_moe_rec(), lines=lines), trials=False)
    flags = dict(tuner.parse_flags(st["result"]["args"]))
    assert flags["--moe-strategy"] == "offload" and "--moe-cpu-threads" not in flags and "--moe-cpu-layers" not in flags


def test_prompt_progress_is_reported_while_the_request_runs(tmp_path):
    import time as _time

    counter = {"n": 5000}

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            counter["n"] += 2304  # one chunk per look
            return {"ranks": [{"counters": {"prefill_new_tokens": counter["n"]}}]}
        if path == "/v1/completions":
            _time.sleep(2.3)
            return {"usage": {"prompt_tokens": 8192}}
        raise AssertionError(path)

    job = tuner.TuneJob(FakeManager(), state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    doc = job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert doc["usage"]["prompt_tokens"] == 8192
    progress = [e for e in job.events if e["k"] == "progress"]
    assert len(progress) >= 2
    assert progress[0]["done"] == 2304 and progress[0]["total"] == 8192 and progress[0]["rate"] > 0
    assert all(e["done"] <= 8192 for e in progress)


def test_a_hardware_step_that_goes_quiet_is_stopped(tmp_path):
    import threading

    release = threading.Event()

    class HungProc:
        killed = False

        def __init__(self):
            def lines():
                yield json.dumps({"type": "step", "id": "gather1"}) + "\n"
                release.wait(10)  # a CUDA kernel that never returns

            self.stdout = lines()
            self.returncode = None

        def kill(self):
            HungProc.killed = True
            release.set()

        terminate = kill

        def wait(self):
            return -9

    job = tuner.TuneJob(FakeManager(), state_dir=str(tmp_path), python="python", default_port=1919,
                        spawn=lambda argv: HungProc())
    with pytest.raises(RuntimeError, match="gather1"):
        job._hardware("/models/M", quiet_s=0.3)
    assert HungProc.killed


def test_a_second_start_while_running_is_refused(tmp_path):
    m = FakeManager()
    job = _job(tmp_path, m, FakeServe(m))
    job.state = "running"
    with pytest.raises(tuner.Busy):
        job.start("/models/M")


def test_cancel_restores_and_reports(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    job = _job(tmp_path, m, FakeServe(m))
    job._cancel.set()
    job._run("/models/M", True, 1919)
    assert job.state == "cancelled" and m.cfg == prev
