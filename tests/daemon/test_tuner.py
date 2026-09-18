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
import os

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
    drafted, base_both = _t(560, 26.0, code=45.0), _t(600, 30.0, code=30.0)
    assert tuner.keep(base_both, drafted, mtp, "code") and not tuner.keep(base_both, drafted, mtp, "prose")
    assert tuner.keep(base_both, drafted, mtp, "both")  # sqrt(26*45)=34.2 > 30.9
    assert not tuner.keep(base, drafted, mtp, "code")  # no code figure for the base: prose against prose
    # a base measured on code and prose against a run measured on prose only: prose against prose
    base_code = _t(260, 18.18, code=18.34)
    assert tuner.keep(base_code, _t(569, 17.75), gen, "code")  # the 3060's 16-bit KV cache
    assert not tuner.close_on_generation(base_code, _t(260, 17.9), gen, "code")  # 98% of prose: no swing to check
    # tried for generation, kept for prompt processing: a 16-bit KV cache on two 3060s went 251 -> 517
    assert tuner.keep(_t(251, 18.4), _t(517, 18.4), gen, "prose")
    assert not tuner.keep(_t(251, 18.4), _t(517, 17.0), gen, "prose")  # but not at 8% of generation


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


def test_bench_profiles_say_which_version_measured_them(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = tmp_path / "freetoken" / "benchbw"
    os.makedirs(d)
    for uuid, extra in (("GPU-a", {}), ("GPU-b", {"freetoken_version": "0.0.1"})):
        with open(d / f"{uuid}.json", "w") as fh:
            json.dump({"version": 4, "epoch": 1, "gpu": {"index": 0, "name": "RTX", "uuid": uuid}, "dtypes": {"nvfp4": "hybrid"}, **extra}, fh)
    tuner.stamp_profile(str(d / "GPU-a.json"))
    doc = tuner.bench_profiles()
    assert doc["current"] == tuner.FT_VERSION
    assert [(p["uuid"], p["version"]) for p in doc["profiles"]] == [("GPU-a", tuner.FT_VERSION), ("GPU-b", "0.0.1")]
    from freetoken.moe.bench_profile import load_backend_recommendation

    assert load_backend_recommendation("nvfp4", "RTX", gpu_uuid="GPU-a") == "hybrid"  # the stamp does not disturb the reader

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


ALL_MODULES = {"vllm", "flashinfer"}


def _keys(args, facts, hw, mode, modules=ALL_MODULES):
    return [c.key for c in search.plan(args, facts, hw, mode, modules=modules)]


def test_kernels_that_cannot_run_here_are_not_tried_and_say_why():
    turing = {"measurements": dict(HW_2060["measurements"], gpus=[{"index": 0, "compute_cap": 7.5}])}
    keys = _keys(ORNITH_ARGS, ORNITH, turing, "thorough", modules=set())
    assert "kernel_moe_triton" in keys
    assert not {"kernel_moe_marlin", "kernel_moe_b12x", "kernel_linear_marlin"} & set(keys)
    why = {x["flag"]: x["why_en"] for x in search.unavailable(ORNITH, turing, modules=set())}
    assert "vLLM" in why["--quant-backend moe.nvfp4=marlin"] and "sm_75" in why["--quant-backend moe.nvfp4=b12x"]
    blackwell = {"measurements": {"gpus": [{"index": 0, "compute_cap": 12.0}]}}
    assert "kernel_moe_b12x" in _keys(ORNITH_ARGS, ORNITH, blackwell, "standard")


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
        self.refuse_stop = 0  # how many stops fail their accounting request (a dead backend)

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
        if self.refuse_stop and self.cfg is not None and not force:
            self.refuse_stop -= 1
            raise RuntimeError("prepare-stop request failed: <urlopen error timed out>")
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

    def __init__(self, manager, base=(30.0, 600.0), effects=None, fail=None, free=2.5 * GiB, kv_extra=0, lucky=None):
        self.m, self.base, self.effects, self.fail, self.free = manager, base, effects or {}, fail, free
        self.kv_extra = kv_extra  # KV pages beyond the advertised context, as the allocator rounds up
        self.lucky = lucky or {}  # "--flag=value" -> factors on the first start with it only
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
        starts = [c for c in self.m.calls if c[0] == "switch"]
        for key, (dp, pp, dc) in self.lucky.items():
            k, v = key.split("=", 1)
            if f.get(k) == v and sum(1 for c in starts if dict(tuner.parse_flags(c[3])).get(k) == v) == 1:
                prose, prefill, code = prose * dp, prefill * pp, code * dc
        return prose, prefill, code

    def http(self, port, path, body=None, timeout=0):
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": "m"}]}
        if path == "/v1/cache/status":
            return {"geometry": _geo(now=self.ctx() + self.kv_extra)}
        if path.startswith("/v1/kai/experts"):
            return {"layers": [{"active": 100, "miss": 10}], "ranks": [{"gpu": {"free_bytes": int(self.free)}}]}
        if path == "/v1/completions":
            assert isinstance(body["prompt"], str)  # token-id prompts are refused by ft serve
            n = int(len(body["prompt"]) / 4)
            limit = int(self.flags().get("--max-seq-len-override") or 1 << 30)
            if n + body["max_tokens"] > limit:
                raise RuntimeError("HTTP Error 400: Bad Request")
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


class FourCharTokenizer:
    """One token per four characters, as FakeServe counts them."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [text[i:i + 4] for i in range(0, len(text), 4)]}

    def decode(self, ids):
        return "".join(ids)


def test_prompts_are_cut_to_the_token_count_not_estimated():
    tok = FourCharTokenizer()
    assert tuner.fit_tokens(tok, "x" * 1000, 100) == "x" * 400
    assert tuner.fit_tokens(tok, "short", 100) == "short"
    assert tuner.fit_tokens(None, "anything", 1) == "anything"


CORPUS = {"prose": "FreeToken docs. " * 4000, "code": "def f():\n    return 1\n" * 3000}


def _job(tmp_path, manager, serve, rec=DENSE_REC, lines=HW_LINES):
    return tuner.TuneJob(manager, state_dir=str(tmp_path), python="python", default_port=1919,
                         recommend=lambda m: rec, spawn=lambda argv: FakeProc(lines), corpus=CORPUS, tokenizer=lambda m: FourCharTokenizer(),
                         http=serve.http, stream=serve.stream, clock=serve.clock, sleep=lambda s: None, gpu_indices=lambda: [0])


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
    plans = [e for e in st["events"] if e["k"] == "plan"]
    assert not plans[0]["update"] and plans[-1]["update"] and plans[-1]["items"] == []  # sent before every run, empty at the end
    assert plans[0]["eta_s"] is not None and all(p["eta_s"] >= 0 for p in plans)  # the time left, from the runs so far


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


def test_a_win_is_measured_again_and_a_lucky_run_is_not_kept(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, lucky={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0)}, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    # lucky first run, a miss, and a third run that misses too: two out of three missed
    assert _tried(r)[:4] == [("A", None, "base"), ("B", "budget_075", "rejected"), ("C", "budget_075", "rejected"),
                             ("D", "budget_075", "rejected")]
    assert r["trials"][2]["recheck"] and "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))
    decisions = [e["decision"] for e in _run_events(m, serve, tmp_path) if e["k"] == "decision"][:3]
    assert decisions == ["recheck", "tiebreak", "rejected"]


def _run_events(m, serve, tmp_path):
    m2 = FakeManager()
    serve2 = FakeServe(m2, lucky=serve.lucky, effects=serve.effects)
    return _run(_job(tmp_path / "again", m2, serve2))["events"]


def test_an_unlucky_check_is_outvoted_by_a_third_run(tmp_path):
    m = FakeManager()

    class Disturbed(FakeServe):
        def rates(self):
            prose, prefill, code = super().rates()
            starts = [c for c in self.m.calls if c[0] == "switch"]
            # the second start with the budget (the check) lands on a busy PC
            if dict(tuner.parse_flags(starts[-1][3])).get("--prefill-chunk-budget") == "0.75" and \
                    sum(1 for c in starts if dict(tuner.parse_flags(c[3])).get("--prefill-chunk-budget") == "0.75") == 2:
                return prose * 0.8, prefill * 0.85, code * 0.8
            return prose, prefill, code

    serve = Disturbed(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.15, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert _tried(r)[:4] == [("A", None, "base"), ("B", "budget_075", "kept"), ("C", "budget_075", "kept"), ("D", "budget_075", "kept")]
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"
    assert [e["decision"] for e in st["events"] if e["k"] == "decision"][:2] == ["recheck", "tiebreak"]


def test_a_change_that_holds_twice_needs_no_third_run(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert [k for _, k, _ in _tried(r)].count("budget_075") == 2


def test_a_confirmed_win_is_kept_at_the_slower_of_its_two_runs(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)},
                      lucky={"--prefill-chunk-budget=0.75": (1.0, 1.2, 1.0)})
    r = _run(_job(tmp_path, m, serve))["result"]
    assert _tried(r)[:3] == [("A", None, "base"), ("B", "budget_075", "kept"), ("C", "budget_075", "kept")]
    note = [n for n in r["notes"] if n["flag"] == "--prefill-chunk-budget"][0]
    assert "780" in note["why_en"] and "936" not in note["why_en"]  # 600 x 1.3, not the lucky 600 x 1.56


def test_the_prompt_fits_the_advertised_context_when_the_kv_pages_are_rounded_up(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, kv_extra=8192, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    assert all(t["ok"] for t in st["result"]["trials"])


def test_a_measurement_that_fails_does_not_drop_float16(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, fail=lambda f: True)
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "error"
    assert [t for t in st["result"]["trials"] if t.get("change")] == [] if st.get("result") else True
    assert all("--dtype" in c[3] for c in m.calls if c[0] == "switch")


def test_the_recommendation_follows_the_last_benchmark(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.3, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _job(tmp_path, m, serve)
    result = _run(job)["result"]
    rec = json.loads(json.dumps(DENSE_REC))
    rec["notes"].append({"flag": "--new-rule", "value": "1", "why": "r", "why_en": "r"})
    out = tuner.with_benchmark(rec, job.last("/models/M"))
    assert out["flags"] == result["args"] and out["benchmark"]["trials"] == len(result["trials"])
    assert out["benchmark"]["version"] == out["benchmark"]["current"] == tuner.FT_VERSION
    by = {n["flag"]: n for n in out["notes"]}
    assert by["--prefill-chunk-budget"]["value"] == "0.75" and by["--prefill-chunk-budget"]["source"] == "measured"
    assert "--new-rule" in by  # a rule the benchmark never saw is still offered
    assert not [n for n in out["notes"] if n.get("rejected")]
    # every recommended flag is carried by the benchmark's flags: a profile saved from it has nothing to apply
    flags = dict(tuner.parse_flags(result["args"]))
    assert all((n["flag"] not in flags) if n.get("removed") else n["flag"] in flags for n in out["notes"] if n["flag"] != "--new-rule")
    info = dict(result, notes=result["notes"] + [{"flag": "--moe-hybrid-max-fetch", "value": "-1", "why": "w", "why_en": "w",
                                                   "source": "measured", "info": True}])
    assert "--moe-hybrid-max-fetch" not in {n["flag"] for n in tuner.with_benchmark(rec, info)["notes"]}
    assert tuner.with_benchmark(rec, None) is rec and tuner.with_benchmark(rec, {"error": "x", "args": ["--a"]}) is rec


def test_a_change_dropped_for_generation_alone_is_judged_against_the_base_measured_again(tmp_path):
    m = FakeManager()
    # the budget gains 20% prefill; its start lands on a slow patch of the PC (generation -12%), and the
    # base started right after is just as slow: the change is kept

    class Swinging(FakeServe):
        def rates(self):
            prose, prefill, code = super().rates()
            starts = len([c for c in self.m.calls if c[0] == "switch"])
            return (prose * 0.88, prefill, code * 0.88) if starts in (2, 3) else (prose, prefill, code)

    serve = Swinging(m, effects={"--prefill-chunk-budget=0.75": (1.0, 1.2, 1.0), "--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    r = _run(_job(tmp_path, m, serve))["result"]
    got = _tried(r)[:4]
    assert got[1] == ("B", "budget_075", "kept") and got[2] == ("C", "base_again", "base") and got[3][1:] == ("budget_075", "kept")
    assert dict(tuner.parse_flags(r["args"]))["--prefill-chunk-budget"] == "0.75"


def test_close_on_generation_only_when_generation_alone_fell():
    c = search.Candidate("x", {"--a": "1"})
    base = _t(600, 30.0)
    assert tuner.close_on_generation(base, _t(700, 27.0), c, "prose")       # prefill +17%, generation -10%
    assert not tuner.close_on_generation(base, _t(700, 20.0), c, "prose")   # generation -33%: not a swing
    assert not tuner.close_on_generation(base, _t(600, 30.5), c, "prose")   # nothing fell
    assert not tuner.close_on_generation(base, _t(500, 27.0), c, "prose")   # prefill fell too


def test_every_run_reads_the_same_prompts(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    seen = []
    http = serve.http

    def record(port, path, body=None, timeout=0):
        if path == "/v1/completions" and body["max_tokens"] == 1:
            seen.append((dict(tuner.parse_flags(m.cfg[2])).get("--prefill-chunk-budget"), body["prompt"].split("\n", 1)))
        return http(port, path, body, timeout)

    serve.http = record
    _run(_job(tmp_path, m, serve))
    base = [b for flag, (_, b) in seen if flag is None][:2]
    budget = [b for flag, (_, b) in seen if flag == "0.75"][:2]
    assert base and base == budget  # the same text, run for run
    heads = [h for _, (h, _) in seen]
    assert len(set(heads)) == len(heads)  # but never the same first line


def test_a_server_that_cannot_answer_its_stop_is_replaced_by_force(tmp_path):
    m = FakeManager()
    m.refuse_stop = 1
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    assert all(t["ok"] for t in st["result"]["trials"])
    assert any(e["k"] == "log" and "force" in e["msg"] for e in st["events"])


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
    assert all(t.get("decode_code_tps") for t in r["trials"] if t.get("ok"))  # every run, so the figures compare

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


def test_upstreams_bench_runs_first_on_every_gpu_and_is_stamped(tmp_path):
    prof = tmp_path / "GPU-x.json"
    started = []

    def spawn(argv, env=None):
        started.append(argv[2:5])
        if "bench" in argv:
            assert env["FREETOKEN_BENCH_PROGRESS"] == "1" and argv[-2:] == ["--gpu", str(len(started) - 1)]
            prof.write_text(json.dumps({"version": 4, "gpu": {"name": "RTX"}}))
            return _TextProc(["FTBENCH 0 2 ceilings", "FTBENCH 1 2 dtype:nvfp4", f"FTBENCH_OUT {prof}"])
        return FakeProc(HW_LINES)

    m = FakeManager()
    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, recommend=lambda x: DENSE_REC,
                        spawn=spawn, gpu_indices=lambda: [0, 1], http=FakeServe(m).http)
    st = _run(job, trials=False)
    assert st["state"] == "done", st["error"]
    assert started == [["freetoken.cli", "bench", "bw"], ["freetoken.cli", "bench", "bw"], ["freetoken.webui.hwbench", "--model", "/models/M"]]
    assert json.loads(prof.read_text())["freetoken_version"] == tuner.FT_VERSION
    assert [u["gpu"] for u in st["result"]["upstream"]] == [0, 1] and st["result"]["freetoken_version"] == tuner.FT_VERSION
    samples = [e for e in st["events"] if e["k"] == "hw" and e.get("id") == "upstream0" and e["kind"] == "sample"]
    assert [s["value"] for s in samples] == [0, 50]
    phases = [e["phase"] for e in st["events"] if e["k"] == "phase"]
    assert phases.index("upstream") < phases.index("hw")


class _TextProc(FakeProc):
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(x + "\n" for x in lines))
        self.returncode = 0


def test_a_failing_upstream_bench_stops_the_benchmark(tmp_path):
    class Failed(_TextProc):
        def wait(self):
            return 1

    m = FakeManager()
    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, recommend=lambda x: DENSE_REC,
                        spawn=lambda argv, env=None: Failed(["RuntimeError: no CUDA device"]), gpu_indices=lambda: [0])
    st = _run(job, trials=False)
    assert st["state"] == "error" and "no CUDA device" in st["error"]


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

    job = tuner.TuneJob(FakeManager(running=("/m", 1919, [])), state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    doc = job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert doc["usage"]["prompt_tokens"] == 8192
    progress = [e for e in job.events if e["k"] == "progress"]
    assert len(progress) >= 2
    assert progress[0]["done"] == 2304 and progress[0]["total"] == 8192 and progress[0]["rate"] > 0
    assert all(e["done"] <= 8192 for e in progress)


def test_a_prompt_to_a_server_that_exited_is_not_waited_on(tmp_path):
    import threading
    import time as _time

    release = threading.Event()
    m = FakeManager(running=("/m", 1919, []))

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            return {"ranks": []}
        release.wait(30)  # the reply that never comes
        return {"usage": {"prompt_tokens": 1}}

    job = tuner.TuneJob(m, state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    threading.Timer(1.5, lambda: setattr(m, "cfg", None)).start()
    t0 = _time.monotonic()
    with pytest.raises(RuntimeError, match="exited"):
        job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert _time.monotonic() - t0 < 10
    release.set()


def test_a_prompt_that_stops_progressing_is_given_up(tmp_path):
    import threading

    release = threading.Event()
    clock = {"t": 0.0}

    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            clock["t"] += 20  # every look is 20 s later; the counter never moves again
            return {"ranks": [{"counters": {"prefill_new_tokens": 4096}}]}
        release.wait(30)
        return {"usage": {"prompt_tokens": 1}}

    job = tuner.TuneJob(FakeManager(running=("/m", 1919, [])), state_dir=str(tmp_path), python="python", default_port=1919,
                        http=http, clock=lambda: clock["t"])
    with pytest.raises(RuntimeError, match="no progress"):
        job._prefill_once("A", 1919, {"prompt": "x"}, 8192)
    assert clock["t"] <= tuner.PROMPT_STALL_S + 60
    release.set()


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


# ------------------------------------------------------------------ picking up from the last result
def _no_hw(argv, env=None):
    raise AssertionError(f"picking up must not measure the hardware again: {argv}")


def _pickup_job(tmp_path, m, serve, rec=DENSE_REC):
    job = _job(tmp_path, m, serve, rec=rec)
    return job


def test_picking_up_runs_only_what_the_last_run_did_not_try(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    first = _run(job, mode="standard")["result"]
    assert first["tried"] and "budget_075" in first["tried"]
    expected = [x["key"] for x in job.pending("/models/M", "thorough")["items"]]
    assert expected, "the thorough plan should hold candidates a standard run did not try"
    assert job.pending("/models/M", "standard")["items"] == []

    job._spawn = _no_hw
    job.start("/models/M", True, mode="thorough", from_last=True)
    job._thread.join(30)
    st = job.status()
    assert st["state"] == "done", st["error"]
    r = st["result"]
    keys = [t.get("key") for t in r["trials"] if t.get("decision") != "base"]
    assert [k for k in keys if not k.startswith("context_")] == expected
    assert not any(k.startswith("context_") for k in keys)  # nothing new kept: the last tiers stand
    assert r["from_last"]["finished"] == first["finished"] and r["hw"] == first["hw"]
    assert set(first["tried"]) | set(expected) <= set(r["tried"])
    base = [t for t in r["trials"] if t.get("decision") == "base"][0]
    assert dict(tuner.parse_flags(base["args"]))["--kv-reserve-tokens"] == dict(tuner.parse_flags(first["args"]))["--kv-reserve-tokens"]
    assert json.load(open(tmp_path / "tune" / "M.json"))["from_last"]  # the saved result is this one
    # and a third pick-up has nothing left
    assert job.pending("/models/M", "thorough")["items"] == []


def test_a_new_win_when_picking_up_runs_the_context_tiers_again(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    _run(job, mode="standard")
    new = job.pending("/models/M", "thorough")["items"][0]
    flag, value = new["change"]["flag"], new["change"]["value"]
    serve.effects[f"{flag}={value}" if value is not None else flag] = (1.0, 1.4, 1.0)
    job._spawn = _no_hw
    job.start("/models/M", True, mode="thorough", from_last=True)
    job._thread.join(30)
    r = job.status()["result"]
    assert any(t.get("key") == new["key"] and t.get("decision") == "kept" for t in r["trials"])
    assert any((t.get("key") or "").startswith("context_") for t in r["trials"])
    assert new["key"] in r["kept"]


def test_nothing_to_pick_up_from_is_refused_before_anything_stops(tmp_path):
    m = FakeManager(running=("/models/Old", 1919, ["--old"]))
    job = _pickup_job(tmp_path, m, FakeServe(m))
    with pytest.raises(tuner.NoLastResult) as exc:
        job.start("/models/M", True, from_last=True)
    assert str(exc.value) == "no_last" and m.calls == []  # the running engine was left alone
    assert job.pending("/models/M")["available"] is False


def test_a_result_saved_before_the_record_still_counts_its_trials():
    old = {"trials": [{"key": None, "decision": "base"}, {"key": "budget_075", "decision": "kept"},
                      {"key": "max_prefill_16k", "decision": "rejected"}, {"key": "base_again", "decision": "base"}]}
    assert tuner.prior_trials(old) == ({"budget_075", "max_prefill_16k"}, {"budget_075"})
    assert tuner.last_refusal({"args": ["--x"], "hw": {}, "trials": []}, [0]) == "last_incomplete"
    assert tuner.last_refusal({"args": ["--x"], "hw": {"a": 1}, "upstream": [{}], "trials": [{"decision": "base"}]}, [0, 1]) \
        == "last_other_gpus"


def test_a_kept_change_closes_the_one_that_would_undo_it_in_an_older_result():
    # an older result has no record of groups: the plan its first settings opened says that the kept
    # strategy_offload closed the "strategy" group, so strategy_hybrid -- offered by the final
    # (offload) settings -- is not listed as untested
    rec = _moe_rec()
    facts, hw = tuner.host_facts(rec), {"measurements": {}}
    base = list(rec["flags"])
    final = tuner.set_flags(base, {"--moe-strategy": "offload"}, drop=("--moe-cpu-layers", "--moe-cpu-threads"))
    offered = {c.key: c.group for c in search.plan(final, facts, hw, "thorough")}
    assert offered.get("strategy_hybrid") == "strategy"
    prev = {"base_args": base, "args": final, "trials": [{"key": None, "decision": "base"},
                                                         {"key": "strategy_offload", "decision": "kept"}]}
    tried, _ = tuner.prior_trials(prev)
    assert "strategy" in tuner.closed_groups(prev, tried, facts, hw)


def test_a_candidate_the_settings_already_carry_is_not_listed(tmp_path):
    # four prefill pieces set the chunk ceiling to 16384, so "raise the ceiling to 16384" changes
    # nothing: the run skips it without a start, and the list shown before must not count it
    m = FakeManager()
    job = _pickup_job(tmp_path, m, FakeServe(m))
    args = DENSE_REC["flags"] + ["--max-prefill-length", "16384"]
    os.makedirs(tmp_path / "tune")
    with open(tmp_path / "tune" / "M.json", "w") as fh:
        json.dump({"args": args, "base_args": DENSE_REC["flags"], "hw": {"measurements": {}}, "upstream": [{"gpu": 0}],
                   "trials": [{"key": None, "decision": "base"}], "tried": ["budget_075"], "kept": [],
                   "finished": 1.0, "mode": "standard"}, fh)
    items = [x["key"] for x in job.pending("/models/M", "thorough")["items"]]
    assert "max_prefill_16k" not in items


def test_picking_up_keeps_the_earlier_runs_on_record(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, effects={"--kv-reserve-tokens=32768": (0.5, 1.0, 0.5)})
    job = _pickup_job(tmp_path, m, serve)
    first = _run(job, mode="standard")["result"]
    job._spawn = _no_hw
    for _ in range(2):
        job.start("/models/M", True, mode="thorough", from_last=True)
        job._thread.join(30)
    r = job.status()["result"]
    assert [e["finished"] for e in r["earlier"]][0] == first["finished"] and len(r["earlier"]) == 2
    assert r["earlier"][0]["trials"] == json.loads(json.dumps(first["trials"]))  # as saved
