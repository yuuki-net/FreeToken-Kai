"""The console's benchmark job (webui/tuner.py) and the decisions in the hardware bench (webui/hwbench.py).

The job runs against fakes: a manager that records lifecycle calls, a child process that prints
canned JSON lines, and a serve whose speed depends on the context it was started with. What is
pinned: the previous engine comes back, the measured flags replace the rule-of-thumb ones, a longer
context is kept only when the measurements allow it, and a model that will not load in float16 is
retried without it."""

from __future__ import annotations

import io
import json

import pytest

from freetoken.webui import hwbench, tuner

GiB = 1 << 30


# ------------------------------------------------------------------ pure decisions
def test_set_flags_overrides_adds_and_drops_in_place():
    args = ["--dtype", "float16", "--moe-cache-auto", "--moe-cpu-threads", "8", "--host", "0.0.0.0"]
    out = tuner.set_flags(args, {"--moe-cpu-threads": "6", "--kv-reserve-tokens": "65536", "--moe-collect-stats": None},
                          drop=("--dtype",))
    assert out == ["--moe-cache-auto", "--moe-cpu-threads", "6", "--host", "0.0.0.0",
                   "--kv-reserve-tokens", "65536", "--moe-collect-stats"]
    assert tuner.parse_flags(["--moe-hybrid-max-fetch", "-1", "--x=3"]) == [("--moe-hybrid-max-fetch", "-1"), ("--x", "3")]


def _geo(now=16384, kv=10880, slots=1104, per_expert=1775616, kv_max=249909):
    return {"num_pages": now, "page_size": 1, "moe_cache_size": slots,
            "unit_bytes": {"kv_per_token": kv, "moe_per_expert": per_expert},
            "limits": {"kv_tokens": {"max": kv_max}}}


def _ranks(free):
    return [{"gpu": {"free_bytes": int(free)}}]


def test_longer_context_from_free_vram_alone():
    c = tuner.longer_context(_geo(), _ranks(2.5 * GiB), cache_auto=False, model_max=262144)
    assert c["tokens"] == 131072 and c["experts_lost"] == 0


def test_longer_context_takes_at_most_a_quarter_of_an_auto_cache():
    # 0.2 GiB free on a 2060: 32k costs 96 slots, 64k 300 (27%) -> only 32k
    c = tuner.longer_context(_geo(), _ranks(0.2 * GiB), cache_auto=True, model_max=262144)
    assert c["tokens"] == 32768 and 0 < c["experts_lost"] <= 1104 // 4


def test_no_longer_context_without_room_or_past_the_model():
    assert tuner.longer_context(_geo(), _ranks(0.2 * GiB), cache_auto=False, model_max=262144) is None
    assert tuner.longer_context(_geo(), _ranks(8 * GiB), cache_auto=False, model_max=16384) is None
    assert tuner.longer_context(_geo(kv_max=20000), _ranks(8 * GiB), cache_auto=False, model_max=262144) is None


def test_keep_longer_only_when_speed_holds():
    a = {"ok": True, "decode_tps": 38.0, "prefill_tps": 580}
    assert tuner.keep_longer(a, {"ok": True, "decode_tps": 36.5, "prefill_tps": 540})
    assert not tuner.keep_longer(a, {"ok": True, "decode_tps": 34.0, "prefill_tps": 580})
    assert not tuner.keep_longer(a, {"ok": True, "decode_tps": 38.0, "prefill_tps": 500})
    assert not tuner.keep_longer(a, {"ok": False})


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
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
    from freetoken.moe.benchbw import Workload

    gpu = {"index": 0, "name": "NVIDIA GeForce RTX 2060", "uuid": "GPU-test"}
    path = default_profile_path(gpu["uuid"])
    import os
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

REC = {
    "flags": ["--dtype", "float16", "--moe-strategy", "hybrid", "--moe-cpu-layers", "auto", "--moe-cpu-threads", "8",
              "--moe-cache-auto", "--kv-reserve-tokens", "16384", "--max-seq-len-override", "16384"],
    "notes": [{"flag": f, "value": v, "why": "r", "why_en": "r"} for f, v in
              [("--dtype", "float16"), ("--moe-strategy", "hybrid"), ("--moe-cpu-layers", "auto"), ("--moe-cpu-threads", "8"),
               ("--moe-cache-auto", None), ("--kv-reserve-tokens", "16384"), ("--max-seq-len-override", "16384")]],
    "host": {"model": {"max_context": 262144}},
}


class FakeServe:
    """A serve whose speed and geometry follow the flags it was started with: ``speeds`` by
    --kv-reserve-tokens as (decode, prefill), ``budget`` and ``offload`` as (decode, prefill)
    factors for --prefill-chunk-budget 0.75 and --moe-strategy offload, ``fail`` a predicate on
    the flags that makes the prompt request error (a chunk that runs out of VRAM)."""

    def __init__(self, manager, speeds, free=2.5 * GiB, budget=(1.0, 1.0), offload=(1.0, 1.0), fail=None):
        self.m, self.speeds, self.free = manager, speeds, free
        self.budget, self.offload, self.fail = budget, offload, fail
        self.t = 0.0

    def flags(self):
        return dict(tuner.parse_flags(self.m.cfg[2]))

    def rates(self):
        f = self.flags()
        d, p = self.speeds[self.ctx()]
        if f.get("--prefill-chunk-budget") == "0.75":
            d, p = d * self.budget[0], p * self.budget[1]
        if f.get("--moe-strategy") == "offload":
            d, p = d * self.offload[0], p * self.offload[1]
        return d, p

    def clock(self):
        return self.t

    def ctx(self):
        args = self.m.cfg[2]
        return int(next(v for k, v in tuner.parse_flags(args) if k == "--kv-reserve-tokens"))

    def http(self, port, path, body=None, timeout=0):
        ctx = self.ctx()
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": "m"}]}
        if path == "/v1/cache/status":
            return {"geometry": _geo(now=ctx)}
        if path.startswith("/v1/kai/experts"):
            return {"layers": [{"active": 100, "miss": 10}], "ranks": _ranks(self.free)}
        if path == "/v1/completions":
            assert isinstance(body["prompt"], str)  # token-id prompts are refused by ft serve
            n = int(len(body["prompt"].split()) * 1.3)
            if body["max_tokens"] == 1:
                if self.fail and self.fail(self.flags()):
                    raise RuntimeError("HTTP Error 500: Internal Server Error")
                self.t += n / self.rates()[1]
            return {"usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]}}
        raise AssertionError(path)

    def stream(self, port, path, body):
        rate = self.rates()[0]
        for i in range(body["max_tokens"]):
            self.t += 1 / rate
            yield "data: " + json.dumps({"choices": [{"text": "x"}]}) + "\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}) + "\n"
        yield "data: [DONE]\n"


def _job(tmp_path, manager, serve):
    return tuner.TuneJob(manager, state_dir=str(tmp_path), python="python", default_port=1919,
                         recommend=lambda m: REC, spawn=lambda argv: FakeProc(HW_LINES),
                         http=serve.http, stream=serve.stream, clock=serve.clock, sleep=lambda s: None)


def _run(job, model="/models/M", trials=True):
    job.start(model, trials)
    job._thread.join(10)
    return job.status()


def _changes(r):
    return [(t["label"], (t.get("change") or {}).get("flag")) for t in r["trials"]]


def test_full_run_keeps_the_longer_context_and_brings_the_old_engine_back(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    serve = FakeServe(m, {16384: (38.0, 580.0), 131072: (37.0, 560.0)})
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    # one change per run: the budget (no faster here, not kept), then the context on the best so far
    assert _changes(r) == [("A", None), ("B", "--prefill-chunk-budget"), ("C", "--kv-reserve-tokens")]
    assert r["chosen"] == "C" and r["candidate"]["tokens"] == 131072
    assert "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))
    assert [n for n in r["notes"] if n["flag"] == "--prefill-chunk-budget"][0]["rejected"]
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--kv-reserve-tokens"] == flags["--max-seq-len-override"] == "131072"
    assert flags["--moe-cpu-threads"] == "6"  # measured beats the rule's 8
    assert "--moe-collect-stats" not in flags
    assert [n for n in r["notes"] if n["flag"] == "--moe-cpu-threads"][0]["source"] == "measured"
    assert m.calls[0] == ("stop",) and m.calls[-1] == ("switch", *prev)
    assert m.cfg == prev
    assert json.load(open(tmp_path / "tune" / "M.json"))["chosen"] == "C"
    kinds = {e["k"] for e in st["events"]}
    assert {"phase", "hw", "trial", "sample"} <= kinds
    decode = [t["decode_tps"] for t in r["trials"]]
    assert decode[0] == pytest.approx(38.0, rel=0.01) and decode[2] == pytest.approx(37.0, rel=0.01)


def test_longer_context_that_slows_generation_is_not_kept(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, {16384: (38.0, 580.0), 131072: (30.0, 560.0)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert r["chosen"] == "A"
    assert dict(tuner.parse_flags(r["args"]))["--kv-reserve-tokens"] == "16384"
    assert m.cfg is None  # nothing was running before: nothing is left running


def test_a_faster_prefill_budget_is_kept_and_carried_into_the_context_run(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, {16384: (38.0, 520.0), 131072: (37.5, 510.0)}, budget=(1.0, 1.35))
    r = _run(_job(tmp_path, m, serve))["result"]
    assert r["chosen"] == "C"
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--prefill-chunk-budget"] == "0.75" and flags["--kv-reserve-tokens"] == "131072"
    assert dict(tuner.parse_flags(r["trials"][2]["args"]))["--prefill-chunk-budget"] == "0.75"
    note = [n for n in r["notes"] if n["flag"] == "--prefill-chunk-budget"][0]
    assert note["source"] == "measured" and not note.get("rejected")


def test_a_run_that_fails_mid_measurement_is_not_kept_and_the_job_goes_on(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, {16384: (38.0, 580.0), 131072: (37.0, 560.0)},
                      fail=lambda f: f.get("--prefill-chunk-budget") == "0.75")
    st = _run(_job(tmp_path, m, serve))
    assert st["state"] == "done", st["error"]
    r = st["result"]
    assert r["trials"][1]["ok"] is False and "500" in r["trials"][1]["error"]
    assert "--prefill-chunk-budget" not in dict(tuner.parse_flags(r["args"]))
    assert r["chosen"] == "C"


def _hw_lines(cpu, gather):
    lines = [dict(x) for x in HW_LINES]
    lines[-1] = dict(lines[-1], measurements={"cpu_moe": {"best_gbs": cpu}, "gather": {"0": {"gbs": gather}}})
    return lines


def test_the_other_strategy_is_tried_only_when_the_kernels_were_close(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, {16384: (20.0, 500.0), 131072: (19.0, 480.0)}, offload=(1.10, 1.0))
    job = _job(tmp_path, m, serve)
    job._spawn = lambda argv: FakeProc(_hw_lines(cpu=12.0, gather=6.0))  # 2.0x: close
    r = _run(job)["result"]
    assert _changes(r)[-1] == ("D", "--moe-strategy")
    flags = dict(tuner.parse_flags(r["args"]))
    assert flags["--moe-strategy"] == "offload" and "--moe-cpu-threads" not in flags and "--moe-cpu-layers" not in flags

    m2 = FakeManager()
    job2 = _job(tmp_path / "far", m2, FakeServe(m2, {16384: (20.0, 500.0), 131072: (19.0, 480.0)}))
    job2._spawn = lambda argv: FakeProc(_hw_lines(cpu=30.3, gather=6.3))  # 4.8x: the kernels decide
    r2 = _run(job2)["result"]
    assert "--moe-strategy" not in [c for _, c in _changes(r2)]


def test_float16_that_does_not_load_is_dropped(tmp_path):
    m = FakeManager()
    m.fail_with = {"float16"}
    serve = FakeServe(m, {16384: (38.0, 580.0), 131072: (37.0, 560.0)})
    st = _run(_job(tmp_path, m, serve))
    r = st["result"]
    assert st["state"] == "done", st["error"]
    assert "--dtype" not in dict(tuner.parse_flags(r["args"]))
    assert [n for n in r["notes"] if n["flag"] == "--dtype"][0].get("removed")
    assert r["trials"][0]["ok"] is False


def test_hardware_only_does_not_start_a_server(tmp_path):
    m = FakeManager()
    serve = FakeServe(m, {})
    st = _run(_job(tmp_path, m, serve), trials=False)
    assert st["state"] == "done"
    assert not [c for c in m.calls if c[0] == "switch"]
    assert dict(tuner.parse_flags(st["result"]["args"]))["--moe-strategy"] == "hybrid"


def test_offload_drops_the_cpu_flags(tmp_path):
    lines = [dict(x) for x in HW_LINES]
    lines[-1] = {"type": "result", "measurements": {}, "notes": [{"flag": "--moe-strategy", "value": "offload", "why": "w", "why_en": "w"}]}
    m = FakeManager()
    serve = FakeServe(m, {})
    job = _job(tmp_path, m, serve)
    job._spawn = lambda argv: FakeProc(lines)
    st = _run(job, trials=False)
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


def test_prompt_progress_is_skipped_on_a_serve_without_counters(tmp_path):
    def http(port, path, body=None, timeout=0):
        if path.startswith("/v1/kai/experts"):
            return {"ranks": [{}]}
        return {"usage": {"prompt_tokens": 10}}

    job = tuner.TuneJob(FakeManager(), state_dir=str(tmp_path), python="python", default_port=1919, http=http)
    assert job._prefill_once("A", 1919, {"prompt": "x"}, 10)["usage"]["prompt_tokens"] == 10


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
    job = _job(tmp_path, m, FakeServe(m, {}))
    job.state = "running"
    with pytest.raises(tuner.Busy):
        job.start("/models/M")


def test_cancel_restores_and_reports(tmp_path):
    prev = ("/models/Old", 1919, ["--old"])
    m = FakeManager(running=prev)
    job = _job(tmp_path, m, FakeServe(m, {}))
    job._cancel.set()
    job._run("/models/M", True, 1919)
    assert job.state == "cancelled" and m.cfg == prev
