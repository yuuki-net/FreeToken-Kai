"""The console's hardware measurement: what this PC does with THIS model's experts.

Run by ``ft mgr`` as a child process (it imports torch; the manager must not):

    python -m freetoken.webui.hwbench --model ~/models/Ornith-1.5-35B-A3B-NVFP4

It prints one JSON object per line, so the page can draw every sample as it arrives:

    {"type": "plan", "steps": [...]}           the steps this run will take
    {"type": "step", "id": ...}                 a step starts
    {"type": "sample", "id": ..., "value": ...} one reading (GB/s unless the step says otherwise)
    {"type": "done", "id": ..., "value": ...}   the step's figure, plus details
    {"type": "skip", "id": ..., "reason": ...}  a step that cannot run here
    {"type": "result", ...}                     everything, and the flags derived from it

The kernels are upstream's ``ft bench bw`` ones (``moe/benchbw.py``), driven with the model's own
expert geometry instead of a canonical one, and the CPU side swept over thread counts. The
result is also merged into the ``ft bench bw`` profile for the GPU, which the engine reads for the
hybrid fetch split (``--moe-hybrid-max-fetch -1``)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

GiB = 1 << 30

# PCIe payload rate per lane after line coding, GB/s
_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def emit(kind: str, **fields) -> None:
    print(json.dumps({"type": kind, **fields}), flush=True)


# ------------------------------------------------------------------ what the model is
def expert_format(quant: str | None) -> str:
    q = (quant or "").lower()
    if q in ("modelopt", "nvfp4", "modelopt_fp4"):
        return "nvfp4"
    if q == "mxfp4":
        return "mxfp4_triton"
    if q in ("fp8",):
        return "fp8_block"
    return "bf16"


def workload_of(path: str):
    """The model's expert geometry as a benchbw Workload, or None for a dense model."""
    from freetoken.moe.benchbw import Workload

    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or {}
    get = lambda k, d=None: cfg.get(k, text.get(k, d))  # noqa: E731
    experts = get("num_experts") or get("n_routed_experts") or get("num_local_experts")
    if not experts:
        return None, None
    quant = (cfg.get("quantization_config") or text.get("quantization_config") or {})
    fmt = expert_format(quant.get("quant_method") or quant.get("quant_algo"))
    gpt_oss = cfg.get("model_type") == "gpt_oss"
    wl = Workload(
        name=os.path.basename(path.rstrip("/")),
        hidden=int(get("hidden_size")),
        inter=int(get("moe_intermediate_size") or get("intermediate_size")),
        experts=int(experts),
        top_k=int(get("num_experts_per_tok") or get("top_k") or 8),
        formats=(fmt,),
        activation="gpt_oss_swiglu" if gpt_oss else "silu",
        swiglu_limit=7.0 if gpt_oss else None,
    )
    return wl, fmt


# ------------------------------------------------------------------ steps
def gpu_links() -> list[dict]:
    fields = "index,name,uuid,memory.total,compute_cap,pcie.link.gen.max,pcie.link.gen.gpucurrent,pcie.link.width.max,pcie.link.width.current"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        num = lambda s: int(s) if s.isdigit() else None  # noqa: E731
        gen, width = num(p[6]) or num(p[5]), num(p[8]) or num(p[7])
        gpus.append({
            "index": int(p[0]), "name": p[1], "uuid": p[2], "total_bytes": int(float(p[3])) << 20,
            "compute_cap": float(p[4]) if p[4] else None,
            "pcie_gen": gen, "pcie_gen_max": num(p[5]), "pcie_width": width, "pcie_width_max": num(p[7]),
            "link_gbs": round(_LANE_GBS.get(gen, 0) * width, 2) if gen and width else None,
        })
    return gpus


def step_pcie(device_index: int, rounds: int = 8) -> dict:
    import torch

    from freetoken.moe.benchbw import measure_pcie_bw

    device = torch.device("cuda", device_index)
    sid = f"pcie{device_index}"
    emit("step", id=sid)
    h2d, d2h = [], []
    for _ in range(rounds):
        r = measure_pcie_bw(device, nbytes=256 << 20, iters=6)
        h2d.append(r["h2d_gbs"])
        d2h.append(r["d2h_gbs"])
        emit("sample", id=sid, value=round(r["h2d_gbs"], 2))
    torch.cuda.empty_cache()
    out = {"h2d_gbs": round(statistics.median(h2d), 2), "d2h_gbs": round(statistics.median(d2h), 2)}
    emit("done", id=sid, value=out["h2d_gbs"], **out)
    return out


def step_ram(rounds: int = 5) -> dict:
    from freetoken.moe.benchbw import measure_cpu_mem_bw

    emit("step", id="ram")
    vals, threads = [], None
    for _ in range(rounds):
        r = measure_cpu_mem_bw(iters=2)
        vals.append(r["bw_gbs"])
        threads = r["threads"]
        emit("sample", id="ram", value=round(r["bw_gbs"], 2))
    out = {"read_gbs": round(statistics.median(vals), 2), "threads": threads}
    emit("done", id="ram", value=out["read_gbs"], **out)
    return out


def _read_ahead_kb(path: str) -> int | None:
    try:
        dev = os.stat(path).st_dev
        base = f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}"
        for q in (f"{base}/queue/read_ahead_kb", f"{base}/../queue/read_ahead_kb"):
            if os.path.exists(q):
                with open(q) as fh:
                    return int(fh.read().strip())
    except (OSError, ValueError):
        pass
    return None


def step_ssd(model_path: str, budget_bytes: int = GiB + GiB // 2, max_seconds: float = 20.0) -> dict | None:
    names = [n for n in os.listdir(model_path) if n.endswith((".safetensors", ".gguf", ".bin"))]
    if not names:
        emit("skip", id="ssd", reason="no_weights")
        return None
    path = max((os.path.join(model_path, n) for n in names), key=os.path.getsize)
    size = os.path.getsize(path)
    emit("step", id="ssd")
    chunk = 16 << 20
    fd = os.open(path, os.O_RDONLY)
    try:
        start = max(0, size // 2 - budget_bytes // 2)
        # measure the disk, not the page cache: drop what is cached of this file first
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.lseek(fd, start, os.SEEK_SET)
        done, t0, mark, mark_t, rates = 0, time.perf_counter(), 0, time.perf_counter(), []
        while done < budget_bytes and time.perf_counter() - t0 < max_seconds:
            got = len(os.read(fd, chunk))
            if not got:
                break
            done += got
            if done - mark >= 128 << 20:
                now = time.perf_counter()
                rate = (done - mark) / (now - mark_t) / 1e9
                rates.append(rate)
                emit("sample", id="ssd", value=round(rate, 2))
                mark, mark_t = done, now
        wall = time.perf_counter() - t0
    finally:
        os.close(fd)
    fs = _fs_type(path)
    out = {"read_gbs": round(done / wall / 1e9, 2) if wall else None, "bytes": done, "fs": fs,
           "read_ahead_kb": _read_ahead_kb(path)}
    emit("done", id="ssd", value=out["read_gbs"], **out)
    return out


def _fs_type(path: str) -> str | None:
    best, fs = "", None
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 2 and os.path.realpath(path).startswith(parts[1]) and len(parts[1]) > len(best):
                    best, fs = parts[1], parts[2]
    except OSError:
        pass
    return fs


def thread_candidates(cores: int) -> list[int]:
    return sorted({t for t in (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, cores) if 1 <= t <= cores})


def step_cpu_moe(fmt: str, wl) -> dict:
    """CPU expert compute over thread counts: the knee, not the maximum, is the setting."""
    from freetoken.moe import benchbw as bb
    from freetoken.moe.cpu_executor import physical_core_cpus

    import logging

    cores = len(physical_core_cpus()) or (os.cpu_count() or 4)
    logging.getLogger("freetoken.moe.cpu_executor").setLevel(logging.WARNING)  # a line per executor
    emit("step", id="cpu_moe")
    eb = bb._expert_bytes(fmt, wl.hidden, wl.inter)
    E = bb._synth_experts(wl.experts, eb)
    banks = bb._cpu_moe_bank_sources(fmt, wl.hidden, wl.inter, E)
    sweep = {}
    for t in thread_candidates(cores):
        ex = bb._build_cpu_moe_executor(fmt, wl, banks, t, E)
        gbs = round(bb._time_cpu_moe(ex, wl, eb, E, 32), 2)
        del ex
        sweep[t] = gbs
        emit("sample", id="cpu_moe", value=gbs, threads=t)
    del banks
    out = {"sweep": {str(k): v for k, v in sweep.items()}, "best_gbs": max(sweep.values()),
           "threads": pick_threads(sweep), "cores": cores, "expert_bytes": eb}
    emit("done", id="cpu_moe", value=out["best_gbs"], **out)
    return out


def pick_threads(sweep: dict[int, float], share: float = 0.95) -> int:
    """The fewest threads within ``share`` of the best: more only takes cores from everything else."""
    best = max(sweep.values())
    return min(t for t, v in sweep.items() if v >= share * best)


def step_gather(fmt: str, wl, device_index: int, rounds: int = 4) -> dict:
    import torch

    from freetoken.moe.benchbw import measure_pcie_gather_bw

    sid = f"gather{device_index}"
    emit("step", id=sid)
    device = torch.device("cuda", device_index)
    vals = []
    for _ in range(rounds):
        r = measure_pcie_gather_bw(fmt, wl, device, iters=8)
        vals.append(r["bw_gbs"])
        emit("sample", id=sid, value=round(r["bw_gbs"], 2))
    torch.cuda.empty_cache()
    out = {"gbs": round(statistics.median(vals), 2)}
    emit("done", id=sid, value=out["gbs"], **out)
    return out


def step_overlap(fmt: str, wl, device_index: int, threads: int) -> dict:
    import torch

    from freetoken.moe.benchbw import measure_overlap_bw

    emit("step", id="overlap")
    r = measure_overlap_bw(fmt, wl, torch.device("cuda", device_index), num_threads=threads, seconds=3.0)
    torch.cuda.empty_cache()
    out = {"cpu_gbs": round(r["cpu_gbs"], 2), "pcie_gbs": round(r["pcie_gbs"], 2),
           "fetch_fraction": round(r["pcie_gbs"] / (r["pcie_gbs"] + r["cpu_gbs"]), 3) if r["cpu_gbs"] + r["pcie_gbs"] else None}
    emit("done", id="overlap", value=out["pcie_gbs"] + out["cpu_gbs"], **out)
    return out


# ------------------------------------------------------------------ what it means
def derive(m: dict, threshold: float = 2.0) -> list[dict]:
    """Flags the measurements decide, each with the numbers behind it (both languages)."""
    notes: list[dict] = []
    cpu, gathers = m.get("cpu_moe"), [g for g in (m.get("gather") or {}).values() if g]
    if cpu and gathers:
        pcie = min(g["gbs"] for g in gathers)
        ratio = cpu["best_gbs"] / pcie if pcie else 0
        if ratio > threshold:
            notes.append({"flag": "--moe-strategy", "value": "hybrid",
                          "why": f"CPU でのエキスパート計算（{cpu['best_gbs']:.1f} GB/s）が GPU への転送（{pcie:.1f} GB/s）の {ratio:.1f} 倍速いので、GPU に無いエキスパートは CPU で計算します。",
                          "why_en": f"Computing experts on the CPU ({cpu['best_gbs']:.1f} GB/s) is {ratio:.1f}x the transfer to the GPU ({pcie:.1f} GB/s), so experts not on the GPU are computed on the CPU."})
            best = cpu["best_gbs"]
            notes.append({"flag": "--moe-cpu-threads", "value": str(cpu["threads"]),
                          "why": f"{cpu['threads']} スレッドで最大（{best:.1f} GB/s）の 95% 以上が出ます。それ以上増やしてもほとんど速くならず、ほかの処理のコアを奪うだけです。",
                          "why_en": f"{cpu['threads']} threads reach 95% of the best ({best:.1f} GB/s). More barely helps and only takes cores from everything else."})
        else:
            notes.append({"flag": "--moe-strategy", "value": "offload",
                          "why": f"GPU への転送（{pcie:.1f} GB/s）が CPU でのエキスパート計算（{cpu['best_gbs']:.1f} GB/s）の 1/{threshold:.0f} より速いので、エキスパートは GPU に送って計算します。",
                          "why_en": f"The transfer to the GPU ({pcie:.1f} GB/s) beats computing experts on the CPU ({cpu['best_gbs']:.1f} GB/s) by more than 1/{threshold:.0f}, so experts are sent to the GPU."})
    elif gathers:
        notes.append({"flag": "--moe-strategy", "value": "offload",
                      "why": "この形式のエキスパートは CPU で計算できないので、GPU に送って計算します。",
                      "why_en": "Experts in this format cannot be computed on the CPU, so they are sent to the GPU."})
    ov = m.get("overlap")
    if ov and ov.get("fetch_fraction") is not None:
        notes.append({"flag": "--moe-hybrid-max-fetch", "value": "-1",
                      "why": f"同時に動かすと GPU への転送 {ov['pcie_gbs']:.1f} GB/s・CPU 計算 {ov['cpu_gbs']:.1f} GB/s でした。足りないエキスパートの {ov['fetch_fraction'] * 100:.0f}% を GPU に送り、残りを CPU で計算すると両方が同時に終わります（この値は測定の記録からエンジンが読みます）。",
                      "why_en": f"Run together, the transfer did {ov['pcie_gbs']:.1f} GB/s and the CPU {ov['cpu_gbs']:.1f} GB/s. Fetching {ov['fetch_fraction'] * 100:.0f}% of the missing experts and computing the rest on the CPU makes both finish together (the engine reads this from the measurement record)."})
    return notes


def update_bench_profile(gpu: dict, fmt: str, wl, m: dict) -> str | None:
    """Merge this model's figures into the GPU's ``ft bench bw`` profile, keeping other formats."""
    from freetoken.moe.bench_profile import default_profile_path
    from freetoken.moe.benchbw import _atomic_write_json

    cpu, g, ov = m.get("cpu_moe"), (m.get("gather") or {}).get(str(gpu["index"])), m.get("overlap")
    if not (g and cpu):
        return None
    path = default_profile_path(gpu.get("uuid"))
    try:
        with open(path) as fh:
            prof = json.load(fh)
    except (OSError, ValueError):
        prof = {"version": 4, "threshold": 2.0, "dtypes": {}, "dtype_kernels": {}, "workloads": {}}
    rec = "hybrid" if cpu["best_gbs"] > 2.0 * g["gbs"] else "offload"
    prof.setdefault("dtypes", {})[fmt] = rec
    prof.setdefault("dtype_kernels", {})[fmt] = {
        "expert_bytes": cpu.get("expert_bytes"), "cpu_moe_gbs": cpu["best_gbs"], "pcie_gather_gbs": g["gbs"],
        "cpu_moe_overlap_gbs": (ov or {}).get("cpu_gbs"), "pcie_gather_overlap_gbs": (ov or {}).get("pcie_gbs"),
        "ratio": round(cpu["best_gbs"] / g["gbs"], 3) if g["gbs"] else None, "recommended": rec,
        "note": f"ft mgr benchmark, {wl.name}",
    }
    prof["gpu"] = {"index": gpu["index"], "name": gpu["name"], "uuid": gpu.get("uuid")}
    prof["epoch"] = int(time.time())
    prof.setdefault("ceilings", {}).update({
        k: v for k, v in {
            "cpu_stream_read_gbs": (m.get("ram") or {}).get("read_gbs"),
            "pcie_linear_h2d_gbs": ((m.get("pcie") or {}).get(str(gpu["index"])) or {}).get("h2d_gbs"),
            "pcie_linear_d2h_gbs": ((m.get("pcie") or {}).get(str(gpu["index"])) or {}).get("d2h_gbs"),
        }.items() if v is not None
    })
    _atomic_write_json(path, prof)
    return path


# ------------------------------------------------------------------ run
def run(model_path: str) -> dict:
    wl, fmt = workload_of(model_path)
    gpus = gpu_links()
    cpu_capable = False
    if wl is not None:
        from freetoken.moe.benchbw import _CPU_MOE_FORMATS

        cpu_capable = fmt in _CPU_MOE_FORMATS
    steps = [{"id": "gpu", "unit": ""}]
    steps += [{"id": f"pcie{g['index']}", "unit": "GB/s", "gpu": g["index"], "max": g["link_gbs"]} for g in gpus]
    steps += [{"id": "ram", "unit": "GB/s"}, {"id": "ssd", "unit": "GB/s"}]
    if wl is not None:
        if cpu_capable:
            steps.append({"id": "cpu_moe", "unit": "GB/s"})
        steps += [{"id": f"gather{g['index']}", "unit": "GB/s", "gpu": g["index"]} for g in gpus]
        if cpu_capable and gpus:
            steps.append({"id": "overlap", "unit": "GB/s"})
    emit("plan", steps=steps, model={"name": wl.name if wl else os.path.basename(model_path), "format": fmt,
                                     "experts": wl.experts if wl else None, "top_k": wl.top_k if wl else None})

    m: dict = {"gpus": gpus, "format": fmt, "pcie": {}, "gather": {}}
    emit("step", id="gpu")
    emit("done", id="gpu", value=len(gpus), gpus=gpus)

    def guarded(sid, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001 -- one step failing must not end the run
            emit("skip", id=sid, reason="failed", message=str(exc)[:400])
            return None

    for g in gpus:
        m["pcie"][str(g["index"])] = guarded(f"pcie{g['index']}", step_pcie, g["index"])
    m["ram"] = guarded("ram", step_ram)
    m["ssd"] = guarded("ssd", step_ssd, model_path)
    if wl is not None:
        if cpu_capable:
            m["cpu_moe"] = guarded("cpu_moe", step_cpu_moe, fmt, wl)
        for g in gpus:
            m["gather"][str(g["index"])] = guarded(f"gather{g['index']}", step_gather, fmt, wl, g["index"])
        if cpu_capable and gpus and m.get("cpu_moe"):
            m["overlap"] = guarded("overlap", step_overlap, fmt, wl, gpus[0]["index"], m["cpu_moe"]["threads"])
    notes = derive(m)
    profile = None
    if wl is not None and gpus:
        try:
            profile = update_bench_profile(gpus[0], fmt, wl, m)
        except Exception:  # noqa: BLE001 -- the record is a convenience, the result stands without it
            profile = None
    result = {"measurements": m, "notes": notes, "bench_profile": bool(profile)}
    emit("result", **result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m freetoken.webui.hwbench")
    p.add_argument("--model", required=True)
    ns = p.parse_args(argv)
    try:
        run(os.path.expanduser(ns.model))
    except Exception as exc:  # noqa: BLE001
        emit("error", message=str(exc)[:1000])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
