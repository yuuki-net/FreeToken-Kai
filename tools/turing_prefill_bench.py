"""Where does prefill time go on this GPU? Times each kernel of the Qwen3.5-MoE prefill path
at a real prompt length, in fp16 and bf16, and prints ms and effective TFLOPS per kernel.
Each case runs in its own subprocess. Stop `ft serve` first (the bench needs ~1.5 GB of VRAM).

    ~/FreeToken-src/.venv-dev/bin/python turing_prefill_bench.py            # T=2785
    ~/FreeToken-src/.venv-dev/bin/python turing_prefill_bench.py --tokens 512

Shapes: hidden 2048, 256 experts top-8 (I=512), GDN 16/32 heads of 128, attention 16 q / 2 kv
heads of 256, fp8 in_proj N=12288 K=2048, out_proj N=2048 K=4096.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback

H, I, E, TOPK = 2048, 512, 256, 8
KH, VH, D = 16, 32, 128
QH, KVH, HD = 16, 2, 256

CASES = [
    "cublas_fp16_matmul", "cublas_bf16_matmul",
    "moe_check_arith_fp16",                       # arithmetic vs LUT dequant vs torch reference
    "moe_prefill_lut_fp16", "moe_prefill_arith_fp16", "moe_prefill_scratch_fp16", "moe_prefill_scratch_bf16",
    "fp8_gemm_fp16", "fp8_gemm_bf16",
    "attn_extend_fp16", "attn_extend_bf16",
    "gdn_prefill_fp16", "gdn_prefill_bf16",
    "nvfp4_dense_fp16",
    "moe_align",
]


def _time(fn, warmup=1, iters=3):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _report(name, ms, flops):
    print(f"{name}: {ms:8.1f} ms   {flops / ms / 1e9:6.2f} TFLOPS")


def _nvfp4(dev, n, k):
    import torch

    packed = torch.randint(0, 256, (E, n, k // 2), device=dev, dtype=torch.uint8)
    scale = (torch.rand(E, n, k // 16, device=dev) * 0.75 + 0.25).to(torch.float8_e4m3fn)
    glob = torch.full((E, n), 0.01, device=dev, dtype=torch.float16)
    return packed, scale, glob


def c_cublas(dev, T, dt):
    import torch

    a = torch.randn(T, H, device=dev, dtype=dt)
    w = torch.randn(12288, H, device=dev, dtype=dt)
    ms = _time(lambda: a @ w.t())
    _report(f"cuBLAS {dt} matmul [{T}x{H}] x [{H}x12288]", ms, 2 * T * H * 12288)


def c_moe_prefill(dev, T, dt):
    import torch
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    x = torch.randn(T, H, device=dev, dtype=dt)
    gu = _nvfp4(dev, 2 * I, H)
    dn = _nvfp4(dev, H, I)
    logits = torch.randn(T, E, device=dev)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float().contiguous()
    ids = ids.to(torch.int32).contiguous()
    fn = lambda: fused_experts_nvfp4(x, *gu, *dn, w, ids.clone(), E)  # noqa: E731
    ms = _time(fn)
    flops = 2 * T * TOPK * (2 * I * H + H * I)
    _report(f"NVFP4 MoE prefill (one layer, {dt}) T={T}", ms, flops)
    print(f"      x40 layers = {ms * 40 / 1000:.1f} s per prefill")


def c_fp8_gemm(dev, T, dt):
    import torch
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    for n, k in ((12288, H), (H, 4096)):
        w = torch.randn(n, k, device=dev, dtype=dt) * 0.02
        s = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
        q = (w.float() / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        x = torch.randn(T, k, device=dev, dtype=dt)
        ms = _time(lambda: fp8_pertensor_linear(x, q, s))
        _report(f"fp8 W8A16 GEMM ({dt}) T={T} N={n} K={k}", ms, 2 * T * n * k)
    print("      per layer: in_proj + out_proj, x30 GDN layers (attention layers similar)")


def c_attn_extend(dev, T, dt):
    import torch
    from freetoken.kernel.triton.attention import extend_paged_attention

    q = torch.randn(T, QH, HD, device=dev, dtype=dt)
    k = torch.randn(T, KVH, HD, device=dev, dtype=dt)
    v = torch.randn(T, KVH, HD, device=dev, dtype=dt)
    k_cache = torch.zeros(16, KVH, HD, device=dev, dtype=dt)
    v_cache = torch.zeros(16, KVH, HD, device=dev, dtype=dt)
    qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, 0], dtype=torch.int32, device=dev)
    kv_indices = torch.zeros(1, dtype=torch.int32, device=dev)
    prefix_lens = torch.zeros(1, dtype=torch.int32, device=dev)
    fn = lambda: extend_paged_attention(  # noqa: E731
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, prefix_lens, T, HD ** -0.5,
        k_extend=k, v_extend=v,
    )
    ms = _time(fn)
    flops = 2 * 2 * QH * T * T * HD / 2  # QK^T and PV, causal half
    _report(f"Triton extend attention ({dt}) T={T} hd={HD}", ms, flops)
    print(f"      x10 full-attention layers = {ms * 10 / 1000:.1f} s per prefill")


def c_gdn_prefill(dev, T, dt):
    import torch
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    q = torch.randn(1, T, KH, D, device=dev, dtype=dt)
    k = torch.randn(1, T, KH, D, device=dev, dtype=dt)
    v = torch.randn(1, T, VH, D, device=dev, dtype=dt)
    g = -torch.rand(1, T, VH, device=dev) * 0.5
    beta = torch.rand(1, T, VH, device=dev)
    state = torch.zeros(2, VH, D, D, device=dev)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
    fn = lambda: gdn_prefill_chunk_fla(  # noqa: E731
        q, k, v, g, beta, state_source=state, indices=idx, cu_seqlens=cu, scale=D ** -0.5)
    ms = _time(fn)
    flops = 2 * 4 * T * VH * D * D  # rough: four [T,D]x[D,D]-class products per head
    _report(f"GDN chunk prefill ({dt}) T={T}", ms, flops)
    print(f"      x30 GDN layers = {ms * 30 / 1000:.1f} s per prefill")


def c_nvfp4_dense(dev, T, dt):
    import torch
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t, nvfp4_transpose_resident

    packed = torch.randint(0, 256, (2 * I, H // 2), device=dev, dtype=torch.uint8)
    scale = (torch.rand(2 * I, H // 16, device=dev) * 0.75 + 0.25).to(torch.float8_e4m3fn)
    g = torch.full((2 * I,), 0.01, device=dev, dtype=torch.float16)
    wt, st = nvfp4_transpose_resident(packed, scale)
    x = torch.randn(T, H, device=dev, dtype=dt)
    ms = _time(lambda: nvfp4_dense_linear_t(x, wt, st, g))
    _report(f"NVFP4 dense (shared expert gate_up, {dt}) T={T}", ms, 2 * T * 2 * I * H)


def c_moe_check(dev, T, dt):
    """Correctness: the prefill MoE kernel with LUT dequant and with arithmetic dequant, both
    against a torch reference built on dequant_nvfp4 (E=32, T=256 to keep the reference cheap)."""
    import torch
    import torch.nn.functional as F
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe import fused_nvfp4 as fm

    Es, Ts = 32, 256
    x = torch.randn(Ts, H, device=dev, dtype=dt)

    def mk(n, k):
        packed = torch.randint(0, 256, (Es, n, k // 2), device=dev, dtype=torch.uint8)
        scale = (torch.rand(Es, n, k // 16, device=dev) * 0.75 + 0.25).to(torch.float8_e4m3fn)
        glob = torch.full((Es, n), 0.01, device=dev, dtype=torch.float16)
        return packed, scale, glob

    gu, dn = mk(2 * I, H), mk(H, I)
    logits = torch.randn(Ts, Es, device=dev)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float().contiguous()
    ids = ids.to(torch.int32).contiguous()
    outs = {}
    for mode in ("0", "1", "scratch"):
        os.environ["FREETOKEN_NVFP4_MOE_ARITH"] = "1" if mode == "1" else "0"
        os.environ["FREETOKEN_NVFP4_MOE_SCRATCH"] = "1" if mode == "scratch" else "0"
        fm._arith_dequant.cache_clear()
        if hasattr(fm, "_scratch_moe_preferred"):
            fm._scratch_moe_preferred.cache_clear()
        outs[mode] = fm.fused_experts_nvfp4(x, *gu, *dn, w, ids.clone(), Es).float()
        torch.cuda.synchronize()
    slots = torch.arange(Es, dtype=torch.int32, device=dev)
    Wgu = dequant_nvfp4(*gu, slots).float()   # [Es, 2I, H]
    Wd = dequant_nvfp4(*dn, slots).float()    # [Es, H, I]
    ref = torch.zeros(Ts, H, device=dev)
    for e in range(Es):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        h = x[t_idx].float() @ Wgu[e].t()
        act = F.silu(h[:, :I]) * h[:, I:]
        ref.index_add_(0, t_idx, (act @ Wd[e].t()) * w[t_idx, k_idx][:, None])
    scale_ref = ref.abs().max().item() + 1e-6
    for mode, label in (("0", "LUT dequant"), ("1", "arithmetic dequant"), ("scratch", "dequant + cuBLAS")):
        rel = (outs[mode] - ref).abs().max().item() / scale_ref
        print(f"{'OK  ' if rel < 5e-2 else 'FAIL'} MoE prefill {label} vs torch reference: rel={rel:.3g}")
    print(f"      LUT vs arithmetic max|diff| = {(outs['0'] - outs['1']).abs().max().item():.3g} (expected ~0)")


def c_moe_sweep(dev, T, dt):
    """Time the prefill MoE kernel under a set of tile configurations (the H100-tuned default
    first). BLOCK_SIZE_M feeds moe_align_block_size on the host, so every config is consistent."""
    import torch
    from freetoken.moe import fused_nvfp4 as fm

    x = torch.randn(T, H, device=dev, dtype=dt)
    gu = _nvfp4(dev, 2 * I, H)
    dn = _nvfp4(dev, H, I)
    logits = torch.randn(T, E, device=dev)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float().contiguous()
    ids = ids.to(torch.int32).contiguous()
    flops = 2 * T * TOPK * (2 * I * H + H * I)
    configs = [
        (32, 64, 32, 8, 4),   # upstream default (H100 sweep)
        (32, 64, 32, 4, 2),
        (32, 64, 32, 4, 1),
        (64, 64, 32, 4, 2),
        (64, 64, 64, 4, 2),
        (64, 128, 32, 4, 2),
        (64, 128, 64, 8, 2),
        (128, 64, 32, 4, 2),
        (128, 128, 32, 8, 2),
        (16, 64, 32, 4, 2),
        (32, 128, 64, 4, 2),
    ]
    orig = fm._prefill_config
    try:
        for bm, bn, bkb, warps, stages in configs:
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_KB=bkb, GROUP_SIZE_M=8,
                       num_warps=warps, num_stages=stages)
            fm._prefill_config = lambda M, _c=cfg: _c
            try:
                ms = _time(lambda: fm.fused_experts_nvfp4(x, *gu, *dn, w, ids.clone(), E), warmup=1, iters=2)
                print(f"  M={bm:3d} N={bn:3d} KB={bkb:2d} warps={warps} stages={stages}: {ms:8.1f} ms  {flops / ms / 1e9:5.2f} TFLOPS")
            except Exception as exc:  # noqa: BLE001
                print(f"  M={bm:3d} N={bn:3d} KB={bkb:2d} warps={warps} stages={stages}: FAILED {type(exc).__name__}: {str(exc)[:80]}")
    finally:
        fm._prefill_config = orig


def c_moe_align(dev, T, dt):
    import torch
    from freetoken.moe.fused import moe_align_block_size

    ids = torch.randint(0, E, (T, TOPK), device=dev, dtype=torch.int32)
    ms = _time(lambda: moe_align_block_size(ids, 32, E))
    print(f"moe_align_block_size T={T}: {ms:.2f} ms")


def _run(case, T):
    import torch

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    dt = torch.float16 if "fp16" in case else torch.bfloat16
    fns = {
        "cublas": c_cublas, "moe_check": c_moe_check, "moe_sweep": c_moe_sweep,
        "moe_prefill": c_moe_prefill, "fp8_gemm": c_fp8_gemm,
        "attn_extend": c_attn_extend, "gdn_prefill": c_gdn_prefill,
        "nvfp4_dense": c_nvfp4_dense, "moe_align": c_moe_align,
    }
    base = next(k for k in fns if case.startswith(k))
    fns[base](dev, T, dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2785)
    ap.add_argument("--run")
    args = ap.parse_args()
    if args.run:
        try:
            _run(args.run, args.tokens)
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(2)
        return
    import torch

    print(f"device {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} T={args.tokens}")
    noise = ("_POSIX_C_SOURCE", "__triton_launcher", "features.h", "libc-header", "pyconfig.h",
             "cuda.h:56", "previous definition", "| #define", "|          ^", "|         ^", "In file included")
    for c in CASES:
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="0")
        if "_lut_" in c:
            env["FREETOKEN_NVFP4_MOE_ARITH"] = "0"
            env["FREETOKEN_NVFP4_MOE_SCRATCH"] = "0"
        elif "_arith_" in c:
            env["FREETOKEN_NVFP4_MOE_ARITH"] = "1"
            env["FREETOKEN_NVFP4_MOE_SCRATCH"] = "0"
        elif "_scratch_" in c:
            env["FREETOKEN_NVFP4_MOE_SCRATCH"] = "1"
        p = subprocess.run([sys.executable, __file__, "--run", c, "--tokens", str(args.tokens)],
                           capture_output=True, text=True, timeout=1800, env=env)
        lines = [l for l in (p.stdout + p.stderr).strip().splitlines() if not any(n in l for n in noise)]
        print("\n".join(lines[-6:]) if lines else f"{c}: (no output, rc={p.returncode})")


if __name__ == "__main__":
    main()
