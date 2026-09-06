"""Per-kernel probe for FreeToken on Turing (sm_75): runs each GPU kernel the qwen3_5_moe
decode path uses in its own subprocess (a crashed CUDA context cannot be reused), compares
it with a plain-torch reference and prints one line per kernel.

    ~/FreeToken-src/.venv-dev/bin/python turing_kernel_probe.py          # all kernels
    ~/FreeToken-src/.venv-dev/bin/python turing_kernel_probe.py --run fp8_gemv   # one kernel

Shapes follow Ornith-1.5-35B-A3B / Qwen3.6-35B-A3B (hidden 2048, GDN 16 key / 32 value heads
of 128, shared expert 512, vocab 248320). Every subprocess runs with CUDA_LAUNCH_BLOCKING=1 so a
crashing torch/flashinfer kernel is reported at its own launch (Triton launches are reported
at the next synchronizing call regardless). Running the same file on an Ampere card validates
the references: everything must be OK there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback

TESTS = [
    "rmsnorm", "fused_add_rmsnorm", "fp8_gemv", "fp8_gemm", "nvfp4_dequant",
    "nvfp4_gemv", "nvfp4_gemv_t", "nvfp4_gemv_lmhead", "nvfp4_gemm8", "nvfp4_gemm100",
    "conv_decode", "conv_varlen",
    "gdn_decode", "gdn_decode_nol2", "gdn_decode_tiny", "gdn_recurrent_t1", "gdn_recurrent_t8",
    "gdn_prefill", "gdn_prefill_t64",
    "rms_norm_gated", "silu_and_mul", "moe_align",
    "router_topk", "fp8_gemv_small", "shared_expert_chain",
]

H, KH, VH, D = 2048, 16, 32, 128           # hidden, GDN key heads, value heads, head dim
KEY_DIM, VAL_DIM = KH * D, VH * D
CONV_DIM, CONV_K = 2 * KEY_DIM + VAL_DIM, 4
VOCAB = 248320


def _err(out, ref):
    out = out.float()
    ref = ref.float()
    d = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    return d, d / scale


def _report(name, out, ref, tol=3e-2):
    d, rel = _err(out, ref)
    ok = rel < tol and out.isfinite().all().item()
    print(f"{'OK  ' if ok else 'FAIL'} {name}: max|diff|={d:.4g} rel={rel:.3g} shape={tuple(out.shape)}")
    return ok


def _silu(x):
    return x * x.sigmoid()


# ----------------------------------------------------------------------------------- norms
def t_rmsnorm(dev):
    import torch
    from freetoken.kernel.triton.norm import rmsnorm

    x = torch.randn(4, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(H, device=dev, dtype=torch.bfloat16)
    out = rmsnorm(x, w, 1e-6)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    torch.cuda.synchronize()
    return _report("rmsnorm (triton, sgl_kernel fallback)", out, ref)


def t_fused_add_rmsnorm(dev):
    import torch
    from freetoken.kernel.triton.norm import fused_add_rmsnorm

    x = torch.randn(4, H, device=dev, dtype=torch.bfloat16)
    r = torch.randn(4, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(H, device=dev, dtype=torch.bfloat16)
    s = (x.float() + r.float())
    ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    fused_add_rmsnorm(x, r, w, 1e-6)  # in place: x <- norm(x + r), r <- x + r
    torch.cuda.synchronize()
    return _report("fused_add_rmsnorm out", x, ref) & _report("fused_add_rmsnorm residual", r, s)


# ----------------------------------------------------------------------------------- fp8 W8A16
def _fp8_case(dev, M, N, K):
    import torch
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    s = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)          # per-row scale
    q = (w.float() / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    out = fp8_pertensor_linear(x, q, s)
    ref = x.float() @ (q.float() * s[:, None]).t()
    torch.cuda.synchronize()
    return _report(f"fp8 W8A16 linear M={M} N={N} K={K}", out, ref)


def t_fp8_gemv(dev):
    return _fp8_case(dev, 1, CONV_DIM + VAL_DIM, H) & _fp8_case(dev, 1, H, VAL_DIM)


def t_fp8_gemm(dev):
    return _fp8_case(dev, 8, CONV_DIM + VAL_DIM, H) & _fp8_case(dev, 100, H, VAL_DIM)


# ----------------------------------------------------------------------------------- NVFP4
def _nvfp4_weight(dev, N, K):
    import torch

    packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8)
    scale = (torch.rand(N, K // 16, device=dev) * 0.75 + 0.25).to(torch.float8_e4m3fn)
    gscale = torch.full((N,), 0.01, device=dev, dtype=torch.float16)
    return packed, scale, gscale


def t_nvfp4_dequant(dev):
    import torch
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    N, K = 512, H
    packed, scale, gscale = _nvfp4_weight(dev, N, K)
    slots = torch.zeros(1, dtype=torch.int32, device=dev)
    w = dequant_nvfp4(packed.unsqueeze(0), scale.unsqueeze(0), gscale.unsqueeze(0), slots)[0]
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    lo = lut[(packed & 15).long()]
    hi = lut[(packed >> 4).long()]
    codes = torch.stack([lo, hi], -1).reshape(N, K)
    ref = codes * scale.float().repeat_interleave(16, dim=1) * gscale.float()[:, None]
    torch.cuda.synchronize()
    return _report("nvfp4 dequant (cache -> bf16)", w, ref)


def _nvfp4_case(dev, M, N, K, transposed):
    import torch
    from freetoken.kernel.triton.nvfp4_linear import (
        _ref, nvfp4_dense_linear, nvfp4_dense_linear_t, nvfp4_transpose_resident,
    )

    packed, scale, gscale = _nvfp4_weight(dev, N, K)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = _ref(x, packed, scale, gscale, torch.bfloat16)
    if transposed:
        wt, st = nvfp4_transpose_resident(packed, scale)
        out = nvfp4_dense_linear_t(x, wt, st, gscale)
    else:
        out = nvfp4_dense_linear(x, packed, scale, gscale)
    torch.cuda.synchronize()
    return _report(f"nvfp4 W4A16 linear{' (K-major)' if transposed else ''} M={M} N={N} K={K}", out, ref)


def t_nvfp4_gemv(dev):
    return _nvfp4_case(dev, 1, 1024, H, False) & _nvfp4_case(dev, 1, H, 512, False)


def t_nvfp4_gemv_t(dev):
    return _nvfp4_case(dev, 1, 1024, H, True) & _nvfp4_case(dev, 1, H, 512, True)


def t_nvfp4_gemv_lmhead(dev):
    return _nvfp4_case(dev, 1, VOCAB, H, True)


def t_nvfp4_gemm8(dev):
    return _nvfp4_case(dev, 8, 1024, H, True) & _nvfp4_case(dev, 8, 8192, H, True)


def t_nvfp4_gemm100(dev):
    return _nvfp4_case(dev, 100, 1024, H, True) & _nvfp4_case(dev, 100, H, 512, False)


# ----------------------------------------------------------------------------------- conv
def t_conv_decode(dev):
    import torch
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode

    B, slots = 1, 2
    x = torch.randn(B, CONV_DIM, device=dev, dtype=torch.bfloat16)
    state = torch.randn(slots, CONV_DIM, CONV_K - 1, device=dev, dtype=torch.bfloat16)
    w = torch.randn(CONV_DIM, CONV_K, device=dev, dtype=torch.bfloat16) * 0.3
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    window = torch.cat([state[1].float(), x[0].float()[:, None]], dim=1)  # [conv_dim, K]
    ref = _silu((window * w.float()).sum(-1))[None]
    ref_state = window[:, 1:]
    out = causal_conv1d_decode(x, state, w, idx)
    torch.cuda.synchronize()
    return _report("causal conv decode out", out, ref) & _report("causal conv decode state", state[1], ref_state)


def t_conv_varlen(dev):
    import torch
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen

    T, slots = 37, 2
    x = torch.randn(CONV_DIM, T, device=dev, dtype=torch.bfloat16)
    state = torch.zeros(slots, CONV_DIM, CONV_K - 1, device=dev, dtype=torch.bfloat16)
    w = torch.randn(CONV_DIM, CONV_K, device=dev, dtype=torch.bfloat16) * 0.3
    xf = x.float()
    padded = torch.nn.functional.pad(xf, (CONV_K - 1, 0))
    ref = _silu(sum(padded[:, j:j + T] * w.float()[:, j:j + 1] for j in range(CONV_K)))
    ref_state = xf[:, -(CONV_K - 1):]
    out = causal_conv1d_varlen(
        x, w, state, torch.tensor([0, T], dtype=torch.int32, device=dev),
        torch.tensor([0], dtype=torch.int32, device=dev), torch.tensor([False], device=dev),
    )
    torch.cuda.synchronize()
    return _report("causal conv varlen out", out, ref) & _report("causal conv varlen state", state[0], ref_state)


# ----------------------------------------------------------------------------------- GDN
def _gdn_ref(q, k, v, g, beta, state):
    """q/k [T, KH, D] -> expanded to VH (kernel: i_h = i_hv // (HV // H)); returns out
    [T, VH, D], new state [VH, D, D]."""
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    rep = v.shape[1] // q.shape[1]
    qe = q.repeat_interleave(rep, dim=1)[None]
    ke = k.repeat_interleave(rep, dim=1)[None]
    out, st = recurrent_gated_delta_rule(qe, ke, v[None], g[None], beta[None], initial_state=state[None])
    return out[0], st[0]


def _l2n(x):
    import torch

    return (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _gdn_decode_case(dev, kh, vh, d, in_kernel_l2norm=True):
    """The model's decode kernel (fused_sigmoid_gating_delta_rule_update) at head counts
    kh/vh and head dim d; the model calls it with l2norm in-kernel."""
    import torch
    import torch.nn.functional as F
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    B, slots = 1, 2
    q = torch.randn(1, B, kh, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, kh, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, vh, d, device=dev, dtype=torch.bfloat16)
    a = torch.randn(B, vh, device=dev, dtype=torch.bfloat16)
    b = torch.randn(B, vh, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(vh, device=dev).abs().log()
    dt_bias = torch.randn(vh, device=dev) * 0.1
    state = torch.randn(slots, vh, d, d, device=dev) * 0.1
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()
    # the kernels store the state as [HV, V, K] (V-major: o_v * K + o_k); the reference uses [K, V]
    ref_out, ref_state = _gdn_ref(q[0], k[0], v[0], g, beta, state[1].transpose(-1, -2).clone())
    qk, kk = (q, k) if in_kernel_l2norm else (_l2n(q), _l2n(k))
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
        q=qk, k=kk, v=v, b=b, initial_state_source=state, initial_state_indices=idx,
        scale=d ** -0.5, use_qk_l2norm_in_kernel=in_kernel_l2norm,
        cu_seqlens=torch.arange(B + 1, dtype=torch.int32, device=dev),
    )
    out = o[0]
    torch.cuda.synchronize()
    tag = f"gdn decode kh={kh} vh={vh} d={d} l2norm_in_kernel={in_kernel_l2norm}"
    return _report(tag + " out", out, ref_out, tol=5e-2) & \
        _report(tag + " state", state[1].transpose(-1, -2), ref_state, tol=5e-2)


def t_gdn_decode(dev):
    return _gdn_decode_case(dev, KH, VH, D)


def t_gdn_decode_nol2(dev):
    return _gdn_decode_case(dev, KH, VH, D, in_kernel_l2norm=False)


def t_gdn_decode_tiny(dev):
    ok = _gdn_decode_case(dev, 1, 1, 16)
    ok &= _gdn_decode_case(dev, 1, 2, 32)
    ok &= _gdn_decode_case(dev, 2, 2, 128)
    return ok


def _gdn_recurrent_case(dev, T, kh=KH, vh=VH, d=D):
    """fused_recurrent_gated_delta_rule (the token-loop kernel): same math, separate kernel."""
    import torch
    from freetoken.kernel.fla.fused_recurrent import fused_recurrent_gated_delta_rule

    q = torch.randn(1, T, kh, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, kh, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, vh, d, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, vh, device=dev) * 0.5
    beta = torch.rand(1, T, vh, device=dev)
    state0 = torch.randn(vh, d, d, device=dev) * 0.1
    ref_out, ref_state = _gdn_ref(q[0], k[0], v[0], g[0], beta[0], state0.clone())
    o, st = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=d ** -0.5, initial_state=state0[None].transpose(-1, -2).contiguous(),
        inplace_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    st = st.reshape(-1, vh, d, d)[-1]
    ok = _report(f"gdn fused_recurrent T={T} out", o.reshape(T, vh, d), ref_out, tol=5e-2)
    d1, r1 = _err(st, ref_state)
    d2, r2 = _err(st.transpose(-1, -2), ref_state)
    print(f"      final state vs ref: rel={r1:.3g} (as [K,V])  rel={r2:.3g} (as [V,K])")
    return ok & (min(r1, r2) < 5e-2)


def t_gdn_recurrent_t1(dev):
    return _gdn_recurrent_case(dev, 1)


def t_gdn_recurrent_t8(dev):
    return _gdn_recurrent_case(dev, 8)


def _gdn_prefill_case(dev, T):
    import torch
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    slots = 2
    q = torch.randn(1, T, KH, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, KH, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, VH, D, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, VH, device=dev) * 0.5
    beta = torch.rand(1, T, VH, device=dev)
    state = torch.zeros(slots, VH, D, D, device=dev)
    ref_out, ref_state = _gdn_ref(q[0], k[0], v[0], g[0], beta[0], torch.zeros(VH, D, D, device=dev))
    out = gdn_prefill_chunk_fla(
        q, k, v, g, beta, state_source=state, indices=torch.tensor([1], dtype=torch.int32, device=dev),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int64, device=dev), scale=D ** -0.5,
    )
    torch.cuda.synchronize()
    return _report(f"gdn prefill T={T} out (chunk_gated_delta_rule)", out, ref_out, tol=5e-2) & \
        _report(f"gdn prefill T={T} state", state[1].transpose(-1, -2), ref_state, tol=5e-2)


def t_gdn_prefill(dev):
    return _gdn_prefill_case(dev, 100)


def t_gdn_prefill_t64(dev):
    return _gdn_prefill_case(dev, 64) & _gdn_prefill_case(dev, 8)


# ----------------------------------------------------------------------------------- misc
def t_rms_norm_gated(dev):
    import torch
    from freetoken.kernel.fla import rms_norm_gated

    x = torch.randn(VH, D, device=dev, dtype=torch.bfloat16)
    z = torch.randn(VH, D, device=dev, dtype=torch.bfloat16)
    w = torch.randn(D, device=dev, dtype=torch.bfloat16)
    out = rms_norm_gated(x=x, weight=w, bias=None, z=z, eps=1e-6, is_rms_norm=True,
                         norm_before_gate=True, activation="silu")
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float() * _silu(z.float())
    torch.cuda.synchronize()
    return _report("rms_norm_gated (fla)", out, ref)


def t_silu_and_mul(dev):
    import torch
    from freetoken.layers.activation import silu_and_mul

    x = torch.randn(4, 2 * 512, device=dev, dtype=torch.bfloat16)
    out = silu_and_mul(x)
    ref = _silu(x[:, :512].float()) * x[:, 512:].float()
    torch.cuda.synchronize()
    return _report("silu_and_mul (flashinfer JIT)", out, ref)


def t_moe_align(dev):
    import torch
    from freetoken.moe.fused import moe_align_block_size

    topk = torch.randint(0, 256, (1, 8), device=dev, dtype=torch.int32)
    sorted_ids, expert_ids, n = moe_align_block_size(topk, 16, 256)
    torch.cuda.synchronize()
    n = int(n.item())
    eids = expert_ids[: max(n // 16, 1)]
    ok = 0 <= int(eids.min()) and int(eids.max()) < 256 and n % 16 == 0
    print(f"{'OK  ' if ok else 'FAIL'} moe_align_block_size (triton): post_pad={n} expert_ids[:8]={expert_ids[:8].tolist()}")
    return ok


def _router_case(dev, M, E, K):
    import torch
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.moe.fused import _torch_fused_topk

    logits = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    w, ids = fused_topk_softmax(logits, K, True, None)
    torch.cuda.synchronize()
    rw, rids = _torch_fused_topk(logits, K, True, None)
    same_ids = torch.equal(ids.sort(dim=-1).values, rids.sort(dim=-1).values)
    d, rel = _err(w.sort(dim=-1).values, rw.sort(dim=-1).values)
    ok = same_ids and rel < 1e-2
    print(f"{'OK  ' if ok else 'FAIL'} router top-k softmax M={M} E={E} K={K}: ids_match={same_ids} "
          f"weights rel={rel:.3g} ids={ids[0].tolist()}")
    return ok


def t_router_topk(dev):
    ok = _router_case(dev, 1, 256, 8)     # Ornith / Qwen3.6-35B-A3B
    ok &= _router_case(dev, 4, 256, 8)
    ok &= _router_case(dev, 1, 32, 4)     # gpt-oss-20b (known to work on this card)
    return ok


def t_fp8_gemv_small(dev):
    ok = _fp8_case(dev, 1, 2 * VH, H)     # in_proj_ba: N = 64
    ok &= _fp8_case(dev, 1, 16, H)
    ok &= _fp8_case(dev, 1, 9216, H)      # full-attention qkv (q gate doubles q)
    ok &= _fp8_case(dev, 1, H, 4096)      # o_proj
    return ok


def t_shared_expert_chain(dev):
    """The exact shared-expert chain of the decode step through the layer classes:
    Nvfp4DenseColMerged(gate|up) -> silu_and_mul -> Nvfp4DenseLinear(down), at bs=1."""
    import torch
    from freetoken.kernel.triton.nvfp4_linear import _ref, nvfp4_dense_linear_t, nvfp4_transpose_resident
    from freetoken.layers.activation import silu_and_mul

    inter = 512
    p1, s1, g1 = _nvfp4_weight(dev, 2 * inter, H)
    p2, s2, g2 = _nvfp4_weight(dev, H, inter)
    x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    wt1, st1 = nvfp4_transpose_resident(p1, s1)
    wt2, st2 = nvfp4_transpose_resident(p2, s2)
    h = nvfp4_dense_linear_t(x, wt1, st1, g1)
    a = silu_and_mul(h)
    y = nvfp4_dense_linear_t(a, wt2, st2, g2)
    torch.cuda.synchronize()
    ref_h = _ref(x, p1, s1, g1, torch.bfloat16)
    ref_a = _silu(ref_h[:, :inter].float()) * ref_h[:, inter:].float()
    ref_y = _ref(ref_a.to(torch.bfloat16), p2, s2, g2, torch.bfloat16)
    return _report("shared expert chain gate_up -> silu_and_mul -> down (M=1)", y, ref_y, tol=5e-2)


# ----------------------------------------------------------------------------------- driver
_NOISE = ("_POSIX_C_SOURCE", "__triton_launcher", "features.h", "libc-header", "pyconfig.h",
          "cuda.h:56", "previous definition", "| #define", "|          ^", "|         ^", "In file included")


def _run_one(name):
    import torch

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    try:
        from freetoken.kernel.backend import is_sgl_kernel_installed
        from freetoken.kernel.triton.e4m3_compat import e4m3_native
        print(f"  [{name}] sgl_kernel gated off: {not is_sgl_kernel_installed()}  e4m3 native: {e4m3_native()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] (env probe failed: {exc})")
    ok = globals()[f"t_{name}"](dev)
    torch.cuda.synchronize()
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        try:
            _run_one(sys.argv[2])
        except SystemExit:
            raise
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(2)
    import torch

    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"device {name} cc={cc} torch {torch.__version__}")
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    results = {}
    for t in TESTS:
        print(f"\n=== {t}")
        try:
            p = subprocess.run([sys.executable, __file__, "--run", t], env=env, timeout=900,
                               capture_output=True, text=True)
            lines = [l for l in (p.stdout + p.stderr).strip().splitlines()
                     if not any(n in l for n in _NOISE)]
            print("\n".join(lines[-25:]))
            results[t] = {0: "OK", 1: "NUMERIC-FAIL", 2: "EXCEPTION"}.get(p.returncode, f"CRASH(rc={p.returncode})")
        except subprocess.TimeoutExpired:
            results[t] = "TIMEOUT"
    print("\n=== summary")
    for t in TESTS:
        print(f"{results[t]:14s} {t}")


if __name__ == "__main__":
    main()
