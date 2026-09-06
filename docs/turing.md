# Running on Turing (sm_75)

Upstream FreeToken states Ampere (RTX 30 series) or newer as the requirement. This fork makes the
engine run on Turing with three isolated changes. Each fixes a distinct failure that was found on an
RTX 2060 6 GB by narrowing the crash down kernel by kernel; the debugging notes are below so the
next person does not have to repeat it.

## 1. `sgl_kernel` is treated as absent below compute capability 8.0

`python/freetoken/kernel/backend.py`, `is_sgl_kernel_installed()`.

**Symptom.** With gpt-oss-20b, the first request dies in the MXFP4 grouped GEMM (a Triton kernel)
with `an illegal memory access was encountered`. `CUDA_LAUNCH_BLOCKING=1` moves the reported
location to an unrelated `.to()` call.

**Cause.** The `sgl_kernel` (sglang-kernel 0.4.5) wheel ships no sm_75 cubins. Every one of its
launches fails with `cudaErrorNoKernelImageForDevice`. `moe_align_block_size` does not check the
result and returns its `torch.empty` outputs (`sorted_ids`, `expert_ids`) uninitialised; the
grouped GEMM then indexes expert weights with garbage ids. The error surfaces far from its origin.

**Fix.** Report the package as not installed when the device's compute capability is below 8.0.
All four call sites (`kernel/causal_conv1d.py` x2, `layers/norm.py`, `moe/fused.py`) already have
Triton or native fallbacks.

## 2. The automatic attention backend skips flashinfer below Ampere

`python/freetoken/engine/engine.py` (`_resolve_auto_attention_backend`), `python/freetoken/utils/arch.py`
(`is_pre_ampere()`).

**Symptom.** With Ornith-1.5-35B-A3B (Qwen3.5-MoE architecture, head_dim 256), the eager warm-up
forward before CUDA graph capture fails in the first full-attention layer:
`BatchPrefillWithPagedKVCache failed with error unspecified launch failure`, inside the attention
kernel flashinfer JIT-compiled for sm_75. Layers 0-2 (GDN, MoE, shared expert, router) had run
fine. gpt-oss-20b has head_dim 64 and never hit this.

**Fix.** The `auto` resolver's FULL-attention candidates become
`("fi", not is_pre_ampere())` followed by `("triton", True)`, so Turing lands on the Triton
attention backend (paged KV, extend and decode kernels, CUDA-graph capable). An explicit
`--attention-backend fi` is still accepted.

## 3. The Triton extend (prefill) tiles respect the 64 KB shared-memory limit

`python/freetoken/kernel/triton/attention.py`, `_select_extend_tile()`.

**Symptom.** After change 2, decode graph capture succeeds but the prefill warm-up raises
`OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536`.

**Cause.** For head_dim 256 the tile selector only knew `(128, 64)` and `(64, 32)` and returned the
latter without checking it against the device limit. The extend kernel that also reads the cached
prefix stages K/V tiles for both sources, so its shared memory is
`(BLOCK_M + 4 * BLOCK_N) * BLOCK_D * 2` bytes = `(64 + 128) * 512` = 98,304; Turing's opt-in limit is
65,536.

**Fix.** After the existing preference logic, halve the tile (floor 16) until it fits the hard
limit, using `(M + 4N)` rows for the split kernel and `(M + 2N)` for the plain one. On Turing with
head_dim 256 that gives `(32, 16)` (48 KB) for the split kernel. Choices on sm_89 (99 KB) and
H100 (227 KB) are unchanged, which the unit tests pin down.

## What was checked and found fine

`tools/turing_kernel_probe.py` runs every GPU kernel of the Qwen3.5-MoE decode path in its own
subprocess and compares it with a plain-torch reference: RMSNorm, fp8 W8A16 linears, NVFP4 W4A16
linears (including the 248k-row lm_head) and dequant, causal conv, the GDN decode and prefill
kernels, gated RMSNorm, flashinfer's `silu_and_mul`, `moe_align_block_size`, and the router top-k.
All 22 probes pass on the RTX 2060, so the Triton and flashinfer element-wise kernels are not the
problem; only the two attention issues above were.

Two things that look like failures but are not:

- The GDN recurrent state is stored `[num_heads, V, K]` (V-major) by both fla kernels although the
  pool is documented as `[.., K, V]`. A probe that assumes `[K, V]` reports garbage for correct kernels
  (the probe's first version did).
- Under `--moe-backend hybrid` the CPU MoE executor spins on pinned cores while idle, so one core
  sits at 100 % between requests.

## Debugging notes

- Triton launches kernels through the driver API. Without `CUDA_LAUNCH_BLOCKING=1` an
  asynchronous fault is reported at the next synchronising call, which is often flashinfer's
  post-launch check or the load of the next Triton kernel, both unrelated to the faulting kernel.
  With `CUDA_LAUNCH_BLOCKING=1` torch and flashinfer kernels are reported at their own launch.
- `FREETOKEN_DEBUG_FP8_REF=1` and `FREETOKEN_DEBUG_DENSE_NVFP4_REF=1` swap the fp8 and NVFP4
  linears for torch references (upstream escape hatches). Useful to rule those kernels out.
- `py-spy dump --pid <scheduler pid> --native` shows what a "stuck" scheduler is doing. A main
  thread in `zmq poll` inside `overlap_loop` means the server is up and waiting for requests.

## Performance on the RTX 2060

- Decode: 20-37 tok/s for Ornith-1.5-35B-A3B in `hybrid` mode (3B active parameters, NVFP4
  experts; the CPU path reads about 0.5 GB per token at 50 GB/s). 13-14 tok/s for gpt-oss-20b.
- Prefill: slow. Turing has no bf16 tensor cores, so Triton lowers the bf16 matmuls of the MoE and
  attention prefill kernels to fp32 FMAs; a 5k-token prompt takes minutes. Turing does have fp16
  tensor cores, so `--dtype float16` should help substantially; this is not yet validated (fp16 has
  a narrower range than bf16 and some models overflow).

## WSL2 notes

- `.wslconfig` `memory=` bounds what the engine can use; the expert banks of a 35B-A3B NVFP4 model
  are 17 GB and the CPU vision tower adds 1.7 GB. `memory=24GB` on a 32 GB machine works.
- `bank lock failed ... RLIMIT_MEMLOCK` is a warning about `mlock` of the CPU layers' banks. It
  does not affect startup. A normal user cannot raise the limit with `ulimit -l`; use
  `/etc/security/limits.conf`.
- With `networkingMode=mirrored`, bind with `--host 0.0.0.0` and, if other machines still cannot
  connect, add a Hyper-V firewall rule for the port (the Windows Defender rules do not cover WSL
  inbound traffic in mirrored mode).
