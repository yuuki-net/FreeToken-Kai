# Running on Turing (sm_75)

Upstream FreeToken states Ampere (RTX 30 series) or newer as the requirement. This fork makes the
engine run on Turing with six isolated changes: three that make it start (1-3) and three that
make prefill usable (4-6). Each fixes a distinct failure that was found on an
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

**Four things this was not.** The symptom lands inside a Triton kernel, so every one of these is a
natural next guess, and each one cost time to rule out. They are all wrong:

- *Triton's `tl.dot` does not support bf16 on Turing.* It does. fp16 and bf16 `tl.dot` were both run
  directly on sm_75 under Triton 3.6 and give correct results.
- *Everything has to be forced to fp16 on Turing.* Not for correctness. fp16 is worth choosing on
  Turing for speed (there are no bf16 tensor cores — see section 4), but it is not what fixes this.
- *Triton 3.6 does not support Turing.* It does.
- *This PyTorch build has no sm_75.* Check before believing it: `torch.cuda.get_arch_list()` lists
  `sm_75`.

The cause is in a different package, and it fails silently: a launch whose error nobody checks, and
a `torch.empty` output passed downstream still uninitialised.

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

## 4. fp8 W8A16 prefill GEMM: dequant + cuBLAS below Ampere

`python/freetoken/kernel/triton/fp8_pertensor_linear.py`, `_gemm_scratch()` / `_scratch_gemm_preferred()`.

**Symptom.** Ornith prefills a 2785-token prompt in 68 s even in fp16. `tools/turing_prefill_bench.py`
timed each prefill kernel at that shape: the inline-dequant fp8 GEMM used for the GDN and attention
projections ran at **0.15 TFLOPS** (fp16 and bf16 alike) while cuBLAS fp16 did 19 TFLOPS on the same
card; the 30 GDN layers' projections alone were ~37 s.

**Fix.** Below Ampere, M>1 W8A16 GEMMs expand the fp8 weight into a per-call scratch in the activation
dtype (row scale folded in, ~50 MB for a 12288 x 2048 projection) and run cuBLAS. Measured 16.4
TFLOPS afterwards. The M=1 GEMV (decode) is untouched. `FREETOKEN_FP8_SCRATCH_GEMM=0/1` overrides.

## 5. Arithmetic e2m1 dequant in the prefill MoE kernel

`python/freetoken/kernel/triton/nvfp4_fused_moe.py`, constexpr `ARITH_DEQUANT`.

The prefill MoE kernel looked up each 4-bit code in a 16-entry LUT (two gathers per byte). The dense
NVFP4 kernels already use a gather-free bit-placement dequant, so the same was wired into the MoE
kernel. It is bit-identical (max diff 0 against the LUT path) but, on the 2060, **not faster**: the
kernel stays at 0.56 TFLOPS. Kept as a knob (`FREETOKEN_NVFP4_MOE_ARITH`) since it costs nothing and
may matter on other GPUs; the real fix is 6.

## 6. NVFP4 prefill MoE: chunked dequant + per-expert cuBLAS below Ampere

`python/freetoken/moe/fused_nvfp4.py`, `_fused_experts_nvfp4_scratch()` / `_scratch_moe_preferred()`.

**Symptom.** The inline-dequant prefill MoE GEMM runs one layer of a 2785-token prompt in 248 ms
(0.56 TFLOPS), 10 s over 40 layers. A sweep of 11 tile configurations changed nothing (best 0.58
TFLOPS; larger tiles collapsed to 0.05-0.13). Upstream itself notes that in-kernel-dequant GEMMs are
~3x slower than cuBLAS even on H100; on Turing the gap is ~40x.

**Fix.** Below Ampere (and only for the silu activation), the layer's experts are dequantised 32 at a
time into an fp16/bf16 scratch (~200 MB) and each expert runs one cuBLAS GEMM over the tokens routed
to it; router weights and the sum over a token's routes are applied in fp32. One host sync per layer
(route counts). Measured 48-53 ms per layer in fp16 (2.9 TFLOPS; the remaining gap to cuBLAS is the
512 small launches per layer, not the GEMMs), 106 ms in bf16. `FREETOKEN_NVFP4_MOE_SCRATCH=0/1` overrides.

**Result.** The 2785-token prompt went from 68 s to ~8.5 s. About 5 s of that is now the per-chunk
expert streaming (the CPU-side layers' banks are pageable under WSL and are copied synchronously each
chunk); the compute part is ~1.3 ms per token.

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

- Decode: 25-37 tok/s for Ornith-1.5-35B-A3B in `hybrid` mode with `--dtype float16` (3B active
  parameters, NVFP4 experts; the CPU path reads about 0.5 GB per token at 50 GB/s). 13-14 tok/s
  for gpt-oss-20b.
- Prefill (fp16, after changes 4-6): 2785 tokens in ~8.5 s, of which ~5 s is the per-chunk expert
  streaming and ~1.3 ms/token is compute. Use `--dtype float16`: Turing has fp16 tensor cores but
  no bf16 ones (cuBLAS 19 vs 3 TFLOPS). Ornith's output is unaffected by fp16.
- Per-kernel numbers at 2785 tokens (`tools/turing_prefill_bench.py`): fp8 GEMM 16.4 TFLOPS (scratch),
  MoE prefill 2.9 TFLOPS (scratch), Triton extend attention 0.68 TFLOPS (0.9 s / prefill), GDN chunk
  0.12-0.19 TFLOPS (2-3 s / prefill), NVFP4 dense 16 TFLOPS.

## WSL2 notes

- `.wslconfig` `memory=` bounds what the engine can use; the expert banks of a 35B-A3B NVFP4 model
  are 17 GB and the CPU vision tower adds 1.7 GB. `memory=24GB` on a 32 GB machine works.
- `bank lock failed ... RLIMIT_MEMLOCK` is a warning about `mlock` of the CPU layers' banks. It
  does not affect startup. A normal user cannot raise the limit with `ulimit -l`; use
  `/etc/security/limits.conf`.
- With `networkingMode=mirrored`, bind with `--host 0.0.0.0` and, if other machines still cannot
  connect, add a Hyper-V firewall rule for the port (the Windows Defender rules do not cover WSL
  inbound traffic in mirrored mode).
