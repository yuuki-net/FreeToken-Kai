# FreeToken Kai (改)

An unofficial fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken),
based on upstream `main` at commit `af71ba4` (2026-09-03). It is not affiliated with, endorsed
by, or supported by FlashML. The license is unchanged (Apache-2.0).

The fork adds two things upstream does not have:

1. **Image input over the OpenAI API** for checkpoints that ship a vision tower but were served
   text-only: Qwen3.8-Flash-Next and the Qwen3.5-MoE family (Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B).
   The vision tower runs on the CPU, so it costs no VRAM. Works from Open WebUI and any OpenAI
   client that sends `image_url` parts. See [image-input.md](image-input.md).
2. **Turing (RTX 20 series, sm_75) support.** Upstream requires Ampere or newer; six small,
   isolated changes make the engine run on an RTX 2060 at a usable speed. See [turing.md](turing.md).

Everything else is upstream FreeToken. The two feature sets are independent: the image input
patch also applies to a plain upstream checkout on Ampere+, and the Turing patch is useful on
its own for text-only models such as gpt-oss-20b.

## Who wrote this

The code in this fork was designed and written by **Claude Fable 5.1** (Anthropic's model, used
through Claude Code) at the request of the fork's maintainer, who provided the hardware, ran every
build and test on it, and made the decisions about scope. Commits carry a `Co-Authored-By` trailer
for Claude. Bug reports about this fork go to this repository, not to FlashML.

## Tested configurations

| Machine | Model | Result |
|---|---|---|
| RTX 2060 6 GB, 32 GB RAM, Windows 11 + WSL2 (`memory=24GB`) | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (35B MoE, 3B active, vision) | Text and image input work. Decode 25-39 tok/s (`--moe-backend hybrid`, `--dtype float16`), 64k of context with `--host-embedding`. Prefill: a 2062-token prompt in ~7.8 s (68 s before the Turing GEMM changes); ~5 s of that is the per-chunk expert streaming, the rest ~1.3 ms/token; a follow-up turn behind a cached prefix answers in 2-3 s |
| same | `openai/gpt-oss-20b` (MXFP4) | 13-14 tok/s with the Turing patch alone |
| RTX 3060 12 GB x2, 128 GB RAM, Linux | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (125B MoE, vision) | Image input validated end to end (colour probe 6/6, chunked prefill, M-RoPE). That machine runs a private pipeline-parallel build that is **not** part of this fork; the image code here is the same |

Nothing else has been tested. Other Turing cards (RTX 2070/2080, T4, GTX 16 series without
tensor cores) should behave like the 2060 but are unverified.

## Install

Source install, same as upstream, plus Pillow for image decoding. torchvision is deliberately not
required (the Qwen-VL processor is reimplemented on Pillow).

```bash
git clone https://github.com/<your-account>/freetoken-kai.git && cd freetoken-kai
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv pip install pillow
```

CUDA kernels are JIT-compiled on first use (CUDA 13 toolkit with `nvcc`, as upstream).

## Running Ornith-1.5-35B-A3B on an RTX 2060 (6 GB) under WSL2

```bash
FT_IMAGE_MAX_PIXELS=262144 ft serve \
  --model models/Ornith-1.5-35B-A3B-NVFP4 --dtype float16 --host 0.0.0.0 --port 1919 \
  --moe-backend hybrid --disable-moe-prefill-overlap --max-running-req 1 \
  --kv-reserve-tokens 16384 --max-seq-len-override 16384 --memory-ratio 0.85 \
  --moe-cpu-threads 6
```

| Flag / variable | Why |
|---|---|
| `--moe-backend hybrid` | Experts live in host RAM; misses are split between PCIe and the CPU (`ft bench bw` once to calibrate). NVFP4 experts decode at 50 GB/s on a 6-core CPU |
| `--disable-moe-prefill-overlap` | The prefill double buffer needs 2 x 256 expert slots, which a 6 GB card cannot spare |
| `--max-running-req 1` | GDN state slots 8 -> 2 and one CUDA graph; saves ~200 MB |
| `--kv-reserve-tokens 16384` | KV pages are carved from the same budget as the expert cache; the default 8192 was too small for Open WebUI prompts, 4096 far too small |
| `--memory-ratio 0.85` | The ratio is of total VRAM and the desktop's ~600 MB counts against it; 0.92 left 60 MB for graph capture |
| `--dtype float16` | Turing has fp16 tensor cores but no bf16 ones (cuBLAS: 19 vs 3 TFLOPS on a 2060); Ornith's output is unaffected |
| `FT_IMAGE_MAX_PIXELS=262144` | One image becomes at most 256 soft tokens (512 x 512); the default 1024 x 1024 is 1024 tokens |
| WSL `.wslconfig` `memory=24GB` | The expert banks are 17 GB, the CPU vision tower 1.7 GB in fp32; with `memory=16GB` loading swaps and appears to hang |

The attention backend resolves to `triton` automatically on Turing. Pipe the log through
`grep --line-buffered` if you filter it; a block-buffered `grep` hides the "Scheduler is idle"
line and makes a healthy server look stuck.

## Speculative decoding with the checkpoint's MTP head (`--spec-mtp K`)

Upstream FreeToken has no speculative decoding. This fork adds it for the Qwen3.5-MoE family
using the MTP (multi-token prediction) head the checkpoints ship under `mtp.*` (one
full-attention decoder layer plus `fc` / norms; Ornith-1.5-35B-A3B-NVFP4 carries it in bf16).

```bash
ft serve ... --max-running-req 1 --spec-mtp 3
```

How it works, per decode step:

1. The head drafted `K` tokens after the previous step. The target runs one extend forward
   over `[t_last, d_1, ..., d_K]` (K+1 rows) and samples every row as usual.
2. Draft `d_i` is accepted while it equals the target's sample at row `i-1`; the sample after
   the last accepted draft is the correction. So 1 to K+1 tokens are committed per step, and
   with greedy drafts the output distribution is the target's own (`temperature 0` gives the
   same text with and without `--spec-mtp`).
3. The GDN recurrent / conv states are rolled back to the last accepted row (the verify pass
   runs the per-token recurrent kernel and keeps every intermediate state); the KV rows of the
   rejected positions are simply overwritten by the next window.
4. The head runs over the window (its own KV slab, layer `num_layers`), then chains `K` greedy
   drafts one token at a time through the shared `lm_head`.

What it costs: one more full-attention KV layer (11 instead of 10 on Ornith), ~80 MB of bf16
head weights, and one more expert-bank layer -- the head's 256 bf16 experts (1.6 GB in the
checkpoint) are quantized to NVFP4 at load and appended to the offload cache (~0.43 GB of
pinned host memory). Loading reads 1.6 GB of bf16 experts through host RAM once.

Scope: single running request (`--max-running-req 1`); modelopt MIXED_PRECISION checkpoints
only (fp8 attention + NVFP4 experts, the Ornith / Qwen3.6-35B-A3B-NVFP4 format). The `Spec
decode` line in the decode status shows the mean accepted length.

CUDA graphs: the K+1-row verify window and the draft head (window pass + one chain step) are
captured at startup and replayed per step, the way the decode step is; windows shorter than
K+1 rows (the output budget's tail) and a capture failure fall back to the eager path with a
log line. An attention backend takes part by implementing `init_spec_capture` / `stage_spec`
(the Triton backend does, so `--attention-backend triton` gets the graphs on any GPU). The
window graph keeps the K+1 per-token GDN states of every layer resident (~250 MB at K=3 on a
30-layer GDN stack); when less than `FT_SPEC_GRAPH_MIN_FREE_MB` (256) of VRAM is left after
capture the graphs are dropped again, because a starved allocator makes the step slower than
eager.

Measured on the RTX 2060 (Ornith, fp16, `--moe-backend hybrid`, K=3): the head is good
(2.5 tokens accepted per step on average, up to 4), and the verify path matches plain decode
within fp16 rounding (`FT_SPEC_CHECK_STEP`: per-layer residual divergence 1e-4 at layer 0 to
1e-2 at layer 39, identical top-3 logits, GDN state after rollback within 1e-3). But it does
**not** make decoding faster there: a 4-row verify forward costs ~130-150 ms (graph and eager
alike) against ~45 ms for one row, because with the experts on the CPU every row pays its own
expert traffic (~27 ms per token), which is the dominant cost; there is little launch overhead
for a graph to remove. Result 8-22 tok/s versus 25-37 tok/s plain. Speculative decoding pays
off when a multi-row forward costs about as much as a single-row one -- experts resident on
the GPU -- which a 6 GB card cannot offer for a 35B MoE. Treat `--spec-mtp` on Turing/offload
setups as a correctness-verified feature, not a speed-up.

## 64k of context on 6 GB: `--host-embedding`

The input embedding table (250k x 2048 fp16 = 1 GB on Ornith) can live in pinned host
memory; the GPU gathers the rows it needs in place over PCIe (one row per decode step, 16 MB
per 4096-token prefill chunk), inside the CUDA graphs like any other kernel. The freed VRAM
goes to KV pages: on the RTX 2060 that is the difference between 12k and 64k of context.

```bash
ft serve ... --host-embedding --kv-reserve-tokens 65536 --max-seq-len-override 65536 --memory-ratio 0.82
```

Measured on the 2060 (Ornith, fp16, hybrid): 65,606 KV tokens allocated (1.25 GiB) with
0.44 GiB of VRAM still free after the graphs, decode 37-39 tok/s, host RAM +1 GB (pinned; it
counts against the pin quota, so two or three more bank layers decode on the CPU). Untied
vocabularies only (Ornith's is untied); Qwen3.5-MoE family.

## Environment variables added by this fork

| Variable | Default | Meaning |
|---|---|---|
| `FT_IMAGE_MAX_PIXELS` | `1048576` | Resolution cap the image processor keeps (soft tokens per image = pixels / 1024) |
| `FT_IMAGE_EMBED_CACHE` | `32` | LRU entries of vision-tower output per image (chat clients resend every image each turn); `0` disables |
| `FREETOKEN_FP8_SCRATCH_GEMM` | arch (on below Ampere) | fp8 W8A16 prefill GEMM as dequant + cuBLAS instead of the inline-dequant Triton kernel |
| `FREETOKEN_NVFP4_MOE_SCRATCH` | arch (on below Ampere) | NVFP4 prefill MoE as chunked dequant + per-expert cuBLAS instead of the inline-dequant kernel |
| `FREETOKEN_NVFP4_MOE_ARITH` | arch (on below Ampere) | Arithmetic (gather-free) e2m1 dequant in the prefill MoE kernel; bit-identical, speed knob only |
| `FREETOKEN_CPU_PREFILL_MAX_TOKENS` | `256` | Prefill extends up to this many rows compute their routed experts on the CPU executor (offload/hybrid) instead of streaming every layer's bank; `0` disables |
| `FREETOKEN_STAGED_COPY` / `FREETOKEN_STAGED_COPY_MB` | on / `32` | Whole-layer prefill copies of non-pinned bank layers go through two pinned staging buffers of this size |
| `FT_SPEC_TRACE` | `0` | Log the first n verify windows of `--spec-mtp` (input ids, drafts, samples, accepted, next drafts, top-3 logits) |
| `FT_SPEC_PROFILE` | off | Per-phase wall time of the verify step (target forward, sample, rollback, head window, head chain), logged every 20 steps |
| `FT_SPEC_PLAIN` | off | Drafts are produced but never verified (plain decode; measures the head's own cost) |
| `FT_SPEC_MAX_DRAFTS` | unset | Cap the drafts per verify window; `0` gives one-row windows (verify path vs plain decode parity checks) |
| `FT_SPEC_CHECK_STEP` | `0` | Cross-check the first n one-row verify windows against the plain decode path from the same state (per-layer residual, final logits, GDN state) |
| `FT_SPEC_NO_GRAPH` / `FT_SPEC_NO_MTP_GRAPH` | off | Keep the verify window / the draft head eager (A/B runs) |
| `FT_SPEC_GRAPH_MIN_FREE_MB` | `256` | Drop the verify-window graphs when less VRAM than this is left after capture (`0` keeps them) |
| `FREETOKEN_PREMAP_VRAM` | off | Pre-map the remaining VRAM into the allocator cache at startup; an experiment that did not help on the 2060 (per-stream pools), left as a knob |

## Known limitations

- Video and the Anthropic / Responses adapters (still text-only) are not covered.
- `--spec-mtp` serves one request at a time, runs eagerly (no CUDA graph for the verify window
  yet) and reads the head from modelopt MIXED_PRECISION checkpoints only.
- Image prompts bypass the shared prefix cache (by upstream design), so a conversation with images
  is prefilled in full every turn.
- On Turing, use `--dtype float16`. A long prefill chunk costs ~5 s of expert streaming on the
  2060 under WSL (every layer's bank crosses PCIe at ~3.4 GB/s) plus ~1.3 ms per token; extends
  of up to `FREETOKEN_CPU_PREFILL_MAX_TOKENS` (256) rows -- a chat turn behind a cached prefix
  -- skip the streaming and compute their experts on the CPU executor instead (a follow-up
  turn answers in 2-3 s including 64 generated tokens, against ~6 s before).
- Under WSL2 on a full 6 GB card, PyTorch's expandable-segment allocator intermittently died
  with `CUDA driver error: device not ready` when it had to release cached segments while
  other streams were busy. The Turing prefill scratches (MoE dequant chunks, fp8 dequant) are
  therefore fixed-size and allocated once at startup, before the graphs, so serving never
  grows or shrinks segments through the driver; keep `--memory-ratio 0.80` there so the
  planner leaves ~0.5 GiB for them.
- The Triton attention backend is used on Turing; flashinfer's JIT attention fails there at
  head_dim 256.
- DeepStack vision checkpoints (Qwen3-VL proper) are refused; only checkpoints with an empty
  `deepstack_visual_indexes` are supported.

## Related work

- [trev222/pocketai-freetoken-sm75](https://github.com/trev222/pocketai-freetoken-sm75) reached
  Turing first (RTX 2070 Max-Q, Windows, FreeToken PR #24 era): a standalone compatibility harness
  that swaps the unsupported Triton kernels for PyTorch ops and its own batch-one MoE backend, with
  Qwen3.6-35B-A3B NVFP4 decoding at ~32 tok/s. Its report lists prompt caching, streaming and a
  serial-prefill bottleneck (30 s TTFT at 1.5k tokens) as open. This fork takes the other route:
  keep upstream's server and kernels, change the six places that break, and validate each kernel
  against a reference so prompt caching, streaming and the OpenAI/Open WebUI path stay upstream's.
- Upstream PR #131 (GGUF: all quant types, Qwen3.5-MoE GGUF) compiles its vendored GGUF kernels for
  sm_75+, but targets the GGUF expert path on Ampere-class cards; the flashinfer attention and
  Triton fallbacks that fail on Turing are not part of it.

## Keeping up with upstream

The fork is a dozen commits on top of `af71ba4`, touching a small set of files (see `git log
af71ba4..`). Rebasing onto a newer upstream is expected to be straightforward until upstream ships
its own multimodal serving or Turing support, at which point the corresponding part of this fork
should be dropped in favour of the official code.
