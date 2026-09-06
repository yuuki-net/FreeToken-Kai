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
| RTX 2060 6 GB, 32 GB RAM, Windows 11 + WSL2 (`memory=24GB`) | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (35B MoE, 3B active, vision) | Text and image input work. Decode 25-37 tok/s (`--moe-backend hybrid`, `--dtype float16`). Prefill: a 2785-token prompt in ~8.5 s (68 s before the Turing GEMM changes); ~5 s of that is the per-chunk expert streaming, the rest ~1.3 ms/token |
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

## Environment variables added by this fork

| Variable | Default | Meaning |
|---|---|---|
| `FT_IMAGE_MAX_PIXELS` | `1048576` | Resolution cap the image processor keeps (soft tokens per image = pixels / 1024) |
| `FT_IMAGE_EMBED_CACHE` | `32` | LRU entries of vision-tower output per image (chat clients resend every image each turn); `0` disables |
| `FREETOKEN_FP8_SCRATCH_GEMM` | arch (on below Ampere) | fp8 W8A16 prefill GEMM as dequant + cuBLAS instead of the inline-dequant Triton kernel |
| `FREETOKEN_NVFP4_MOE_SCRATCH` | arch (on below Ampere) | NVFP4 prefill MoE as chunked dequant + per-expert cuBLAS instead of the inline-dequant kernel |
| `FREETOKEN_NVFP4_MOE_ARITH` | arch (on below Ampere) | Arithmetic (gather-free) e2m1 dequant in the prefill MoE kernel; bit-identical, speed knob only |

## Known limitations

- Video, the Anthropic / Responses adapters (still text-only), and speculative decoding with image
  prompts are not covered.
- Image prompts bypass the shared prefix cache (by upstream design), so a conversation with images
  is prefilled in full every turn.
- On Turing, use `--dtype float16`. Prefill has a fixed cost of ~5 s per chunk on the 2060 under
  WSL (the experts of the CPU-side layers are streamed from pageable host memory each chunk);
  beyond that it is ~1.3 ms per token.
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
