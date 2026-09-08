# Quantized KV cache (`--kv-cache-dtype`)

Stores the paged KV cache as block-quantized codes instead of 16-bit floats.

```
--kv-cache-dtype {auto,q8_0,q4_0}      # default: auto (16-bit, unchanged)
```

| | bytes per value | vs 16-bit |
|---|---|---|
| `auto` | 2.0 | — |
| `q8_0` | 1.0625 | 1.88x smaller |
| `q4_0` | 0.5625 | 3.56x smaller |

## What it is actually for

On a small card the paged KV and the MoE expert slot cache come out of the same VRAM, and
`--moe-cache-auto` hands whatever the KV does not take to the experts. So the flag has two
quite different uses, and the second one is usually the interesting one:

- **more context in the same VRAM** — an RTX 2060 6 GB serves Ornith-1.5-35B-A3B at
  **262,144 tokens** with `q4_0` (1.41 GiB of KV). The same card at 16-bit tops out at 64k,
  which already costs 1.25 GiB.
- **a deeper expert cache at the same context** — at 64k on that card the expert slots go
  from 358 (16-bit) to 713 (`q8_0`) to 902 (`q4_0`), and the measured VRAM hit rate from
  42.2% to 49.4% to 50.7%.

Be aware that on a machine whose host RAM cannot pin the whole expert bank, most of the CPU
expert work comes from layers that are locked to the CPU at startup (`--moe-cpu-layers auto`
says so in the log) and never consult the VRAM cache at all. A deeper cache does not help
those layers. On the 2060 above, 21 of 40 MoE layers are locked that way, and the decode
gain from doubling the cache was about 2%. Check that line in your startup log before
expecting a speedup.

## Where it applies

Only to models whose KV lives in the plain paged pool (`MHAKVCache`), served by the Triton
attention backend. Everything else is **refused at startup**, with a message naming the
reason — a quantized slab read by code that does not know the layout returns plausible wrong
numbers rather than an error, so none of these are left to chance.

| Model | Attention | KV pool | `--kv-cache-dtype` |
|---|---|---|---|
| Qwen3.5-MoE family (Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B) | full | `MHAKVCache` | **supported** (tested) |
| Qwen3 / Qwen2 dense, Llama, Mistral | full | `MHAKVCache` | supported (untested) |
| gpt-oss-20b / gpt-oss-120b | full + sliding window | `HybridSWAKVCache` | refused |
| Qwen3.8-Flash-Next | QSA (compressed-block sparse) | `QSAKVCache` | refused |
| GLM-5.3-Flash | DSA | `KpoolDSAKVCache` | refused |
| DeepSeek-V4-Flash | DSV4 | `DSV4PagedKVCache` | refused |
| MiniMax-M3 | BSA | `BSAKVCache` | refused |
| MLA checkpoints | latent KV | `MLAKVCache` | refused |

The refused families all keep secondary tiers next to the paged slab — an SWA window pool,
a sparse index slab plus its pending ring, a latent KV — which stay 16-bit and are read by
their own kernels. Supporting them is per-family work, not a flag.

**Backend**: only `--attention-backend triton` reads the code slabs. `auto` picks Triton on
Turing but flashinfer on sm_80 and newer, so on an Ampere or later card you have to ask for
Triton explicitly:

```bash
ft serve --model ... --kv-cache-dtype q4_0 --attention-backend triton
```

Without it the run stops at config time rather than serving wrong numbers.

## Accuracy

`q8_0` is close to free: the error is under half a step of a 32-value block scale, and
attention output tracks the unquantized answer to ~1e-4 relative.

`q4_0` costs something real. On synthetic worst-case input (iid Gaussian, no structure for
the block scale to exploit) the attention output lands at cosine ≈ 0.99 against unquantized;
on real prompts the answers stayed correct in Japanese prose, arithmetic-with-working and
Python generation, but this has not been measured on a benchmark suite. Treat `q4_0` as a
capacity trade you should sanity-check on your own workload, and prefer `q8_0` when the
VRAM it buys is enough.

Note that the names are the familiar llama.cpp spellings but **the block layout is not
GGUF's**: the range here is symmetric, so 4-bit spends one of its sixteen codes on nothing
and its step is `absmax/7` rather than `absmax/8` (14% coarser). Nothing is serialized — the
KV cache dies with the process — so byte compatibility would have bought nothing.

## Example

Ornith-1.5-35B-A3B on an RTX 2060 6 GB (WSL2), 256k of context:

```bash
ft serve --model ~/models/Ornith-1.5-35B-A3B-NVFP4 --dtype float16 \
  --moe-backend hybrid --disable-moe-prefill-overlap --max-running-req 1 \
  --host-embedding --kv-cache-dtype q4_0 \
  --kv-reserve-tokens 262144 --max-seq-len-override 262144 \
  --memory-ratio 0.82 --moe-cpu-threads 6
```

The startup log tells you what it bought:

```
--moe-cache-auto resolved moe_cache_size=264 num_pages=262245
Allocating 262245 tokens for KV cache, K + V = 1.41 GiB
```
