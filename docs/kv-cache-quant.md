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

## What it buys, and what it costs

It buys VRAM. On an RTX 2060 6 GB serving Ornith-1.5-35B-A3B at 64k, the KV goes from
1.25 GiB to 0.35 GiB, and `--moe-cache-auto` turns that into expert slots: 358 to 902.

**It does not buy speed, and past a few thousand tokens of context it costs speed.** Measured
on that card with `tools/longctx.py`, four configurations, two runs each:

| context | 16-bit KV | `q4_0` |
|---|---|---|
| ~2k | 31.5 tok/s | 32.0 tok/s |
| ~30k | **20.2 / 20.3 tok/s** | **13.6 / 13.5 tok/s** |

Decode reads the whole KV every step, so at 30k it dequantizes 30k tokens of codes per step
across every full-attention layer. Prefill pays too: 210 s against 242 s for the same 30k
prompt. Both results reproduced on every run.

The deeper expert cache is real -- the VRAM hit rate went from 42.2% to 50.7% on a warm cache
-- but on that machine it moved decode by about 2%, because most of the CPU expert work comes
from layers pinned to the CPU at startup, which never touch the slot cache at all
(`--moe-cpu-layers auto` says how many in the log). Raising the pin budget so fewer layers are
locked did not help either: the same slots then spread across more layers.

So: **use this when you need the VRAM** -- for context you could not otherwise fit, or to leave
room for something else. Do not use it expecting a speed-up, and do not use it at long context
unless the capacity is worth a third of your decode rate.

Be aware that on a machine whose host RAM cannot pin the whole expert bank, most of the CPU
expert work comes from layers that are locked to the CPU at startup (`--moe-cpu-layers auto`
says so in the log) and never consult the VRAM cache at all. A deeper cache does not help
those layers. On the 2060 above, 21 of 40 MoE layers are locked that way, and the decode
gain from doubling the cache was about 2%. Check that line in your startup log before
expecting a speedup.

## Long prompts: read this before raising the context

Making the KV fit is the easy half. What a long context costs on a small card is prefill,
and prefill is where this configuration breaks.

Measured on the 2060 (Ornith, `q4_0`, one 30,448-token prompt, `tools/longctx.py`):

| `--max-prefill-length` | prompt | result |
|---|---|---|
| 8192 (default) | 30,448 tokens | 1081 s (~28 tok/s prefill) |
| 4096 | 30,448 tokens | **232 s (~134 tok/s)** |

Same work, 4.7x apart. The prefill chunk is the unit of transient memory: the GDN kernels
allocate buffers proportional to it (about 0.4 GiB at 8192 for this model, half that at
4096), per chunk, freed after each. When that transient is the same size as the free VRAM,
the run gets slow — and sometimes dies. On the same machine a 30k prompt at 262,144 context
killed the server outright in the GDN prefill kernel ("CUDA driver error: device not ready",
this machine's symptom of running out of VRAM), because only ~0.6 GiB was left for it.

It is not deterministic: the identical configuration passed the same prompt on a later run
when the desktop happened to be using 150 MB less VRAM. **On a machine you also use for
other things, a configuration that leaves half a gigabyte free will work some days and not
others.**

So, if you raise the context:

- pass `--max-prefill-length 4096` (or lower). On this card it is both faster and safer than
  the 8192 default, and there is no reason to think 4096 is optimal -- it is the value that
  worked, not a measured optimum.
- leave real headroom. `--moe-cache-auto` hands the VRAM the KV gives back to the expert
  cache, right up to a small margin; on a desktop machine, set `--moe-cache-size` explicitly
  or lower `--memory-ratio` instead of letting it fill.
- watch decode, not just capacity. See the table above: at ~30k this costs a third of the
  decode rate, reproducibly. It gets worse as the context grows, not better.

## Platform note

Every number here was measured under **WSL2 on Windows**. Two things behave differently on
native Linux, both of which affect this feature:

- Windows can back a GPU allocation that does not fit with system RAM (it shows up as
  "shared GPU memory"), so an over-subscribed run gets slow rather than failing. Native
  Linux fails the allocation instead.
- FreeToken caps host pinning at 40% of RAM on WSL and does not cap it at all on native
  Linux, so the CPU/GPU split of the MoE layers -- and therefore decode speed -- can differ
  on the same hardware.

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

Ornith-1.5-35B-A3B on an RTX 2060 6 GB (WSL2). This is the configuration actually in daily
use on that card -- 64k of context, the VRAM the KV gives back spent on expert slots:

```bash
ft serve --model ~/models/Ornith-1.5-35B-A3B-NVFP4 --dtype float16 \
  --moe-backend hybrid --disable-moe-prefill-overlap --max-running-req 1 \
  --host-embedding --kv-cache-dtype q4_0 \
  --kv-reserve-tokens 65536 --max-seq-len-override 65536 \
  --memory-ratio 0.82 --moe-cpu-threads 6
```

The startup log tells you what it bought:

```
--moe-cache-auto resolved moe_cache_size=902 num_pages=65571
Allocating 65571 tokens for KV cache, K + V = 0.35 GiB
```

902 expert slots against 358 at 16-bit, for the same context.

Raising both token counts to 262144 does allocate (1.41 GiB of KV, 264 slots), but see
"Long prompts" above before you do: that setting killed the server on a 30k prompt on this
card, and it needs `--max-prefill-length 4096` at minimum. It is not a configuration this
fork can recommend yet.
