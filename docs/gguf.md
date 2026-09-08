# GGUF 3-bit and 2-bit experts: what stops you (2026-09-07)

Nothing in this fork uses GGUF. This page exists so that the next person who has the same idea can
find out in ten minutes what took a day, and then decide with the numbers in front of them.

**The idea.** A 125B MoE with NVFP4 experts needs 63 GiB of host RAM for the banks. The published
GGUF quantisations of the same checkpoint are much smaller — UD-Q2_K_XL is 78.9 GB in total against
135 GB. If FreeToken could serve those, the RAM problem would be solved by arithmetic rather than by
the file mapping in [bank-ram.md](bank-ram.md).

**The conclusion.** It is possible, it is not cheap, and the published quantisations do not load as
they are. If you want smaller experts today, `--moe-bank-ram` is the shorter road.

## The blocker you will hit first

Every published UD-series quantisation of Qwen3.8-Flash-Next has **expert tensors whose ggml type
differs from layer to layer**. From the headers of `unsloth/Qwen3.8-Flash-Next-GGUF` (read from the
first 12 MiB of each shard; the totals match the sizes Hugging Face reports), UD-Q3_K_XL is:

```
ffn_gate_exps   IQ3_XXS x47,  IQ4_XS x1   (layer 2)
ffn_up_exps     IQ3_XXS x47,  IQ4_XS x1   (layer 2)
ffn_down_exps   IQ4_NL  x43,  Q8_0   x5   (layers 2, 4, 30, 46, 47)
```

The expert slot pool is one allocation across all layers, so a bank whose type varies by layer
raises `NotImplementedError`. This is not a property of one bad file: the "UD" (unsloth dynamic)
method deliberately raises precision on the layers it measures as sensitive, so **every** UD build
has this shape. A uniform quantisation would load, but you would be making it yourself.

## Sizes, if you get past that

| Build | Experts | bpw | PLE / Ngram | dense | total |
|---|---|---|---|---|---|
| NVFP4 (what this fork runs) | **68.1 GB** | 4.51 | 51.2 GB (fp8) | — | 135 GB |
| UD-Q2_K_XL | 46.1 GB | 3.05 | 28.8 GB (IQ4_NL) | 4.0 | 78.9 |
| UD-Q3_K_XL | 55.8 GB | 3.70 | 28.8 GB | 5.4 | 90.0 |
| UD-IQ4_XS | 59.5 GB | 3.94 | 28.8 GB | 5.4 | 93.7 |
| UD-Q4_K_XL | 77.0 GB | 5.10 | 28.8 GB | 5.5 | 111.3 |

Worth noting that most of the saving is in the PLE table (51.2 -> 28.8 GB), not in the experts.

## The rest of the bill

Upstream has GGUF support in flight —
[FlashML-org/FreeToken#131](https://github.com/FlashML-org/FreeToken/pull/131) (`feat/generic-gguf`,
opened 2026-08-24, 37 files and 5,920 lines, still open as of 2026-09-07). It dispatches all 21 GGUF
types, adds GGUF expert banks and K-quant CPU kernels, and has loaders for qwen3_5_moe, qwen3_moe
and deepseek_v4. Against the use case above:

| Constraint | What it costs you |
|---|---|
| No `qwen4exp` (Flash-Next) among the supported architectures | Write the loader; the qwen3_5_moe one is 1,261 lines |
| Per-layer type variation raises `NotImplementedError` | See above — rules out every published UD build |
| CPU kernels exist only for Q4_0 / Q4_K / Q6_K | Anything else forces `--moe-backend offload` |
| `gate_up` and `down` must share a type for the CPU path | Kills the Q4_K_M family, where `down` is Q6_K |
| CPU kernels are scalar reference implementations (AVX2/VNNI explicitly deferred) | This is the one to worry about: hybrid decode on this hardware needs ~44 GB/s from the CPU MoE path, and a scalar kernel will not be the same order |
| TP=1 only, no parallel reader, NextN/MTP dropped | Extra work to put it on the layer split |

The last row is the quiet one. Shrinking the experts helps only if the CPU can still dequantise and
multiply them fast enough; on this host that rate, not capacity, is what caps decode. See the
`ft bench bw` section of [kai.md](kai.md).

## If you are going to try anyway

Read the headers before you download 90 GB. `tools/gguf_probe.py` in the fork's working notes takes
the first 12 MiB of each shard and prints the per-tensor ggml types and the totals, which is how the
table above was made. A build whose expert tensors are all one type is the only kind worth pursuing.
