# Sizing the prefill chunk (`--prefill-chunk-budget`)

A long prompt is prefilled in chunks. The chunk size is not just a scheduling detail: it is
the unit of *transient* memory. Every chunk the linear-attention kernels allocate buffers
proportional to it, use them, and free them, so what the engine has to find room for scales
with the chunk and not with the prompt.

When that transient is the size of the free VRAM, the run gets slow first and dies later.
Measured on an RTX 2060 6 GB serving Ornith-1.5-35B-A3B, one 30,448-token prompt:

| `--max-prefill-length` | transient needed | result |
|---|---|---|
| 8192 (upstream default) | 0.97 GiB | **1081 s**, and on a busier day it killed the server |
| 4096 | 0.49 GiB | 232 s |
| 1792 | 0.21 GiB | 249 s |

The card had 0.39-0.63 GiB free depending on what the desktop was doing. 8192 never fit.

## What the engine does about it

At startup it runs one small prefill (1024 tokens) and measures the transient with the
allocator's own high-water mark: **124.5 KiB per token** on that machine. It measures rather
than searching downwards from the configured size on purpose — probing 8192 to see whether it
fits means allocating the very thing that kills the process, at startup, on every boot.

Startup only measures. `--max-prefill-length` stays the ceiling for the run, and the chunk is
solved **before every prefill**, against the VRAM free at that moment:

```
--prefill-chunk-budget: 124.5 KiB/token of prefill transient and 0.46 GiB free at 55%
  -> --max-prefill-length 8192 would need 0.97 GiB; a prefill starting now would use 2048.
     8192 stays the ceiling and each prefill is solved against the VRAM free then
prefill chunk 2048 (0.46 GiB usable)           <- the first prompt
prefill chunk 2048 -> 1792 (0.40 GiB usable)
prefill chunk 1792 -> 1280 (0.28 GiB usable)   <- something else took 300 MB
prefill chunk 1280 -> 2048 (0.46 GiB usable)   <- and gave it back
```

That costs a driver query and an integer divide, and it changes nothing that is allocated —
the chunk is a bound, not a buffer. A prefill chunk runs for seconds; the query does not
register. Cached-but-unused allocator blocks count as available, because that is exactly
what the transient will be served from.

The boot number is deliberately not written back. Free VRAM at startup is whatever the moment
happened to hold — a desktop drawing, another model still shutting down, the sizer's own probe
— and a value written into the configuration there would cap the run with no way back up short
of a restart.

**`--pp-size` is the exception.** The chunk sizes the residual one rank hands the next, so it
cannot be a per-rank answer, and each rank reaches the per-prefill solve on its own schedule
with nowhere safe to re-agree. There the ranks agree at boot on the tightest numbers any of
them measured, that value is written into the configuration, and it stands for the run.

## Choosing the budget

`--prefill-chunk-budget` is the share of free VRAM one chunk's transient may take. The rest
is headroom for whatever else wants the card.

| your machine | try |
|---|---|
| serves and nothing else | 0.8 - 0.9 |
| a desktop you also work on | 0.5 - 0.6 (default 0.55) |
| a desktop that grabs VRAM in jumps — games, video editing, another model | 0.3 - 0.4 |
| you want the old behaviour: no measurement, no adaptation | **0** |

At 0 nothing is measured and `--max-prefill-length` is used exactly as written. A value
outside `(0, 1]` is ignored the same way, on the grounds that a nonsense setting should not
invent behaviour.

Headroom is not free. On the 2060, 0.55 chose 1792-2048 where a fixed 4096 ran the same
prompt about 13% faster — when it fit. The whole point is that "when it fit" is a property
of the moment, and the moment changes.

## Two GPUs

Under `--pp-size` the chunk is settled once, at startup, from the tightest numbers any rank
measured -- most transient per token, least VRAM free -- and then left alone. It has to be the
same number on every rank: a rank hands on one residual row per token of the chunk, and the
receiver sizes its buffer from its own answer (Qwen3.8-Flash-Next: 20,480 bytes a row). Ranks that solved it separately
(3328 on one card, 3072 on the other, both correct for their own card) killed the first prefill
in gloo with `Received data size doesn't match expected size`.

So the re-solve before each prefill is a single-GPU feature. A two-card run keeps the boot
value, and a desktop that takes 300 MB mid-session no longer shrinks the chunk to match.

Which is also why the budget is worth more on two cards than on one. The chunk is fixed for the
run, and with the experts in host RAM each chunk pays one full bank transfer per layer on each
rank; a rank on a slow slot needs the chunk to be wide enough that its compute covers that
transfer. On two RTX 3060s serving Qwen3.8-Flash-Next (`--spec-mtp 5 --prefill-mixer-pieces 2
--max-prefill-length 8192`), a 25k-token prompt:

| `--prefill-chunk-budget` | chunk | prefill |
|---|---|---|
| 0.55 (default) | 3,584 | 510 tok/s |
| 0.75 | 5,120 | **702 tok/s** |

The second rank's slot moves 33 GiB of banks per chunk in 5.6 s there, and its compute for a
3,584-token chunk is shorter than that — so the transfer showed up as waiting (0.69 s per 1k
tokens) until the chunk grew. Going further did not help: `--spec-mtp 0` frees enough for a
6,144-token chunk and the prompt ran at the same 702 tok/s, while decode dropped from 19-22 to
18 tok/s without the draft head.

The boot value is checked once more at the end of startup. The measurement has to come before
the prefill warmup, which runs at the chunk it chose, and so before `--spec-mtp` captures its
verify-window and draft-head graphs, which keep their memory pools. Once those are in, the ranks
agree again on the least VRAM any of them has usable and narrow the chunk if it no longer fits
the same budget share; it never widens. On the two RTX 3060s with `--spec-mtp 5` that took 2816
to 2560 (`--prefill-chunk-budget: 2816 -> 2560 after the boot`).

## Wider chunks: `--prefill-mixer-pieces`

With the experts in host RAM, every prefill chunk streams every layer's expert bank to the GPU.
The number of chunks a prompt is cut into is the number of full bank transfers it pays, so a
wider chunk is faster even when the arithmetic per token is the same.

What caps the width is the transient, and the transient is not the experts. Measured per part
of one layer on the 2060 (Ornith, 1024 tokens):

| part | KiB per token |
|---|---|
| GatedDeltaNet | **116.5** |
| full attention | 79.8 |
| MoE | 16.7 |

The parts of a layer run one after another, so the layer's peak is its largest part: the GDN.

`--prefill-mixer-pieces N` runs each layer's sequence mixer (GDN or attention) over N
consecutive pieces of the chunk inside one forward, and the MoE -- together with everything
else that works token by token -- over the whole chunk, so its bank still crosses the bus once
per layer. Each piece continues the one before it exactly as a chunk continues the previous
chunk, with its own attention and GDN metadata built by the same code the scheduler uses. The
startup measurement runs a real forward, so it sees the smaller peak by itself, and the solver
above picks a wider chunk.

**Raise `--max-prefill-length` with it.** The measured width is still capped by the ceiling, and
the default 8192 is reached quickly. Raising the ceiling alone does nothing: the transient was
the limit, and it is unchanged.

RTX 2060 6 GB, Ornith-1.5-35B-A3B, a 19,869-token prompt:

| | transient | chunks | first token | prefill |
|---|---|---|---|---|
| default | 124.5 KiB/token | about 3072 x 7 | 40.5 s | 490 tok/s |
| `--max-prefill-length 16384` only | 124.5 | about 3072 x 7 | 37.2 s | 534 tok/s |
| `--prefill-mixer-pieces 2` | 68.3 | 4864 / 5888 / 5888 / 3229 | 30.6 s | 649 tok/s |
| `--prefill-mixer-pieces 4 --max-prefill-length 16384` | 40.2 | 6912 / 9984 / 2973 | **27.5 s** | **722 tok/s** |

A later run of the same four settings read 430 / 473 / 602 / 649 tok/s: the absolute numbers move
with what the desktop is doing, the order does not.

2x RTX 3060 12 GB, Qwen3.8-Flash-Next, `--pp-size 2 --spec-mtp 5`, a 19,904-token prompt:

| | transient | chunk | chunks | first token | prefill |
|---|---|---|---|---|---|
| `--max-prefill-length 4096` | 270.0 KiB/token | 3072 | 7 | 45.6 s | 437 tok/s |
| `--max-prefill-length 16384` only | 270.0 | 2816 | 8 | 48.8 s | 408 tok/s |
| `--prefill-mixer-pieces 2 --max-prefill-length 8192` | 195.7 | 4352 | 5 | **36.5 s** | **546 tok/s** |
| `--prefill-mixer-pieces 4 --max-prefill-length 16384` | 195.7 | 4352 | 5 | 36.6 s | 544 tok/s |

**On Flash-Next, 2 is the setting; 4 adds nothing.** Measured by part, at two pieces the peak is
no longer a mixer: it is PLE on the first pipeline rank (170.5 KiB per token of its own) and the
routed experts on the second (98.7). Running those over the pieces as well was tried. It worked
for memory -- four pieces came down to 108.1 KiB per token and a 7,936-token chunk -- and made
prefill slower: 562 tok/s at two pieces, 507 at four held to the same 5,376-token chunk, 493 at
four with the wider chunk, the same on a second prompt of each boot. Each extra piece costs a fixed
amount in every layer, and on these two cards a chunk fewer did not win that back. That change is
not in this fork.

Where it applies: Qwen3.8-Flash-Next, on one GPU or under `--pp-size` (a piece never leaves its
rank; what crosses to the next rank is the whole chunk, as before), including the `--spec-mtp`
draft head, which fills its own KV over the same rows and reuses the same pieces; and the
Qwen3.5-MoE family on one GPU. Other models ignore it. A chunk runs whole when pieces cannot
describe it: more than one request prefilling in the same step, an image prompt (M-RoPE), an
MTP verify window, or a chunk too short to split.

The part that took care is the hybrid prefix cache. The scheduler picks the GDN snapshot slot
for a chunk before the forward starts, so only the last piece takes the snapshot, into the slot
already chosen, and the pieces are cut on the 64-token grid the snapshot boundaries live on so
that the last piece's deepest boundary is the chunk's. In every measured run the follow-up
question resumed the whole document from the prefix cache.

On correctness: at the shipping QSA geometry and a reduced width, Flash-Next's QSA layer, GDN
layer (including the state the next chunk resumes from) and a pair of full decoder layers with
hyper-connections and a real MoE are bit-identical in 2, 3 and 4 pieces and whole. Served
temperature-0 outputs are not a usable check: a different chunk width alone, with no pieces,
makes them diverge after a few dozen characters.

## Where this matters less

A card with room to spare will measure, find that the configured chunk fits, and keep it —
the log says so and nothing changes. This is a feature for machines running near the edge,
which is the same population that needs `--kv-cache-dtype`, and for the same reason.
