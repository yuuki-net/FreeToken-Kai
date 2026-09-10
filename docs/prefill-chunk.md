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

## Where this matters less

A card with room to spare will measure, find that the configured chunk fits, and keep it —
the log says so and nothing changes. This is a feature for machines running near the edge,
which is the same population that needs `--kv-cache-dtype`, and for the same reason.
