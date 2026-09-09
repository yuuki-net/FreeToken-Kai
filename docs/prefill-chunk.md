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

At startup it runs one small prefill (1024 tokens), measures the transient with the
allocator's own high-water mark, and divides: **124.5 KiB per token** on that machine. Then
it solves for the largest chunk whose transient fits inside `--prefill-chunk-budget` of the
free VRAM (default 0.55) and lowers `--max-prefill-length` to it, never raising it.

```
--prefill-chunk-budget: 124.5 KiB/token of prefill transient and 0.46 GiB free at 55%
  -> --max-prefill-length 8192 would need 0.97 GiB; using 2048 instead
```

It measures rather than searching downwards from the configured size on purpose: probing
8192 to see whether it fits means allocating the very thing that kills the process, at
startup, on every boot.

Then it **re-solves before every prefill**, against the VRAM free at that moment:

```
prefill chunk 2048 -> 1792 (0.40 GiB usable)
prefill chunk 1792 -> 2048 (0.48 GiB usable)
prefill chunk 1792 -> 1280 (0.28 GiB usable)   <- something else took 300 MB
prefill chunk 1280 -> 2048 (0.46 GiB usable)   <- and gave it back
```

That costs a driver query and an integer divide, and it changes nothing that is allocated —
the chunk is a bound, not a buffer. A prefill chunk runs for seconds; the query does not
register. Cached-but-unused allocator blocks count as available, because that is exactly
what the transient will be served from.

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

## Where this matters less

A card with room to spare will measure, find that the configured chunk fits, and keep it —
the log says so and nothing changes. This is a feature for machines running near the edge,
which is the same population that needs `--kv-cache-dtype`, and for the same reason.
