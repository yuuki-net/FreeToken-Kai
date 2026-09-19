# Why switching conversations re-prefills everything on hybrid GDN models

On Qwen3.5-MoE and Qwen3.8-Flash-Next, the prefix cache can return a long conversation in a
fraction of a second — and then, after you have touched a few other conversations, stop
returning it at all, even though KV usage is near zero. This page explains why, and the one
flag that changes it.

## The limit is not KV

A hybrid model's attention layers keep KV, but its GDN layers keep a **recurrent state**, and
the next token can only be computed from a state that already has the whole prefix folded in.
So a prefix is reusable only if the cache still holds a **snapshot of that state** at a
boundary inside it. KV alone is not enough.

Snapshots live in a fixed pool of slots in VRAM:

```
slots = 4 × max_running_req               (what running requests need: live + 2 ping-pong + 1 committed)
      + max(4, ratio × max_running_req)   (the cache of snapshots other prompts can resume from)
      + 1                                 (padding)
```

With the usual single-user setting (`--max-running-req 1`) and the default ratio of 2.0, the
cache part is **4 snapshots**. That, not KV, is what bounds how many conversations stay
reusable. `--linear-state-cache-ratio` sets it.

## Measured

Sixteen different 643-token prompts were sent once, then again in reverse order; the table
counts how many of the second round still hit the prefix cache (`#cached-token` non-zero)
before the first miss. `--max-running-req 1` throughout, KV usage 0% in every run.

| Machine and model | `--linear-state-cache-ratio` | GDN state slots | Conversations kept |
|---|---|---|---|
| RTX 2060 6 GB, Ornith-1.5-35B-A3B-NVFP4 | 2 (default) | 8 | **2** |
| | 8 | 12 | **4** |
| | 16 | 20 | **8** |
| RTX 3060 12 GB × 2, Qwen3.8-Flash-Next-NVFP4 (`--pp-size 2`) | 2 (default) | 8 | **5** |
| | 8 | 12 | **7** |

**Raising the ratio keeps more conversations on both, but not by the same rule.** On Ornith the
count doubles with the cache (a conversation there costs two snapshots); on Flash-Next the
default already keeps five and ratio 8 adds two. How many snapshots a conversation costs
depends on where the model can checkpoint — the hits above landed at 640 tokens on Ornith and
at 512 on Flash-Next — so **do not carry one model's numbers to another**: measure yours
(below).

## What it costs

Each extra slot is one GDN state of VRAM — every GDN layer's recurrent and conv state — and it
comes out of the expert cache. On the RTX 2060 with Ornith (30 GDN layers × 32 heads × 128 × 128
× fp32 ≈ 62 MiB per state) that is about 36 expert slots per GDN slot: 925 → 780 expert slots at
ratio 8, 585 at ratio 16 (that run also halved `--kv-reserve-tokens` to fit).

From a routing trace of the same model on the same machine, the expert cache's hit rate falls
roughly from 72% at 925 slots to 66% at 780 and 57% at 585, which that machine's cost model puts
at about **−6% decode for ratio 8** and **−14% for ratio 16**. These are estimates read off the
measured hit-rate curve, not measured decode speeds. On a larger card the same slot is a smaller
share of what the expert cache has to work with, so the trade is cheaper.

## When to change it

- **You switch between several long documents or conversations** and the second visit to one
  re-prefills from scratch: raise the ratio, then check with the log (below) that the number
  you keep in play now fits.
- **You mostly continue one conversation**: leave it at the default. Continuing a conversation
  hits, and the VRAM is worth more to the expert cache.

## Keeping snapshots in RAM instead (`--linear-state-host-slots`)

Raising the ratio buys conversations with VRAM. `--linear-state-host-slots N` buys them with
pinned host RAM instead: when the VRAM slots run short, the least recently used snapshot the
cache holds is copied down to one of N host slots and its VRAM slot is reused, instead of the
snapshot being thrown away. A later prompt that resumes from it copies it back up first.

```bash
ft serve ... --linear-state-host-slots 24
```

Measured on the RTX 2060 with Ornith-1.5-35B-A3B-NVFP4 (`--max-running-req 1`, default ratio),
sixteen different 473-token prompts sent once and again in reverse:

| | Conversations kept | Time to first token, hit | miss |
|---|---|---|---|
| off (default) | **2** of 16 | 0.69 s | 3.00 s |
| `--linear-state-host-slots 24` | **14** of 16 | 0.64 s | 2.74 s |

Fourteen is what the arithmetic says: 4 VRAM cache slots + 24 host slots hold 28 snapshots, and a
conversation there costs two. A hit restored from RAM is as fast as one from VRAM; the copy back
up is a few milliseconds.

On two RTX 3060s with Qwen3.8-Flash-Next-NVFP4 (`--pp-size 2`), twenty-four different 703-token
prompts, the same way, alternating twice:

| | Conversations kept | Time to first token, hit | miss | Decode |
|---|---|---|---|---|
| off (default) | **5** of 24 | 0.30-0.41 s | 7.3-7.4 s | 19.15-19.18 tok/s |
| `--linear-state-host-slots 24` | **24** of 24 | 0.49-0.50 s | (none) | 19.07-19.27 tok/s |

All twenty-four stayed reusable, so 24 host slots are not the ceiling there. A hit restored from
RAM costs about 0.15 s more than one from VRAM on these cards, a fifteenth of the re-prefill it
saves, and decode is unchanged.

- **Cost.** One GDN state of pinned RAM per slot (1.44 GiB for 24 on Ornith; with `--pp-size`
  each rank holds only its own GDN layers). It counts against the pin budget, so on WSL, where
  pinning is capped, the expert banks get that much less: on the 2060 `--moe-cpu-layers auto`
  moved from 21 to 24 CPU layers. Decode did not measurably change: 31.0 / 34.6 tok/s off,
  34.9 / 28.3 on, alternating (1500 tokens each) -- the spread within one setting is larger than
  the gap. A card or a quota where the banks are tighter may show a cost; check yours the same way.
- **Nothing else changes.** Off by default; with it off the cache behaves exactly as before.
  VRAM use is the same either way.
- **Pipelines.** Every choice (which snapshot moves down, which one is dropped) follows from the
  tree and the slot counts, which every `--pp-size` rank's replica shares, so the ranks stay in
  step without exchanging anything.
- **With `--prefix-disk-cache`.** Both work together: what reaches the disk is read from either
  tier.

## How to tell on your machine

The decode log prints the pool as `#mamba-slot: used/total`, and each prefill prints
`#cached-token`. If a prompt you sent earlier comes back with `#cached-token: 0` while
`token usage` is low, the snapshot pool is the limit, not KV.

## Keeping prefixes on disk (`--prefix-disk-cache`)

> **Experimental**: measured on an RTX 2060 with Ornith and on two RTX 3060s with Qwen3.8-Flash-Next
> (`--pp-size 2`), and the flag may still change.
>
> Two RTX 3060s, Flash-Next with `--spec-mtp 5`, an 8.8k-token prompt, time to first token: 21.05 s
> prefilled, 1.89 s from the in-memory cache, **2.23 s from disk after a restart** (both ranks
> restored 8,768 tokens in the same step, each reading its own 161-179 MiB in 0.17-0.18 s). With one
> rank's file deleted, both ranks prefilled it (20.93 s) and nothing stalled.
>
> An 8,097-token prompt, time to first token: 13.0-14.1 s prefilled, 0.99 s from the in-memory
> cache, **1.24-1.28 s from disk** after the cache had let it go (about 0.2 s reading, 0.1 s copying
> to the GPU), 1.7-2.1 s from disk right after a restart. A marker planted at the start of the
> prompt was answered correctly on all three paths, the reply text matched the in-memory hit's, and
> `--spec-mtp` accepted as many draft tokens per step either way. A read that takes longer than
> the prefill it saves (about 1 s plus the prefix at 2000 tokens a second) is abandoned and the
> prompt is prefilled.

Raising the ratio buys conversations with VRAM. The other way is to keep what the cache lets go
of on disk:

```bash
ft serve ... --prefix-disk-cache ~/.cache/freetoken-prefix --prefix-disk-cache-size 32G
```

- **What is written.** Every boundary the in-memory cache can resume from when the server goes
  idle -- a snapshot of the GDN state plus the KV pages in front of it -- for prefixes of 1024
  tokens or more. A long prompt leaves one every prefill chunk, but with the default
  `--linear-state-cache-ratio` the snapshot pool is small enough that a long prompt's earliest
  boundaries are already gone by the time it finishes; the later ones, which a follow-up question
  on the same document resumes from, are what reaches the disk.
- **When.** While the server is idle (after a reply, before the next request), so it never
  slows a request down that is already running. Entries are copied out of VRAM on the scheduler
  and written by a background thread, with each file fsynced and renamed into place: a crash
  leaves a temporary file that the next start removes, never a half-written entry.
- **When it is read.** When a prompt arrives whose start is on disk at least a chunk-ish deeper
  than the in-memory cache has it. The request waits while the file is read (the requests behind
  it wait too), then it is admitted as an ordinary cache hit: `#cached-token` shows the restored
  length, and the log says `resumes at N tokens from disk`.
- **What is never read.** An entry written by a different model or weights (config.json and every
  file's size and mtime in the model directory), a different `--dtype`, `--kv-cache-dtype`,
  `--dense-quant`, `--quant-backend`, page size, `--spec-mtp` layout, code version (including
  uncommitted changes to a checkout), or cache layout. Those live in a separate sub-directory
  and only count toward the size cap. Each entry also stores its token ids and a checksum per
  tensor, and both are checked on every read.
- **Size.** One entry is the GDN snapshot (about 62 MiB on Ornith, whose 30 GDN layers hold a
  128 × 128 state per head) plus the KV of the prefix; each entry is self-contained, so the
  boundaries of one long prompt repeat its head. `--prefix-disk-cache-size` caps the whole
  directory, least recently used out first.
- **Page cache.** Entry files are dropped from the page cache after they are written or read, so
  they do not push out a `--moe-bank-ram` bank's cold half.
- **Pipelines (`--pp-size`).** Each rank writes and reads only its own layers, in its own
  sub-directory (`pp<rank>of<ranks>`) with an equal share of `--prefix-disk-cache-size`. Every rank
  runs its own copy of the scheduler, and a disk read finishes at a different moment on each, so
  rank 0 alone decides which entry to read, when it has arrived and when to give up, and sends
  each decision to the other ranks with the next step's requests. Before restoring, the ranks
  check together that every one of them has read its file and uploaded its part; if one has not
  (its file was evicted, say), none restores and the prompt is prefilled as usual. Rank 0 decides
  one step before the others act on it, so a request that is looked up waits one scheduler step
  longer than on one GPU.

Not supported yet, and refused at startup: `--tp-size` > 1, models other than the
hybrid GDN ones, and `--cache-type naive`. Image prompts are never cached, on disk or not.
