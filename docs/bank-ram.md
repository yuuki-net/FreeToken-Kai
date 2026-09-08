# Half the RAM: expert banks on disk (`--moe-bank-ram`)

A 125B MoE with 4-bit experts needs about 63 GiB of host RAM for the expert banks alone. That
is what makes 128 GB the practical floor for serving one on consumer GPUs, and RAM at today's
prices is the expensive half of such a machine.

`--moe-bank-ram` puts the banks in a file and keeps the frequently routed part of them in RAM.
On two RTX 3060 12 GB with 64 GB of RAM, Qwen3.8-Flash-Next decodes at 14-15 tok/s against
18-20 with the banks fully resident in 128 GB. On **one** RTX 3060 12 GB with 64 GB,
gpt-oss-120b decodes at 15 tok/s -- that model puts 98% of its parameters in experts, so
almost nothing has to live on the card and the layer split is not needed at all.

Read "64 GB" in this document as **a 128 GB host held down to 64 GB with a locked balloon**, on the
machine described in [kai.md](kai.md); no real 64 GB host has run this. That cuts in the direction
you may not expect: 128 GB is four DDR5 DIMMs at 4000 MT/s where two would run 4800, and the
CPU-side memory rate is what caps decode here, so an actual two-DIMM 64 GB machine has *faster*
memory than the one these numbers came from.

What it does not buy you: it does not make prefill faster (it makes it slower), it does not
reduce disk space (it adds a copy of the banks), and it is not a way to run a model your GPUs
could not otherwise hold. The GPU-side requirements are unchanged.

## How it works

The obvious design -- a small resident bank plus a cold tier read on demand -- runs into
`OffloadMoeCache`: its banks are `[num_experts, ...]` by contract and prefill streams a whole
layer by that count. Shrinking the bank means changing the copy machinery and the prefill
path, the hot paths, for every quantisation format.

So the bank keeps its shape and the file does the work. Each rank writes its expert banks once
to `~/.cache/freetoken/bankmap/<model>/bank.rankNofM.ftmb`, rows reordered so the frequently
routed experts come first, then maps the file and hands the cache `[num_experts, ...]` tensor
views over it. The cache never sees anything unusual. What changes is which rows are
guaranteed to be in RAM:

- rows `[0, hot)` are `mlock`ed, so the prefill sweep -- which touches every expert of every
  layer on each chunk -- cannot evict them, and they are `cudaHostRegister`ed so the PCIe
  fetch path can still DMA out of them;
- rows `[hot, num_experts)` are ordinary file-backed pages. They fault in when routing reaches
  them and the kernel drops them again under pressure, with no writeback, because the mapping
  is read-only.

Routing ids are renumbered to match, in the layer, before anything reads them.

### The GPU never sees a row it cannot address

Only the resident prefix is registered, and a GPU fetch of a page that is not would be an
illegal access inside the decode graph. Both paths avoid it without touching the kernels:

- **decode**: a miss on a row past the prefix is handed expert 0 (rank 0 of the renumbering, so
  it is all but certainly a cache hit costing no fetch) before `ensure_experts_hybrid`, and
  overwritten with `-1` afterwards -- which is the CPU partial's own signal. The routing
  histogram reads the untouched raw ids.
- **prefill**: the overlap prefetch issues every bank's registered half on the copy stream,
  then bounces the remainder through the pinned staging buffer once the layer's GEMMs are
  enqueued. A plain `cudaMemcpy` reads unregistered host memory; only async DMA cannot.

## Running

### 1. Measure the routing

The placement is only as good as the histogram behind it. Run once with the graph disabled
(under a captured graph the scatter never sees real expert ids) and use the server normally:

```bash
ft serve --model-path /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2 --gpu 0,1 \
  --moe-backend hybrid --ple-backend disk --dense-quant fp8 \
  --disable-cuda-graph --moe-stats-out ~/moe-stats.json
```

`~/moe-stats.rank0.json` and `~/moe-stats.rank1.json` are written on shutdown.

**Routing is domain-dependent.** Measured out-of-sample on three sessions: a histogram taken
from prose predicts a prose session's routing far better than one taken from a coding session
does (cell-level correlation 0.56 against 0.12). Collect two or three sessions of different
kinds and pass them all; pooling is cheap insurance, not a decisive gain (1-2 points).

### 2. Serve

```bash
ft serve --model-path /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2 --gpu 0,1 \
  --moe-backend hybrid --ple-backend disk --dense-quant fp8 \
  --moe-bank-ram 48G --moe-bank-stats ~/moe-stats.rank0.json ~/moe-stats2.rank0.json
```

`--moe-bank-ram` is a **whole-host** cap, not per rank: two ranks on one machine each get half
of it. Leave headroom -- the rest of the process wants about 9 GiB, and what is left after
that is page cache the design leans on.

The first run writes the mapping (31.7 GiB per rank for Flash-Next); later runs reuse it
unless the placement changed. `--moe-bank-dir` moves it off `~/.cache`.

### 3. Set the device readahead

**This is worth more than everything above.** The kernel's readahead window applies to faults
on the mapping, and the default is sized for streaming files, not for expert rows:

| `read_ahead_kb` | Flash-Next, 64 GB | gpt-oss-120b, 64 GB |
|---|---|---|
| 8192 | 5.9 tok/s | 14.6 |
| 4096 | — | 15.5 |
| 2048 | 7.9 | **16.4** |
| 1024 | — | 14.2 |
| 512 | 11.1 | — |
| **256** | **15.0** | — |

A fault that reads 8 MiB to use a few hundred kilobytes spends the difference on the experts
*least* likely to be routed to next, because the non-resident rows are the tail of the
frequency order. The right value depends on the checkpoint's block geometry: the widest
expert-row block is 1600 kB for Flash-Next and 7.91 MiB for gpt-oss-120b, and the measured
optimum was a quarter to a sixth of that.

```bash
DEV=$(df --output=source ~/.cache/freetoken/bankmap | tail -1)
B=$(lsblk -no PKNAME "$DEV" | head -1); B=${B:-$(basename "$DEV")}
echo 256 > /sys/block/$B/queue/read_ahead_kb
```

The startup line names the value and the file it came from, and complains when the window is
wider than the widest block. Being under that threshold does not mean it is optimal -- measure.

## Measured

Two RTX 3060 12 GB, a Core i5-12600KF, 128 GB of DDR5-4000, a Gen4 NVMe on the CPU-direct M.2,
WSL2 (full specs in kai.md). RAM restricted with a locked balloon so a 110 GiB
host behaves like a smaller one.

Every gpt-oss-120b row was run at 32k of context (`--max-seq-len-override 32768
--kv-reserve-tokens 32768`, which page rounding turns into 32889 tokens, K + V = 1.37 GiB); the
Flash-Next rows at 128k (`131072`, K + V = 1.55 GiB). The decode figures are therefore not a
like-for-like comparison between the two models. gpt-oss-120b was not run at 128k: half of its 36
layers are full attention at 2048 B per token per layer, so 36.9 kB/token, and 128k of KV would
want 4.7-5.5 GiB against the 1.62 GiB free after initialisation. Whether it can be made to fit is
untested.

| | RAM | Decode | Prefill, 4096-token chunk |
|---|---|---|---|
| Qwen3.8-Flash-Next, banks pinned | 128 GB | 18-20 tok/s | 6 s |
| Qwen3.8-Flash-Next, `--moe-bank-ram 48G` | 64 GB | 14-15 | 10-45 s |
| gpt-oss-120b, one GPU, `--moe-bank-ram 48G` | 64 GB | 15.2 | 12-52 s |
| gpt-oss-120b, one GPU, `--moe-bank-ram 24G` | 32 GB | 4.5 | 12-52 s |

**Measure over thousands of tokens.** The same Flash-Next configuration reads 11.5 tok/s over
110 tokens and 15.0 over 2360. The non-resident rows a session keeps routing to accumulate in
whatever page cache is left and stop being disk reads, so a short run reports the warm-up.

At 77% residency the static placement covers 87.5% of routes out-of-sample -- ten points
better than an arbitrary slice. The steady state behaves better than that, around 3% of routes
actually reaching the disk, because of the page cache above.

## What this looked like while it was wrong

Two of the three biggest wins here were not tuning. They were the design failing to do what it said
it did, in ways no model predicted, and the reason to write them down is that both are easy to
inherit and neither announces itself. A slow run does not tell you which of these it is.

**The VRAM expert cache was being skipped entirely (worth 40%).** `_decode_routed` in
`layers/moe.py` checks `is_cpu_layer` *before* the hybrid branch, so a layer listed as a CPU layer
goes straight to the CPU executor and never consults the VRAM cache. Declaring the mapped banks
locked, and listing every layer in `cpu_layer_ids` for consistency, quietly routed all of them past
a 1,180-slot cache that was measurably being hit. Nothing logs this: throughput is simply lower than
it should be. If you change which layers are CPU layers, check that the cache hit counters still
move.

**Readahead was the whole story (worth 2.5x), and the first explanation of it was also wrong.**
The kernel default of `read_ahead_kb 8192` turns a random 512 KB expert row into an 8 MB read.
Setting it to 256 was worth more than every code change in this document combined. But the first
diagnosis of *why* was that the mapping lacked `MADV_SEQUENTIAL`/`MADV_WILLNEED`, and that a single
`madvise` would be worth 3-4x. Both claims were false; the madvise changes measured as noise, and
the startup log never showed `fault readahead` engaging. Only the block-device setting mattered.

**And the theory that sent us looking was wrong too.** The slow case was assumed to be page faults
against a file that had fallen out of RAM. `smaps` said otherwise: both ranks had `Rss == Size`
(33,269,760 kB, 144 VMAs), i.e. the entire mapping was resident, and it was still 8 tok/s. On a
110 GiB host the file never goes cold, so the disk-bound case this design was built for had not
actually been reproduced there at all — a 64 GB host is the only place to confirm it.

The estimate that drove the design ("77% resident covers 87.5% of routes, which is fast enough")
held up. The intermediate numbers along the way — +4 ms here, 31 ms there, "3-4x from madvise" —
were worth about a factor of two, and should not have been quoted as if they were measurements.

## Limits and caveats

- **Prefill is 2-7x slower.** Each chunk still streams every expert of every layer, and the
  quarter that is not registered goes through a pinned bounce buffer at about 1.7 GB/s. Three
  quarters of it overlaps the previous layer's GEMMs; the rest does not.
- **Disk space.** The mapping is a second copy of the banks: 63.4 GiB for Flash-Next, on top of
  the checkpoint and (for that model) 47.7 GiB of PLE.
- **A slow disk changes the answer, and so does which M.2 slot it is in.** Decode reads whole
  expert rows at random from several threads. Measured against a Gen4 NVMe on the CPU-direct M.2:
  a Gen3 NVMe (3.2 GB/s) multiplies the disk part by about 1.6, and a SATA SSD (0.5 GB/s) adds
  roughly 306 ms per step even at 64 GB, which is not a configuration worth running. A *chipset*
  M.2 shares DMI with a chipset x4 GPU slot, so with `--pp-size 2` the bank reads and rank 1's
  residual stream fight over one link — put the model on the CPU-direct M.2. Measure yours the way
  the decode path uses it before assuming anything.
- **Halving RAM again is expensive.** gpt-oss-120b at 32 GB runs, at a third of its 64 GB
  speed. 64 GB is where the design pays.
- **`--moe-bank-ram` disables the pin-budget CPU-layer split** (`--moe-cpu-layers auto`), which
  answers the same question differently and would take the VRAM expert cache away from the
  layers it locks. An explicit `--moe-cpu-layers` still applies.
