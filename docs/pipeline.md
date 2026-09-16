# Two consumer GPUs: layer-split serving (`--pp-size`)

FreeToken Kai can run one model across two (or more) GPUs in the same machine by splitting the
decoder **by layers**: one process per GPU, each owning a contiguous block of layers. It is
meant for the hardware people actually have -- two mid-range cards on a consumer board,
possibly of different sizes, with the second card in a chipset x4 slot -- and it needs neither
NCCL nor GPU-to-GPU peer access.

What it buys you:

- **A model whose dense (non-expert) weights do not fit one card.** Qwen3.8-Flash-Next carries
  10.4 GB of bf16 attention / GDN / embedding / head weights; two 12 GB cards hold it with room
  for 128k of context. That was the original reason for this code.
- **Twice the expert cache.** Every rank keeps an expert cache on its own card, so a 35B-A3B
  MoE on two 12 GB cards has about 4,000 expert slots per card instead of 3,000 in total. The
  decode speed gain is whatever the higher hit rate buys on your machine (see Measured: on a
  fast x16 slot it bought nothing for Ornith and +30-40% for gpt-oss-120b); the ranks run one
  after the other, so nothing is computed in parallel.

What it does not buy you: it does not lower the host RAM requirement (the expert banks still
live in RAM, split between the ranks by layer), and it does not add compute.

Models: Qwen3.8-Flash-Next (`qwen4_exp`), the Qwen3.5-MoE family (Qwen3.6-35B-A3B,
Ornith-1.5-35B-A3B; `qwen3_5_moe`) and gpt-oss (`gpt_oss`). Others raise
`NotImplementedError` at startup.

## How it works

- `ft serve --pp-size 2 --gpu 0,1` starts one scheduler process per GPU, reusing the
  multi-rank machinery FreeToken has for tensor parallelism (every rank processes the same
  request stream in lockstep). `--gpu` lists the cards in rank order: the first owns the
  embedding, the last owns the final norm and the lm_head.
- Each forward, a rank hands the residual stream (`[tokens, hidden]` in the model dtype; 4 KB
  per token for a 2048-wide model, 20 KB for Flash-Next's four streams) to the next rank over
  gloo point-to-point, and the last rank sends the sampled tokens back to the first. This
  crosses the PCIe bus through host memory, which costs well under a millisecond per decode
  step.
- Each rank allocates only its own layers' KV pages, GDN state slots, expert cache and pinned
  host banks. The cache budget (`--memory-ratio`) is solved per rank, so cards of different
  sizes get different expert-cache sizes automatically. The KV page count is the exception:
  every rank's scheduler reads it when it admits requests and evicts cached prefixes, so all
  ranks take the smallest count any rank solved (`--pp-size: KV pool 2050 -> 2048 pages` on the
  two 3060s, where the expert fill had left rank 0 two pages more).
- CUDA graphs: the first rank's decode graph outputs the residual stream instead of logits;
  the others capture their graphs against a static input buffer the received stream is
  copied into. `--spec-mtp`'s verify-window graphs work the same way.
- The prefill chunk (`--prefill-chunk-budget`) is one number for the whole pipeline: the ranks
  agree on the tightest measurement at startup and keep it, because the residual stream they
  hand each other is sized from it. The `--spec-mtp` graphs are captured after that measurement
  and keep their pools, so the ranks check the chunk once more when the boot is done and narrow
  it if the tightest rank no longer has the room (2816 -> 2560 on the 3060s). See
  [prefill-chunk.md](prefill-chunk.md#two-gpus).
- Prefill chunks overlap across the ranks: for a chunk whose sampled token nobody reads (every
  chunk but the last), the first rank hands the stream on and starts the next chunk without
  waiting, so a long prompt costs about the slower rank's time per chunk, not the sum.
- The ranks stay in step over a one-int note the first rank sends every scheduler iteration
  ("how many client messages follow this step"), point-to-point rather than as a collective so
  that the first rank can keep the run-ahead the previous bullet depends on. Its sends are
  retired by waiting on the oldest once more than 64 are outstanding, NOT by polling: gloo marks
  a send completed only inside `wait()`, so `is_completed()` is False even for a message the
  peer took long ago, and a backlog filtered that way never shrinks. Every iteration walks that
  backlog, so an unbounded one turns into a per-step cost proportional to the steps served.
- The client messages themselves go from the first rank to the others over a ZeroMQ PUB/SUB
  socket, which drops whatever arrives before the receiving side's subscription has registered
  -- a few hundred milliseconds after the socket exists. Before reporting ready, the ranks
  exchange hellos over it until every rank has received one (`rank relay: 1 rank(s) subscribed
  after N hello(s)`), so a request sent the instant the server is ready is not lost. If a rank
  never hears one within a minute, every rank stops with an error rather than one of them
  waiting forever.
- Every blocking send and receive between the ranks is watched. A wait longer than
  `FREETOKEN_RANK_WAIT_WARN_SECONDS` (60) logs which rank is waiting for what, again every
  minute while it lasts, and how long it took once it ends. Nothing times out. A wait of tens of
  seconds can be a long prefill chunk on the other rank; one that keeps growing means that rank
  has stopped, and its own log (or `py-spy dump`) says where.
- At startup the ranks load different halves of the model and reach the points where they
  agree on the KV page count and the prefill chunk at different times. Each of those points
  starts with a barrier that waits up to `FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS` (3600) for the
  other ranks, and a wait of 10 s or more is logged (`waited N s for the other ranks before
  ...`). Before, the first rank to get there gave up after the group's 60 s: on two RTX 3060s
  with `--moe-bank-ram` and a page cache emptied by a WSL restart, rank 1 was still reading its
  experts and the start failed with `Timed out waiting 60000ms for recv operation`.

## Running

Flash-Next on two 12 GB cards with 128k of context, fp8 dense weights and the MTP head:

```bash
ft serve --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pp-size 2 --gpu 0,1 \
  --moe-strategy hybrid --ple-backend disk --dense-quant fp8 --spec-mtp 5 \
  --max-running-req 1 --memory-ratio 0.85 --max-prefill-length 4096 \
  --max-seq-len-override 131072 --kv-reserve-tokens 131072 --moe-cpu-threads 7
```

Ornith-1.5-35B-A3B on the same two cards (fewer layers on the x4 card, experts fetched over
PCIe only):

```bash
ft serve --model /path/to/Ornith-1.5-35B-A3B-NVFP4 --pp-size 2 --gpu 0,1 --pp-layers 25 \
  --moe-strategy offload --max-running-req 1 --memory-ratio 0.85 \
  --max-seq-len-override 65536 --kv-reserve-tokens 65536
```

gpt-oss-120b on the same two cards (experts mostly off-card, so the CPU helps):

```bash
ft serve --model /path/to/gpt-oss-120b --pp-size 2 --gpu 0,1 --pp-layers 26 \
  --moe-strategy hybrid --max-running-req 1 --memory-ratio 0.85 \
  --max-seq-len-override 65536 --kv-reserve-tokens 65536 --moe-cpu-threads 7
```

- `--gpu` takes indices or UUIDs from `nvidia-smi -L`, in rank order.
- `--pp-layers` sets the split points (N-1 numbers): `--pp-layers 26` gives rank 0 layers 0-25
  and rank 1 the rest. Use it for cards of different sizes (more layers on the bigger card) and
  when the even split leaves a rank without one of the model's attention kinds. Every rank
  needs at least one layer of each kind (full attention, GDN, sliding window) because the KV
  pool family is chosen from the layer set; a bad split fails at startup with a `ValueError`
  that says so.
- `--dense-quant fp8` quantizes the checkpoint's bf16 dense projections (attention, GDN,
  shared expert, lm_head, embedding) to per-row fp8 at load, W8A16. It halves their VRAM and
  per-token read traffic; the freed VRAM goes to the expert cache. Wired for Flash-Next.
- `--moe-strategy hybrid` works with two ranks, but the two CPU executors share the cores and
  each rank's forward pays the executor's per-layer synchronization; when a rank's experts
  mostly fit its cache, `--moe-strategy offload` is faster for that model (Ornith: 40-44 against
  25-30 tok/s). Fewer threads did not help (`--moe-cpu-threads 3`: 20-24 tok/s).
- Logs and progress bars come from rank 0; rank 1 logs only warnings and errors. Its lines
  carry `rank=1`.
- The desktop / `ft ctl` runtime cache rebuild is not available with more than one rank.

## Measured

RTX 3060 12 GB x2 (GPU 0 on the CPU's PCIe 4.0 x16, GPU 1 on a chipset x4 slot at ~6 GB/s),
a Core i5-12600KF, 128 GB of DDR5-4000, Windows + WSL2. Decode at batch 1 behind a ~2,600-token
prompt. Full specs, and why the second slot is x4, are in kai.md.

| Model | 1 GPU | 2 GPUs | Notes |
|---|---|---|---|
| Qwen3.8-Flash-Next-NVFP4 (`--dense-quant fp8`) | does not fit | 18-20 tok/s; 13-27 with `--spec-mtp 5` | 128k of context, image input; the reason the code exists |
| Ornith-1.5-35B-A3B-NVFP4 | 41-46 tok/s (hybrid, 2,947 slots) | 40-44 tok/s (`--pp-layers 25 --moe-strategy offload`, 3,833 slots per card); 25-30 with the even split and hybrid | equal at best |
| gpt-oss-120b (MXFP4) | 9-12 tok/s (202 slots; 9 layers decode on the CPU because 57 GB of banks exceed the pin budget) | 12-17 tok/s (`--pp-layers 26 --moe-strategy hybrid`, 394 slots per card, all banks pinned) | +30-40% |

### Prefill

Decode is the number people quote, but a long prompt spends most of its wall clock in prefill,
and this is where the overlap described above pays. Qwen3.8-Flash-Next, `--max-prefill-length
4096`, timed from the intervals between `Prefill batch` lines in the log:

| | Ranks in sequence | Ranks overlapped |
|---|---|---|
| One 4,096-token chunk | ~12 s | **6 s** |
| 16,159 tokens (4 chunks) | ~48 s | **~29 s** |
| 4,096 + 1,164-1,946 tokens (2 chunks) | ~17 s | 11-12 s |
| 128k of input (32 chunks) | ~6 min | — |

What the same model does now, on a 25k-token prompt with `--prefill-mixer-pieces 2
--max-prefill-length 8192` and the expert-cache defaults of 2026-09:

| | chunk | prefill |
|---|---|---|
| `--prefill-chunk-budget` 0.55 (default) | 3,584 tokens | 510 tok/s |
| `--prefill-chunk-budget` 0.75 | 5,120 tokens | **702 tok/s** |

The chunk is what decides it. An offloaded MoE copies a layer's whole expert bank to the GPU
once per chunk, whatever the chunk holds, and the second rank's slot moves 33 GiB of
Flash-Next banks in 5.6 s: at 3,584 tokens that transfer outlasts the rank's compute for the
chunk and shows up as waiting, at 5,120 it hides behind it. On a card that only serves, this
is the cheapest prefill win available — see [prefill-chunk.md](prefill-chunk.md#choosing-the-budget)
for what the budget trades away.

The last chunk cannot overlap — its sampled token is the one that starts decoding — so a
short prompt sees less of this than a long one. What a user actually feels on a follow-up turn,
where the prefix is already cached and only the new message is prefilled, is 2.4-4.5 s (9 s
before the CPU short-prefill path removed the per-turn expert streaming).

What the per-rank timer (`FT_STEP_PROFILE=1`) shows for Ornith with the even split: the
hand-off costs 0.6 ms, and the step is the sum of the two forwards (11-13 ms on rank 0 and
16-19 ms on rank 1 for 20 layers each), against ~22 ms for all 40 layers on one card. On this
machine the expert traffic was never the bottleneck -- the x16 slot moves a miss in a fraction
of a millisecond -- so a second cache buys nothing, and the x4 rank pays for every miss it
does have. Hence the recommendations: put fewer layers on the slower slot (`--pp-layers 25`
leaves rank 1 with 15 layers whose experts all fit its cache), and use `--moe-strategy offload`
for a model whose experts mostly fit, `hybrid` for one whose experts mostly do not
(gpt-oss-120b). A machine with a slower bus or fewer cores would see more from the second
cache than this one did.

### Two knobs for a slow second rank

Both of these came out of timing a Flash-Next prefill part by part on this machine, and both
are **off by default because they did not make it faster here**. They are documented because
what they address is real, and a machine whose second rank is slower than this one — a narrower
link, fewer lanes, a slower disk behind `--moe-bank-ram` — may see what this one did not.

What the timing showed, at a 3,072-token chunk: the second rank waited on its expert banks for
0.69 s per 1k tokens (33 GiB per chunk over a PCIe x4 slot at 5.9 GiB/s, 5.6 s, against about
3 s of compute for the chunk), and the first rank waited on the second for 0.43 s per 1k.

`--pp-prefill-group N` (Qwen3.8-Flash-Next, `--max-running-req 1`) makes the second rank hold up
to N consecutive chunks of a prompt and then run them layer by layer: every chunk through a layer
before the next layer, so the layer's bank is copied once for all of them. The arithmetic is the
same as one chunk per forward — unit-tested bit for bit on Flash-Next's GDN and QSA layers, and
the prefix-cache stress suite passes with it on — and held chunks are run before anything that
could read or free their state (the next request, a snapshot restore, an abort). It did what it
was meant to: that rank's wait on its banks went from 0.27 to 0.08 s per 1k tokens at a
5,120-token chunk. The prompt was not faster for it (703 tok/s against 716 with the flag, inside
the spread between runs), because grouping adds a rendezvous every N chunks.

`--pp-send-ahead N` is the other half: the hand-off used to block until the peer posted its
receive, so the first rank never got more than one chunk ahead. At N it sends non-blocking from a
ring of N pinned staging buffers and blocks only when all N are in flight. It removes the cost of
grouping (618 tok/s with `--pp-prefill-group 2` alone, 716 with both), and on its own it changed
nothing: at a 5,120-token chunk the two ranks already take the same 1.42 s per 1k tokens each, so
there was no idling left for a deeper lookahead to fill.

If you try them, measure with `--prefill-profile` (and `--pp-prefill-group` needs
`--max-running-req 1`, which is the setting a single-user server runs anyway).

## Limits and caveats

- The ranks run in sequence: two GPUs are not faster than one GPU that could hold everything.
  The gains are the cache size and, for Flash-Next, that it runs at all.
- Mixed cards (a 16 GB and a 12 GB, say) are supported in principle through `--pp-layers`; a
  Turing card next to an Ampere card should also work, since the Turing paths are chosen per
  process. Neither has been tried by the fork's maintainer.
- Every measurement above was taken under WSL2, so that is the configuration that is known to
  work: gloo over loopback between two ranks, no NCCL, no peer access. Native Linux with two GPUs
  has not been tried by the fork's maintainer; it should behave at least as well, but report what
  you see. Note that the pin budget the offload backend works against is a WSL limit, so a native
  Linux host may place more of the expert banks than these numbers suggest.
- `ft checkpoint`-converted (FTW) checkpoints load per layer window but were not run this way.
- Builds of this fork published before 2026-09-09 have the unbounded backlog described under
  "How it works": with `--pp-size` (or any multi-rank run) decoding slowed steadily for as long
  as the server stayed up -- 17.9 -> 2.3 tok/s over one 12.5-hour session on the two 3060s, step
  time 56 ms -> 439 ms. It tracked the number of decode steps served, not the context length or
  the KV occupancy, so a fresh short request was just as slow and only a restart helped; prefill
  was unaffected, because the same overhead lands once per 4096-token chunk. Single-GPU runs
  never took that path. If you are on such a build, update. The fix is unit-tested, and decode
  held flat across 175,000 decode steps of live serving over 4.8 hours -- 19.2 tok/s at the point
  where the old build had fallen to 2.3, and a residual drift of 0.05 us per step against the old
  build's 2.5, which is to say none that this measurement can separate from load.
- Builds of this fork published before 2026-09-13 have neither the relay handshake nor the
  shared KV page count described under "How it works": a request sent right at readiness could
  be dropped between the ranks, leaving both waiting on each other with nothing in the log, and
  the ranks' schedulers could disagree about the KV pool once the prefix cache filled up.
  Neither was observed failing on the 3060s; both were found by reading the code. Update.

When reporting a problem, include `nvidia-smi -L`, the first 60 lines of the server log (both
ranks) and the exact `--pp-size` / `--pp-layers` / `--gpu` values.
