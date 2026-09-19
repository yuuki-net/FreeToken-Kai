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

What it does not buy you: it does not make prefill faster (it makes it slower), and it is not a
way to run a model your GPUs could not otherwise hold. The GPU-side requirements are unchanged.
On its own it also costs disk -- the bank file is a second copy of the experts -- until
[`ft bank pack`](#5-optional-drop-the-second-copy-ft-bank-pack) takes them out of the checkpoint.

## How it works

The obvious design -- a small resident bank plus a cold tier read on demand -- runs into
`OffloadMoeCache`: its banks are `[num_experts, ...]` by contract and prefill streams a whole
layer by that count. Shrinking the bank means changing the copy machinery and the prefill
path, the hot paths, for every quantisation format.

So the bank keeps its shape and the file does the work. The expert banks are written once to
`~/.cache/freetoken/bankmap/<model>/bank.ftmb`, one file for every MoE layer, rows reordered so
the frequently routed experts come first. Each rank maps the file and hands the cache
`[num_experts, ...]` tensor views of its own layers. The cache never sees anything unusual. What
changes is which rows are guaranteed to be in RAM:

- rows `[0, hot)` are `mlock`ed, so the prefill sweep -- which touches every expert of every
  layer on each chunk -- cannot evict them, and they are `cudaHostRegister`ed so the PCIe
  fetch path can still DMA out of them;
- rows `[hot, num_experts)` are ordinary file-backed pages. They fault in when routing reaches
  them and the kernel drops them again under pressure, with no writeback, because the mapping
  is read-only.

Routing ids are renumbered to match, in the layer, before anything reads them.

### What the file holds, and what a restart reads

The file depends on the checkpoint, the expert kernel and each layer's row order -- and on
nothing else about the run:

- **The resident count is not in it.** `hot` is solved from `--moe-bank-ram` at every start, so a
  different budget rewrites nothing.
- **The layer split is not in it.** Every MoE layer is in the one file, by its index in the whole
  model; a rank of `--pp-size 2` writes and maps only its own. Changing `--pp-size` or
  `--pp-layers` rewrites nothing.
- **The row order is per layer and can change in place.** A start with a different
  `--moe-bank-stats` reorders the layers whose order changed, one layer at a time, without
  reading the checkpoint. A layer's old bytes go to a journal next to the file first
  (`bank.ftmb.journal.L<n>`, one layer's worth of disk at a time), and the next start finishes or
  rolls back a reorder that was interrupted. Every reordered byte is therefore written twice:
  measured on an RTX 2060 with Ornith, 19 layers (8.0 GiB) took 123 s and 16 GiB of writes. A start *without* `--moe-bank-stats` keeps the order that
  is already in the file.
- **A layer exists once it is committed**: blocks written and synced, then a manifest inside the
  file names the layer with its order and a SHA-256 per block. A start killed while writing
  leaves the layer out rather than a block of zeros that looks reusable.

When every one of a rank's layers is committed, that rank **does not read the checkpoint's expert
tensors at all** -- the startup log says `holds layers 0-23; the checkpoint's expert tensors are
not read`. Before this, every start read and packed all of them (31.7 GiB per rank for
Flash-Next) only to throw them away when the file already existed.

The file is tied to the checkpoint by a fingerprint of the expert tensors' names, dtypes and
shapes, and by the size and modification time of the shards that hold them: a checkpoint
downloaded again, or copied, gets its bank written again rather than served the old experts.
The kernel (`nvfp4 / triton`, `mxfp4 / triton_gptoss`) is part of the file too; switching to one
with a different bank layout also starts it over. `ft bank info --model-path <model>` prints what
a file holds.

Files written by builds before this one (`bank.rank0of2.ftmb`, ...) are not read. The first start
writes `bank.ftmb` and warns with the old files' names and size; delete them.

This layout has run on an RTX 2060 (Ornith: first start, a restart that read no expert tensor,
a new budget and a new histogram, all serving the same text at temperature 0 as the per-rank files)
and on two RTX 3060s (Flash-Next, `--pp-size 2`: both ranks wrote their halves of one 63.4 GiB file
at once, and a later start with `--pp-layers 25` and a different budget wrote nothing). Most of the
measurements further down were taken with the per-rank files of the earlier builds.

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

### Some hosts will not register a read-only file mapping

Registering one needs `cudaHostRegisterReadOnly`, which needs
`cudaDevAttrHostRegisterReadOnlySupported`. That attribute reads 1 under WSL2 and 0 on an RTX
3060 pair on native Ubuntu 24.04 with the 615.71.09 open kernel module -- the same GA106
silicon, so this is the platform rather than the card. Where it reads 0 the read-only form
cannot be registered at all, because plain `flags=0` asks for read-write pinning and a
read-only mapping cannot give it.

What such a host *will* register is ordinary anonymous memory. So the resident rows are put
into anonymous memory instead: the file is mapped `MAP_PRIVATE` and writable, and each
resident row is written to once -- a byte per page, put back exactly as it was read -- which
copies it out of the page cache into a private page that `flags=0` accepts. The page cache
copy is then handed back with `posix_fadvise(POSIX_FADV_DONTNEED)`, one block at a time, so
the two copies never coexist for more than one block. The non-resident rows stay file-backed
and evictable in either form, which is what makes 31.7 GiB of banks fit in 24 GiB of RAM.

Which form is used is decided at startup by asking the device, with one page of the bank file
itself -- not by reading the attribute, so that a host which advertises the flag and still
refuses this particular file lands on the form that works. The startup line says which:

```
--moe-bank-ram: mapped 31.7 GiB, 24.0 GiB locked resident, 24.0 GiB registered for PCIe, shared page cache, ...
--moe-bank-ram: mapped 31.7 GiB, 0.1 GiB locked resident, 24.0 GiB registered for PCIe, private pages, ...
```

`FREETOKEN_BANK_MAP` overrides the choice: `shared` for the read-only form, `private` for the
copied one, `auto` (the default) to ask. The private form costs the copy -- 8 s against 4 s to
settle a 6 GiB resident half from warm page cache, measured on a 2060 -- and the same amount
of RAM, but a different kind of it: anonymous pages rather than page cache, which the next
start cannot reuse. Decode speed was the same either way within run-to-run variation on that
machine.

If neither form registers, that is still a supported state rather than a failure. The startup
line says `0.0 GiB registered for PCIe`, every decode miss goes to the CPU executor, and the
VRAM expert cache is left unused -- the log says that too. It serves, and it is slower,
because that cache is where the hit rate lived. `FREETOKEN_BANK_REGISTER=none` asks for the
same state deliberately.

The same is true if only *some* of the resident blocks register, which is what a pinning limit
reached partway looks like. `prefix_pinned_rows` is one number for every layer, so a bank that
is registered in part cannot be described to the cache at all; it is treated as none, with a
warning naming how many blocks made it.

## The memlock limit

`--moe-bank-ram 48G` across two ranks asks each process to `mlock` 24 GiB, and
`RLIMIT_MEMLOCK` is per process. systemd's default is `MAX(64M, RAM/8)` -- 8 GiB on a 64 GB
host -- so the resident half comes out at a third of what was asked for and the remainder is
evictable page cache that goes back to disk under pressure. The startup line reports what was
actually locked, and warns when it fell short of the budget.

This matters on the shared form, where `mlock` is what holds the resident rows down. On the
private form they are already copies, and registering them pins them where `mlock` could not
reach -- a low `ulimit -l` there shows as `0.1 GiB locked resident` next to a full
`24.0 GiB registered for PCIe`, and draws no warning, because nothing is actually evictable.
The warning appears only when the registration did not cover them either.

Check it as the user that runs the server. The value is in KB; 25165824 is 24 GiB:

```bash
ulimit -l
```

Raising it needs root when the hard limit is low too, which is the systemd case. Allow a
little over one rank's half:

```bash
printf '%s soft memlock 27262976\n%s hard memlock 27262976\n' "$USER" "$USER" \
  | sudo tee /etc/security/limits.d/90-freetoken.conf
```

PAM applies that at login, so log out and back in; `su` does not pick it up. For a single run
without the re-login, `sudo prlimit --memlock=27917287424 --pid $$` raises the current shell
instead and the server inherits it.

## Running

### 0. Check the host first (`ft doctor disk`)

```bash
ft doctor disk --model /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2
```

It answers, without a GPU or root, the questions the rest of this section was learnt from: where
the bank file is and which layers it already holds, what filesystem and device it is on (and
whether a WSL2 `/mnt/c`, a network share, a USB/SATA disk or a chipset M.2 shared with a GPU is in
the way), whether `read_ahead_kb` suits
this model, how much RAM `--moe-bank-ram auto` would take, how fast the disk reads expert rows
the way decode does, and roughly what each RAM cap costs per token. The last part is an estimate
with its assumptions printed beside it, not a measurement; see [cli.md](cli.md#ft-doctor-disk).

### 1. Measure the routing

The placement is only as good as the histogram behind it. Run once with the graph disabled
(under a captured graph the scatter never sees real expert ids) and use the server normally:

```bash
ft serve --model-path /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2 --gpu 0,1 \
  --moe-strategy hybrid --ple-backend disk --dense-quant fp8 \
  --disable-cuda-graph --moe-stats-out ~/moe-stats.json
```

`~/moe-stats.rank0.json` and `~/moe-stats.rank1.json` are rewritten each time the server goes idle
(after a request finishes), so they hold the whole session however the server is stopped. Pass **every
rank's** file to `--moe-bank-stats`: each holds only its own rank's layers.

**Routing is domain-dependent.** Measured out-of-sample on three sessions: a histogram taken
from prose predicts a prose session's routing far better than one taken from a coding session
does (cell-level correlation 0.56 against 0.12). Collect two or three sessions of different
kinds and pass them all; pooling is cheap insurance, not a decisive gain (1-2 points).

### 2. Serve

```bash
ft serve --model-path /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2 --gpu 0,1 \
  --moe-strategy hybrid --ple-backend disk --dense-quant fp8 \
  --moe-bank-ram 48G --moe-bank-stats ~/moe-stats.rank*.json ~/moe-stats2.rank*.json
```

`--moe-bank-ram` is a **whole-host** cap, not per rank: two ranks on one machine each get half
of it. Leave headroom -- the rest of the process wants about 9 GiB, and what is left after
that is page cache the design leans on.

`--moe-bank-ram auto` does that arithmetic at startup, once, before the ranks start:

```
--moe-bank-ram auto: 47.9 GiB for the banks across 2 ranks = MemAvailable 60.0 GiB - 9.0 GiB for the rest of the server (4.5 per rank) - 3.1 GiB left as page cache for the non-resident rows; pass a size to override
```

The 4.5 GiB per rank is Flash-Next's measured 9 GiB halved; the margin is 5% of MemTotal (at
least 2 GiB), which is what the measured 48G configuration left on a 64 GB host. It reads
MemAvailable, so whatever else is running at startup is left alone -- and a server started while
something large is running gets a smaller cap than it would otherwise. A model that keeps more in
host RAM than Flash-Next (a host embedding, PLE tables in RAM) needs an explicit size.

Under WSL2 it also stays inside the CUDA pin budget, because the resident rows are registered with
CUDA and WSL2 caps page-locked memory near half of the VM's RAM: the budget is 40% of MemTotal
(`FREETOKEN_PIN_BUDGET_GB` overrides it, as it does for the rest of the server), less 2 GiB for
the other buffers the server pins. On an RTX 2060 host with 23.5 GiB, RAM alone chose 15.6 GiB of
Ornith's 16.9; 141 of 240 resident blocks registered and the boot died in a CUDA allocation. With
the cap it chose 7.4 GiB, started, and the server's own anonymous and shared memory came to
3.4 GiB -- inside the 4.5 GiB it budgets.

The first run reads the checkpoint's experts and writes the file (63.4 GiB for Flash-Next, each
rank its half). Later runs read no expert tensor from the checkpoint; a new `--moe-bank-stats`
reorders the file in place, and a new budget or layer split changes nothing on disk
([What the file holds](#what-the-file-holds-and-what-a-restart-reads)). Once a histogram has been
applied, later starts can leave `--moe-bank-stats` off. `--moe-bank-dir` moves the file off
`~/.cache`.

**Builds before this one placed rank 1 by rank 0's histograms.** The placement was keyed by each
rank's own layer index (0-23 on both ranks of Flash-Next) while the histograms were keyed by the
model's (rank 1's are 24-47), so rank 1 sorted its experts by the routing of layers 0-23. Every
`--pp-size 2` figure below was measured that way. Measured against the fix on the two RTX 3060s
(64 GB-equivalent, 2500 tokens, A B A B), it did not make decode faster: the client saw 21.0 tok/s
either way. The page cache holds most of what a poor placement sends to disk.

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

`ft doctor disk` and the startup line both name the file and the recommended value for the
model: the widest row block over sqrt(24), to the nearest power of two, which lands on both
measured optima (256 for Flash-Next, 2048 for gpt-oss-120b) and is a guess for anything else.
The startup line warns when the window is wider than the widest block. To set it by hand:

```bash
echo 256 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb
```

Do it before starting the server. The kernel copies the window into each file when it is opened
and faults read with that copy, so a server already running keeps the window its mapping was
opened with until it restarts.

Or let the server do it: `--moe-bank-readahead auto` writes the recommended value before each rank
opens its mapping of the bank file, and `--moe-bank-readahead 256` writes that one. Every rank maps
the same file, so every rank sets it for itself and only the first logs it. It needs
permission to write sysfs (root, or a container that mounts `/sys` writable); without it the
server logs the command above once and carries on. It is off by default because the window
belongs to the whole device -- every other file on that disk reads with it too -- and the
server does not put the old value back when it exits (the log line names both). It does not
survive a reboot either; a udev rule does, e.g.
`ACTION=="add|change", KERNEL=="nvme0n1", ATTR{queue/read_ahead_kb}="256"` in
`/etc/udev/rules.d/60-freetoken-readahead.rules`.

The window no longer matters to prefill: a chunk reads the non-resident rows from the file
with parallel reads instead of faulting them in (see Limits and caveats), so set it for decode.

Under WSL2 the window that counts is the virtual disk's (`/sys/block/sdX`), since that is the
device the ext4 filesystem sits on. Being under the warning threshold does not mean the value
is optimal -- measure.

### 4. Optional: read the cold rows back while idle (`--moe-bank-rewarm`)

The non-resident rows are ordinary page cache, and anything may take it: another process that
needs memory for a while, a large build, or WSL2 with `autoMemoryReclaim=gradual`, which hands
file cache back to Windows while the VM is quiet. The server keeps working. The next request
pays for every page it touches, one fault at a time -- and the request that meets it is usually
the next turn of a conversation, after a pause.

```bash
ft serve ... --moe-bank-ram 48G --moe-bank-rewarm 5
```

Once the scheduler has been idle for that many seconds, each rank checks how much of its own
bank's non-resident rows the page cache still holds (`mincore` on its mapping). Below 95% it
reads them back in file order, 64 MiB at a time, and stops at the next step boundary when a
request arrives; that request then pays only for what was not read yet. If a complete pass ends
with *less* cached than it started with, something is still pressing on memory and reading
again would only fight it, so the next check waits four times longer, up to ten minutes.

```
--moe-bank-rewarm: cold rows were 36% in page cache; walked 7.7 GiB of them in 11.6 s -> 92%
--moe-bank-rewarm: cold rows were 92% in page cache; walked 7.7 GiB of them in 8.0 s -> 100%
--moe-bank-rewarm: cold rows were 66% in page cache; walked 7.7 GiB of them in 9.1 s -> 60%
--moe-bank-rewarm: memory is still under pressure; next check in 20 s
```

Measured by taking the page cache away on purpose: allocate and touch anonymous memory up to
6 GiB short of `MemAvailable`, hold it 8 s, release it, wait, then send a prompt that the prefix
cache cannot answer. Time to first token:

| | warm | right after, without | right after, with `--moe-bank-rewarm 5` |
|---|---|---|---|
| RTX 2060, Ornith-1.5-35B-A3B, `--moe-bank-ram 6G` (16.9 GiB of banks), 45 s idle: short prompt | 1.0 s | **30.9 s** | **3.1 s** |
| same, ~2k-token prompt | 3.9 s | 13.5 s | 9.4 s |
| 2x RTX 3060, Qwen3.8-Flash-Next, `--pp-size 2 --moe-bank-ram 48G` (63.5 GiB), 90 s idle: short prompt | 3.0 s | **16.2 s** | **5.8 s** |
| same, ~2k-token prompt | 9.2 s | 12.6 s | 10.7 s |

Without the flag, the cold rows were still out of the cache after the idle wait (36% on the 2060,
76-80% of the whole bank on the 3060s); with it, back at 100% on every rank. The 3060 figures
come from the 128 GB host without the balloon described above.

**The short prompt is the one that suffers.** A long prompt's prefill streams whole layers, so
its faults are sequential and readahead covers them. A short one is prefilled and decoded on the
CPU executor, which touches expert rows out of order: 4 KiB random faults.

The flag does not bring the warm figures all the way back. Part of what is left on the long
prompts is the server's own memory: the pressure before those runs also pushed 2.4 GiB (2060)
and 4.7 GiB (3060) of it into swap.

So the same idle thread now also pages the rank's own swap back in, once the bank is back (or when
the bank never left) and there is at least 256 MiB of it -- only the pages the kernel reports as
swapped, never the address space around them. Each pass's line ends with the rank's swap, and a
second line says what was paged back. `FREETOKEN_REWARM_SWAP=0` turns that part off. Measured by
pushing only the server's anonymous memory to swap (`process_madvise(MADV_PAGEOUT)`), with the
bank's page cache left at 100%, then idling 90 s before a short prompt:

| | undisturbed idle | server memory in swap | swap, paged back while idle |
|---|---|---|---|
| RTX 2060, Ornith (2 GiB swapped) | +0.2 s | **+15.9 s**, decode 10 tok/s | +0.0 s |
| 2x RTX 3060, Flash-Next (4 GiB swapped) | +1.4 s | +1.7 s | +1.4 s |

How much swap costs depends on which pages went out; paging them back removed it on both. Two
things are left: the other processes of the server (the frontend and the tokenizer, about 1 GiB
between them) keep their swap, which a long prompt then waits on; and on the 3060s a short prompt
after 90 s idle is 1.4 s slower than one sent right away with nothing disturbed at all -- neither
disk nor swap, not explained yet.

It is off by default because it reads the disk while nothing is running: 7.7 GiB per rank took
1-12 s per pass on the Gen4 NVMe above. 5 s is the only delay that was measured. How fast
`autoMemoryReclaim` actually empties the cache on its own was not measured either -- the
pressure here was made by hand.

### 5. Optional: drop the second copy (`ft bank pack`)

Once the bank file holds every layer, the checkpoint's expert tensors are dead weight on the disk:
nothing reads them. `ft bank pack` writes a checkpoint without them and makes the bank file part
of it.

```bash
# the bank file must be complete: serve the original once with --moe-bank-ram (both ranks)
ft bank info --model-path /models/Qwen3.8-Flash-Next-NVFP4
ft bank pack --model-path /models/Qwen3.8-Flash-Next-NVFP4 --out /models/Qwen3.8-Flash-Next-banked --dry-run
ft bank pack --model-path /models/Qwen3.8-Flash-Next-NVFP4 --out /models/Qwen3.8-Flash-Next-banked
ft serve --model-path /models/Qwen3.8-Flash-Next-banked ... --moe-bank-ram 48G
```

What it does, in order:

1. **Decides what can go by the bytes, not the names.** A checkpoint tensor is dropped only if the
   bank reproduces it exactly. NVFP4 codes and block scales are stored as they are (gate and up
   concatenated); the per-tensor global scales went into the bank as fp16 and stay in the
   checkpoint -- a few hundred kilobytes. gpt-oss MXFP4 blocks, scales and biases all go. Expert
   tensors the server never reads (`input_scale`, the `--spec-mtp` head's experts) stay.
2. **Checks every layer twice**, in one sequential pass over the shards that hold experts: the
   bank's rows must equal what the loader packs from the original (every bank role, the lossy
   ones included), and every tensor being dropped is regenerated from the bank and compared byte
   for byte. A SHA-256 over them per layer and role goes into the record. The bank's own block
   hashes are checked on the way.
3. **Writes the slim checkpoint beside the original**: shards with no expert tensors are
   hard-linked (no space; `--copy` to copy them), shards that mix both are rewritten without the
   experts, shards of nothing but experts are left out. The index is rewritten and every other
   file copied.
4. **Moves the bank file into it** (`bank.ftmb`, a rename; `--keep-bank` leaves it where it is and
   records the path) and marks it as the only copy: a server started on some other checkpoint
   with the same `--moe-bank-dir` refuses to overwrite it.
5. **Deletes nothing.** It ends by saying the original can be deleted. Because of the hard links,
   deleting the original frees the expert data only.

`freetoken_bank_pack.json` in the slim directory records the original shard headers, the index and
the whole-file SHA-256 of every shard that was rewritten or left out. So the step is reversible:

```bash
ft bank verify --model-path /models/Qwen3.8-Flash-Next-banked   # no original needed
ft bank unpack --model-path /models/Qwen3.8-Flash-Next-banked --out /models/Qwen3.8-Flash-Next-NVFP4
```

`verify` checks the block hashes, the removed tensors against the recorded hashes, and the bank
against what the slim checkpoint's kept tensors pack to. `unpack` writes the original files back
and compares each rebuilt shard with the original's SHA-256.

A packed checkpoint is only served with `--moe-bank-ram` and a CPU-capable expert kernel
(`--moe-strategy hybrid`); without them the server stops before loading any weight and says so.
A budget that covers every expert keeps them all resident. A new histogram reorders the bank in
place as before (`ft bank reorder --model-path ... --moe-bank-stats ...` does it without starting
the server); the pack record is about the bytes of each expert, not where their rows sit, so it
stays valid.

What to keep in mind: **the bank file is now the only copy of the experts.** Back it up if the
original is not kept anywhere else, and do not point a cache cleaner at it.

**Disk these steps write**, before you start one:

| step | writes | Ornith-1.5-35B-A3B on an RTX 2060 host (16.9 GiB of experts) |
|---|---|---|
| first start with `--moe-bank-ram` | the whole bank file | 16.9 GiB, 74 s |
| a new `--moe-bank-stats` | twice the reordered layers (journal, then in place) | 19 layers: 16 GiB, 123 s |
| `ft bank pack` | the shards that mix dense and expert tensors, rewritten (the dry run prints it); nothing for hard-linked ones | 3 shards rewritten, 4.9 GiB; 129 s, 3.7 GiB peak RSS |
| `ft bank verify` | nothing | 42 s |
| `ft bank unpack` | the whole original checkpoint, less what can be hard-linked | up to the checkpoint's size (22 GiB here) |

Measure the free space where the files actually land. **Under WSL2, `df` inside the distribution
reports the virtual disk's own capacity, not the space left on the Windows drive that holds it**
-- a 1 TB virtual disk said 855 GiB free while the drive under it ran out. Check the host
drive (`df -h /mnt/c`, or Explorer). And the virtual disk grows as files are written but does not
shrink when they are deleted: a reorder, a pack and an unpack in a row each take their share of the
host drive for good, until the disk is compacted. When the host drive fills, the distribution
stops and will not start until space is freed there.

`ft doctor disk` prints that drive, its free space and whether the virtual disk is sparse (it asks
the registry and PowerShell through WSL interop). A start that is about to write layers into the
bank file warns first when the whole file's missing layers, plus a 20 GiB margin, exceed the
Windows drive's free space. And a bank file is never *created* on a 9p/drvfs path (`/mnt/c/...`), a
network filesystem or tmpfs: the start stops before anything is written there and says where to
point `--moe-bank-dir` instead. A file already placed there is served, with a warning.

If a write into the bank file still fails partway -- the host drive filled after the start checked
it, or an I/O error -- the start stops with `BankFileError: --moe-bank-ram: writing <file> failed:
<reason>`, naming the bank file. That is not the checkpoint: upstream reports a checkpoint it cannot
read as `WeightLoadError`, and this error never carries that name. Free the space, or point
`--moe-bank-dir` elsewhere, rather than fetch the model again.

`pack` and `verify` have run on Ornith on an RTX 2060 host: the packed checkpoint served the same
text at temperature 0 as the original. `unpack` has run on the synthetic checkpoints of the test
suite only -- the run on Ornith was cut short by the full host drive above. Nothing has run on
Flash-Next or gpt-oss-120b yet; the RAM figure for Flash-Next (a few layers' rows, about 5 GiB) is
arithmetic.

### 6. Experimental: ask for the rows each step routes to (`--moe-bank-prefetch`)

```bash
ft serve ... --moe-strategy hybrid --moe-bank-ram 48G --moe-bank-prefetch
```

Without it, a non-resident row is read by whichever CPU executor worker first touches it: a
4 KiB fault, answered by one readahead window around that page, while the other workers fault
on their own pages. That is why step 3 matters so much -- the window decides both how much one
fault brings in and how much of it belongs to experts nobody routed to.

With it, the executor looks at the routing before it wakes its workers. For every distinct
non-resident row the layer is about to read, it checks the page cache (`mincore`) and, for a row
that is not all there, calls `madvise(MADV_WILLNEED)` on exactly that row. That puts every page
of the row into the page cache at once and submits the reads without waiting for them; a worker
that faults afterwards waits for a read already in flight instead of starting its own, and
nothing outside the row is read. The gate/up rows are asked for before the workers start and
the down rows right after, since the down pass cannot begin until the gate/up pass is done.
Rows already in the page cache cost the `mincore` call and nothing else; resident rows and
routes served on the GPU are skipped.

This is not the `madvise` described under "What this looked like while it was wrong" below: that
was a hint on the whole mapping at startup, which changes how faults read ahead. This is a request
for particular rows, made per layer from the routing, and it does not depend on the fault path.

What is known and what is not:

- Measured, off/on/off/on, 2000 tokens each:

  | | TTFT | tok/s, whole run | first 300 | last 300 | p99 gap |
  |---|---|---|---|---|---|
  | 2x RTX 3060, Flash-Next, `--moe-bank-ram 48G`, 40 GiB held (page cache short) | 9.1 → **5.2 s** | 23.9 → **25.7** | 14.0 → 15.7 | 36.6 → 37.8 | 220 → 188 ms |
  | RTX 2060, Ornith, `--moe-bank-ram 6G`, cold rows paged out, RAM to spare | 10.0 / 5.8 → **2.0 s** | 33.4 → 29.8 | 30.1 → 26.4 | 37.4 → 30.9 | same |

  Where the page cache cannot hold the non-resident rows -- the machines this is for -- it helps
  everywhere. Where it can, the fault path's readahead fills the cache with the neighbouring rows
  too and later tokens find them there; asking for exactly the routed rows gives that up, and decode
  once warm was slower. That is why it stays off by default: turn it on for a host whose RAM is
  short of the bank.
- It needs a CPU executor reading the mapped file (`--moe-strategy cpu` or `hybrid`), Linux, and
  a `--moe-bank-ram` that actually split. The startup log says how many file-backed blocks it
  took; otherwise it says why it has nothing to do.
- `MADV_WILLNEED` is advice. Under memory pressure the kernel may allocate fewer pages than asked,
  and the rest are faulted in the old way -- never worse than without the flag, but not better
  either.
- Whether step 3's readahead setting still matters with it on is also unmeasured. Keep it.
- On shutdown (Ctrl+C) it logs how many rows it saw and how much it asked for.

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

**Two lists that have to agree, and only one of them was extended.** `--spec-mtp` appends the
draft head's expert layer to the bank sources *after* the resident placement has been solved.
`layer_residency` is built from the bank sources and covered it; `expert_perm` is built from the
placement's layers and did not. Nothing ties the two together but a comment, and the mismatch is
invisible until an index runs off the end -- on one rank, in a traceback that names neither flag,
while the other rank logs that everything mapped and registered. Anything added to the cache's
banks after `MappedTier` has solved its placement has to grow the permutation list with it.

The estimate that drove the design ("77% resident covers 87.5% of routes, which is fast enough")
held up. The intermediate numbers along the way — +4 ms here, 31 ms there, "3-4x from madvise" —
were worth about a factor of two, and should not have been quoted as if they were measurements.

## Limits and caveats

- **Prefill is slower.** Each chunk still streams every expert of every layer, and the quarter
  that is not registered goes through a pinned bounce buffer. The 2-7x first measured here predates
  the chunk work in [prefill-chunk.md](prefill-chunk.md): on the two RTX 3060s a 19.9k-token prompt
  took 68.3 s at 64 GB-equivalent against 56.5 s with the same flag and all of the RAM, and
  `--prefill-mixer-pieces 2` took those to 52.7 s and 45.0 s.

  Where the page cache is smaller than the non-resident rows, the chunk's bank read is most of
  that: reading one layer's rows evicts the previous layer's, so every chunk reads nearly all of
  them from the disk again. They used to be faulted in through the mapping, one thread and one
  readahead window at a time; they are now read from the file by several threads into the
  bounce buffers, through the page cache, so the decode that follows still finds them there. A
  piece the page cache already holds is copied from the mapping. On an RTX 2060 host with Ornith
  at `--moe-bank-ram 6G` and the server held to 13 GiB, prefill went from 204 to 304 tok/s
  (that host's virtual disk tops out near 2 GiB/s); with readahead turned off entirely, from 28
  to 230 tok/s. With RAM to spare it is unchanged (0.65 s per chunk from the page cache).
  On the two RTX 3060s (Flash-Next, `--moe-bank-ram 42G`, readahead 256 kB) held to about a
  64 GB host's page cache, prefill went from about 310 to 410 tok/s, with one prompt of the old
  path down at 71, and decode stayed within the run-to-run spread (12-14 tok/s).

  `FREETOKEN_BANK_PREAD=direct` reads with `O_DIRECT` instead and leaves the page cache alone. Where
  the page cache is far short of the non-resident rows it is a little faster still (325 against
  304 tok/s on the RTX 2060 host, 440 against 410 on the RTX 3060s, decode unchanged), and worth
  trying on a 64 GB host. Where the page cache nearly holds them it costs decode, which no longer
  finds the rows a prefill read: 15.9 -> 13.2 tok/s on the RTX 3060s with a lighter balloon.
  `--prefill-profile` shows the split per chunk; `FREETOKEN_BANK_PREAD` in
  [kai.md](kai.md#environment-variables-added-by-this-fork) selects the reads.
- **Disk space.** The bank file is a second copy of the experts -- 63.4 GiB for Flash-Next, on
  top of a checkpoint whose other large part is 47.7 GiB of PLE -- until `ft bank pack` removes
  them from the checkpoint (section 5). The PLE is not copied: `--ple-backend disk` reads its rows
  from the checkpoint's own shards, and a packed checkpoint hard-links every shard that holds no
  expert tensors (`--dry-run` shows which).
- **A slow disk changes the answer, and so does which M.2 slot it is in** (`ft doctor disk`
  reports the transport, the link and whether the drive shares the chipset uplink with a GPU;
  the startup log warns about USB, SATA, rotating disks and 9p/network/tmpfs mounts). Decode reads whole
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
