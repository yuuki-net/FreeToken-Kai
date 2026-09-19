# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bank` | Inspect, pack, verify, unpack or reorder the `--moe-bank-ram` bank file |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |
| `ft doctor disk` | Check whether `--moe-bank-ram` suits this host's disk and RAM, and what to set (no GPU, no root) |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--dense-quant` | none | `fp8` serves the checkpoint's bf16 dense (non-expert) weights as per-row fp8-e4m3, quantized at load and read W8A16: attention, GDN, shared expert, lm_head, embedding. Already-quantized projections keep their format; router, hyper-connection, QSA indexer, PLE and GDN b/a gates stay bf16. See [pipeline.md](pipeline.md) |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length, and what `/v1/models` advertises. The KV pool can hold less; the startup log's `context limit` line gives the real figure |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |
| `--pp-size` | 1 | Layer-split (pipeline) parallelism across N GPUs, one process per card, the residual stream handed over gloo -- no NCCL and no peer access. List the cards in rank order with `--gpu`. Mutually exclusive with `--tp-size` > 1. See [pipeline.md](pipeline.md) |
| `--pp-layers` | even split | Layer boundaries of the `--pp-size` split, N-1 comma-separated values: `24` gives rank 0 layers [0,24) and rank 1 [24,48). For cards of different sizes |
| `--spec-mtp` | 0 | Verify K drafts per step from the checkpoint's own MTP head. Single-request decode (`--max-running-req 1`); Qwen3.5-MoE family and Qwen3.8-Flash-Next NVFP4 checkpoints. See [pipeline.md](pipeline.md) |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--host-embedding` | off | Keep the input embedding table in pinned host memory and gather its rows over PCIe in place, inside CUDA graphs too. Frees about 1 GB on a 250k-token vocabulary, which becomes KV pages on a small card. Qwen3.5-MoE and Qwen3.8-Flash-Next families (on Flash-Next the host copy stays in the model dtype: about 0.6 GiB of VRAM for 1.3 GB of pinned RAM under `--dense-quant fp8`) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile. `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts. With the default a server holds about 8k tokens of context whatever the model allows; the startup log warns when the pool is below the model's limit and names this flag |
| `--linear-state-cache-ratio` | 2.0 | Hybrid GDN models: GDN-state snapshots kept for prefix reuse, per running request (floor 4). This, not KV, bounds how many conversations stay reusable (measured: 2 → 4 conversations at ratio 8 on Ornith, 5 → 7 on Flash-Next — the rule differs per model); each slot is one GDN state of VRAM taken from the expert cache. See [prefix-reuse.md](prefix-reuse.md) |
| `--prefix-disk-cache DIR` / `--prefix-disk-cache-size` | off / 32G | Hybrid GDN models (Qwen3.5-MoE, Qwen3.8-Flash-Next), one GPU: write prefix-cache entries (a prompt's KV pages and the GDN snapshot at its end, prefixes of 1024+ tokens) to `DIR` while the server is idle, and read one back instead of prefilling again when a prompt starts with it and the in-memory cache no longer does -- including after a restart. Entries are keyed by the exact tokens and by the model, weights, code version and cache layout; nothing else is ever read. The size caps the directory, least recently used out first. Refused with `--pp-size` / `--tp-size` > 1. Experimental: on an RTX 2060 an 8k-token prompt came back from disk in 1.2-1.3 s against 13-14 s to prefill it. See [prefix-reuse.md](prefix-reuse.md#keeping-prefixes-on-disk---prefix-disk-cache) |
| `--kv-cache-dtype` | auto | Store the paged KV as block-quantized codes: `q8_0` (1.88x smaller) or `q4_0` (3.56x). Plain paged-attention models on `--attention-backend triton`, Flash-Next on its own `qsa_sparse` backend, or gpt-oss (both its full and its sliding-window layers; resolves to Triton by itself; use `q8_0`, `q4_0` breaks its answers); refused at startup otherwise. See [kv-cache-quant.md](kv-cache-quant.md); whether the freed VRAM buys you anything depends on your machine, see [vram-and-speed.md](vram-and-speed.md) |
| `--prefill-chunk-budget` | 0.55 | Share of free VRAM one prefill chunk's transient may take. The engine measures that cost per token at startup, sizes `--max-prefill-length` to fit, and re-solves before every prefill against the VRAM free right then. 0 disables and the flag is used as given. Under `--pp-size` the chunk is agreed across the ranks at startup, checked once more after the `--spec-mtp` graphs are captured (it can only shrink), and then frozen, because it sizes a cross-rank message. On a card that only serves, raising it is the cheapest prefill win there is: on two RTX 3060s (Qwen3.8-Flash-Next, `--pp-size 2 --prefill-mixer-pieces 2`), 0.75 took the chunk from 3,584 to 5,120 tokens and a 25k-token prompt from 510 to 702 tok/s. See [prefill-chunk.md](prefill-chunk.md) |
| `--prefill-mixer-pieces` | 1 | Run each prefill chunk's GDN / attention over this many consecutive pieces and its MoE over the whole chunk. The mixers set the transient that `--prefill-chunk-budget` sizes the chunk from, so the chunk comes out wider and an offloaded MoE streams its banks fewer times per prompt; raise `--max-prefill-length` with it. Qwen3.8-Flash-Next (one GPU or `--pp-size`; use 2, 4 measured no better) and single-GPU Qwen3.5-MoE (4 with a 16384 ceiling); other models ignore it. See [prefill-chunk.md](prefill-chunk.md#wider-chunks---prefill-mixer-pieces) |
| `--pp-send-ahead` | 1 | With `--pp-size`: how many residual streams the first rank may have in flight to the next rank at once. At 1 the hand-off waits for the peer's receive, so the rank stays one chunk ahead; above 1 it keeps prefilling while the next rank works through what it has, at one chunk's residual stream of pinned host memory per slot. On the two RTX 3060s of [pipeline.md](pipeline.md) it changed nothing (the two ranks were already balanced at a 5,120-token chunk); it is here for a machine whose ranks are not. See [pipeline.md](pipeline.md#two-knobs-for-a-slow-second-rank) |
| `--pp-prefill-group` | 1 | Qwen3.8-Flash-Next with `--pp-size 2`, offloaded experts and `--max-running-req 1`: the second rank holds up to this many consecutive prefill chunks of a prompt and runs them layer by layer, so each layer's expert bank is copied to that GPU once per group instead of once per chunk. It cuts that rank's wait on its banks (0.27 to 0.08 s per 1k tokens measured), but on the two RTX 3060s of [pipeline.md](pipeline.md) the prefill came out no faster, so it stays off by default; it is here for a rank whose link is slower still. Costs one residual stream of VRAM per held chunk on that rank (98 MiB for a 5,120-token Flash-Next chunk). See [pipeline.md](pipeline.md#two-knobs-for-a-slow-second-rank) |
| `--prefill-profile` | off | Log one line per prefill forward on every rank: its wall time split into waiting for the other pipeline rank, host copies of expert rows the GPU cannot read directly (the `--moe-bank-ram` remainder; page faults on the bank file land here), per-layer-embedding disk reads, and the GPU plus the rest; then the GiB those copies moved and their rate, major faults and storage reads, the page cache's share of the non-resident rows when the forward began, and the achieved PCIe rate of the registered rows. Go by the storage reads, not the share: a chunk can re-read nearly all of the rows while a third are cached, because reading one layer evicts the one before. On an RTX 2060 host with Ornith at `--moe-bank-ram 6G`: 0.7 s of a 4.5 s chunk with RAM to spare; with the server held to 13 GiB, 9.2 s of 13 s, 10.6 GiB re-read from the disk each chunk at 1.2 GiB/s. One device sync per prefill forward |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for Windows/WSL only, where CUDA pinned memory is capped; every value needs an expert format the CPU executor serves (bf16, nvfp4, mxfp4), so fp8 experts cannot use it |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap. `--moe-cache-auto` turns it off by itself when its 2 x num_experts slot floor does not fit next to the KV reserve |
| `--moe-bank-ram` | off | Half the RAM: keep only the frequently routed experts resident, map the rest from disk. Whole-host cap (`48G`), split across ranks. `auto` = MemAvailable at startup − 4.5 GiB per rank − a page cache margin (5% of MemTotal, at least 2 GiB), and under WSL2 no more than the CUDA pin budget (40% of MemTotal, `FREETOKEN_PIN_BUDGET_GB`) − 2 GiB, with the arithmetic logged. Needs `RLIMIT_MEMLOCK` (`ulimit -l`) at least as large as one rank's share, or the resident half is quietly smaller than asked. See [bank-ram.md](bank-ram.md) |
| `--moe-bank-stats` | — | Routing histograms (from `--moe-stats-out`, every rank's file) that decide which experts stay resident. A change reorders the bank file in place; without the flag the order already in the file is kept |
| `--moe-bank-dir` | `~/.cache/freetoken/bankmap/<model>` | Where the bank file (`bank.ftmb`, one for every rank) lives. A checkpoint packed by `ft bank pack` keeps its own inside it. The startup log warns when that is a 9p/drvfs, network or tmpfs mount, a USB, SATA or rotating disk |
| `--moe-bank-readahead` | off | With `--moe-bank-ram`: `auto` writes the recommended device `read_ahead_kb` for the model's block geometry, a number writes that many kB; `off` only logs the current window and the command to change it. Each rank sets it before opening its mapping, since an open mapping keeps the window it was opened with. Device-wide and left set after exit. See [bank-ram.md](bank-ram.md#3-set-the-device-readahead) |
| `--moe-bank-rewarm` | 0 | With `--moe-bank-ram`: after this many seconds idle, each rank reads back the non-resident rows the page cache has lost (below 95%), in file order, and pages its own swapped memory back in, stopping when a request arrives; backs off while memory stays under pressure. `FREETOKEN_REWARM_SWAP=0` leaves the swap alone. 0 = off, because it reads the disk while idle. See [bank-ram.md](bank-ram.md#4-optional-read-the-cold-rows-back-while-idle---moe-bank-rewarm) |
| `--moe-bank-prefetch` | off | Experimental. With `--moe-bank-ram` and cpu/hybrid decode, Linux: before the CPU executor computes a layer, it asks the kernel for exactly the non-resident expert rows that layer routes to (`mincore`, then `MADV_WILLNEED` per row), instead of leaving them to 4 KiB faults and readahead windows. For hosts whose RAM is short of the bank: on two RTX 3060s at 64 GB-equivalent, time to first token 9.1 → 5.2 s and decode +7%; on a host with RAM to spare decode was slower. See [bank-ram.md](bank-ram.md#6-experimental-ask-for-the-rows-each-step-routes-to---moe-bank-prefetch) |
| `--moe-stats-out` | off | Write the per-expert decode routing histogram, rewritten each time the server goes idle (pass `--disable-cuda-graph`) |
| `--moe-collect-stats` | off | Accumulate the cache's decode miss-rate counters device-side, captured into the decode graph; `--moe-stats-out` reads them back |
| `--disable-cuda-graph` | graphs on | Decode eagerly; needed for `--moe-stats-out` to see real routing |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |

### Image input

Experimental. Needs a checkpoint whose family registers a vision encoder ([models.md](models.md#image-input) lists them and how each one
maps the flags below); a request carrying images is rejected otherwise. Images are accepted on all three protocols (OpenAI `image_url`,
Anthropic `image` blocks, Responses `input_image`) as an http(s) URL or base64. Images inside a tool
result (an Anthropic `tool_result` block from Claude Code's Read, a Responses `function_call_output`
from Codex's view_image) are moved to the user turn that follows the tool message, as vLLM does,
because chat templates render tool messages as plain text.
`GET /v1/stats` reports what the server accepts as `model.input_modalities` (`["text"]` or `["text", "image"]`),
so a client can gate its attachment controls without reading the checkpoint config.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text-model-only` | off | Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to the KV/expert pools) and every multimodal input is rejected. Same as `--mm-disable` with every encoder kind |
| `--mm-disable` | none | Encoder towers to leave unbuilt (`vision`, `audio`); every input they would serve is rejected |
| `--mm-encoder-weights` | host | Where the encoder tower's block weights live. `host` streams them from pinned host banks two blocks at a time behind the compute, so the GPU holds two blocks instead of the whole tower; small images pay the copy time, large ones hide it behind the compute. `gpu` keeps them resident. An encoder without a block stack stays resident either way. `cpu` (Kai): the vision tower runs on the CPU in the tokenizer worker, with no VRAM and no pinned memory, a few seconds per image; Qwen3.5/3.6 and Qwen3.8-Flash-Next, see [image-input.md](image-input.md) |
| `--mm-encoder-dtype` | auto | (Kai) Compute dtype of the Qwen VL vision tower on the GPU (Qwen3.5/3.6, Qwen3.8-Flash-Next, Qwen3-VL); other families' towers follow the model dtype. `auto` builds it in float32 when the model runs bfloat16 and in the model dtype otherwise: the Qwen VL vision tower loses about 9% of its output in bfloat16 (see [image-input.md](image-input.md)). `float32`, `float16`, `bfloat16` force one |
| `--image-min-tokens`, `--image-max-tokens` | processor defaults | Per-image token budget: the image processor resizes every image to take between these many tokens, converted to the family's own units by its processor. A family with fixed budgets honors the maximum only and refuses one below its smallest budget at start-up |
| `--mm-processor-kwargs` | none | JSON object of extra keyword arguments for the checkpoint's image processor call, for knobs the token budget does not cover; applied after the budget, so an explicit key wins |
| `--mm-embed-cache-device` | cpu | Where encoded image embeddings live between prefill chunks. `cpu` keeps them out of the VRAM budget; `cuda` skips the copy back |
| `--allowed-media-domains` | any | Comma-separated hostname allowlist for image URLs; requests for other domains are rejected with a 400. Empty allows any domain |
| `--allowed-local-media-path` | off | Directory `file://` image refs may be read from; unset rejects local files |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, accepted input modalities |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--quant-backend moe.nvfp4=b12x] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

## ft bank

```bash
ft bank info    --model-path <model> [--moe-bank-dir D]
ft bank pack    --model-path <model> --out <slim_dir> [--dry-run] [--copy] [--keep-bank] [--moe-bank-dir D]
ft bank verify  --model-path <slim_dir> [--moe-bank-dir D]
ft bank unpack  --model-path <slim_dir> --out <model> [--copy] [--moe-bank-dir D]
ft bank reorder --model-path <model> --moe-bank-stats a.rank0.json a.rank1.json ... [--moe-bank-dir D]
```

The bank file `--moe-bank-ram` serves the experts from, as the canonical copy of them. `pack`
writes a checkpoint without the expert tensors the bank reproduces -- after checking each one
byte for byte and the bank against what the original packs to -- hard-links the shards that hold
no experts, moves the bank file into it, and deletes nothing. `verify` repeats the check without
the original; `unpack` writes the original files back and checks them against the SHA-256 recorded
at pack time; `reorder` applies a new placement in place. No GPU. See
[bank-ram.md](bank-ram.md#5-optional-drop-the-second-copy-ft-bank-pack).

## ft doctor disk

```bash
ft doctor disk --model /models/Qwen3.8-Flash-Next-NVFP4 --pp-size 2
ft doctor disk --model /models/gpt-oss-120b --moe-bank-stats ~/moe-stats.rank0.json --eval-stats ~/other-session.rank0.json
ft doctor disk --moe-bank-dir /nvme/bankmap/gpt-oss-120b --disk-gbs 3.2 --base-step-ms 55 --ram 32G,48G
```

Whether `--moe-bank-ram` is usable here, before a 60 GiB bank file is written. Needs no GPU
and no root; reads `/proc` and `/sys` and nothing else unless it benchmarks.

- **The bank file**, found the way `ft serve` finds it (`--moe-bank-dir`, the one a packed
  checkpoint names, else the cache directory): which of the model's layers it holds, whether it
  is the only copy of a packed checkpoint's experts, and per-rank files an older build left.
- **Storage** of the bank file (and of the checkpoint when it is elsewhere): filesystem,
  the block device under it through dm/partitions, transport (NVMe/SATA/USB/virtual), the NVMe
  PCIe link, and an estimate of whether the drive sits behind the chipset -- and shares that
  uplink with a GPU. Inside WSL2 the drive is behind a virtual disk: the report names the
  `ext4.vhdx`, the Windows drive it grows on and that drive's free space (which `df` inside the
  distribution does not show), and whether the virtual disk is sparse.
- **Readahead**: the current window against the one recommended for this model's widest
  expert-row block, and the exact command to set it (before starting the server).
- **Memory**: MemTotal/MemAvailable, swap, `ulimit -l`, and what `--moe-bank-ram auto` would choose.
- **GPUs**: each GPU's PCIe link as `nvidia-smi` reports it, and again during pinned host -> GPU
  copies timed for `--h2d-seconds` (default 2; 0 skips) -- an idle GPU may report a lower
  generation. A width or generation below what the GPU and slot allow, under load, is a finding:
  every prefill chunk streams the rank's whole bank through that link. The copies need a GPU a
  running server does not already fill.
- **Read benchmark**: whole expert rows at random from the bank file once it holds every layer
  (otherwise the checkpoint's largest file -- unwritten layers read back as zeros),
  O_DIRECT, one thread and then one per physical core per rank; then, from a complete bank
  file, the reads a prefill chunk makes of the non-resident rows (the server's threads, piece
  size and `FREETOKEN_BANK_PREAD`, from a range dropped from the page cache), and the same range
  copied out of a mapping, which is how it read them before and follows `read_ahead_kb`. Skipped when another process
  maps the file (a running server), unless `--bench-anyway`. `--bench-seconds 0` skips it.
- **Prediction per RAM cap**: resident share, routes covered (from `--moe-bank-stats`, counted
  on `--eval-stats` when given -- the same histogram overstates it), page cache left, disk read
  per token, and with a read rate and `--base-step-ms` a step time. The assumptions are printed
  under the table; the model was within about 2x of the measurements it was checked against.
- **Prefill data movement per chunk, per rank**: for each cap, the rank's non-resident rows, the
  page cache beside them, what comes from the disk (all of them each chunk when the page cache is
  smaller -- reading one layer evicts the one before), the copy time at the prefill read rate
  (`--prefill-read-gbs` to supply it) and the factor the page-cache rate would cost, and the
  transfer time at the slowest host -> GPU rate. Data
  movement only; compare with the server's `--prefill-profile` lines.

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-strategy auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.

