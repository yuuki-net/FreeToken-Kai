# The web console (`ft mgr`)

Upstream ships a desktop app that talks to `ft daemon`. It is built for upstream's engine and does
not know this fork's flags, so this fork serves its own console to a browser instead: nothing to
install on the machine you look from.

![Dashboard](../assets/kai-console-dashboard.png)

## Two commands, two ports

| | `ft daemon` (upstream, unchanged) | `ft mgr` (this fork) |
|---|---|---|
| Port | 1900 | 1901 |
| State directory | `~/.freetoken/daemon` | `~/.freetoken/mgr` |
| Web console at `/ui/` | no | yes |
| Launch profiles, recommended settings, suggestions | no | yes |
| Start, stop, switch, logs | yes | yes |
| From another PC | `--token` guards every route | anyone may look, operating needs the token |

`ft mgr` is the same supervisor as `ft daemon` with the console added on top. It has its own
port and state directory, so both can be installed side by side.

Port 1901 rather than 1900: on Windows the SSDP Discovery service holds 1900, and under WSL's
mirrored networking a listener on 1900 is unreachable even from inside WSL.

```bash
ft mgr --host 0.0.0.0
```

Then open `http://127.0.0.1:1901/ui/` on this PC, or `http://<this PC>:1901/ui/` on another one.
Use `127.0.0.1` rather than `localhost` when the server runs in WSL: `localhost` resolves to `::1`
first, which does not reach WSL. Stopping `ft mgr` leaves a running engine alone, and the next
`ft mgr` takes it over; `ft mgr stop` stops the engine itself.

`ft serve` serves the same dashboard on its own port (`http://<host>:<port>/ui/`), read-only:
no profiles, no start or stop. A server started from a script is still shown by `ft mgr` when it
listens on the default port 1919, marked as not managed by it.

Add `?demo` to the URL to see every panel with made-up numbers and no server.

The console follows the browser's language (English or Japanese) and can be switched from the
header.

## Watching from another PC, operating from this one

`--host 0.0.0.0` exposes the port that also starts and stops the engine. So:

- **Reading is open.** Every GET works from anywhere the port is reachable: the dashboard, the
  logs, the profiles.
- **Operating needs this PC or the token.** Starting, stopping, switching, saving or deleting a
  profile and shutting down are accepted from this PC without a token, and from anywhere else only
  with the right `X-FT-Token` header. Otherwise the answer is `403 write_needs_token`.
- "This PC" means the connection comes from a loopback address **and** the `Host` header names
  loopback (`127.0.0.1`, `localhost`, `::1`). The second half keeps a web page that rebinds its own
  name to 127.0.0.1 from counting as local.
- The token is created on first start in `~/.freetoken/mgr/token` (mode 0600). `--token` (or
  `FREETOKEN_DAEMON_TOKEN`) replaces it. Unlike `ft daemon`, where `--token` guards every route,
  `ft mgr`'s token guards only operations.
- In the console on this PC, **Token** in the header shows it with a copy button. On another PC
  the header says **view only** and the operating buttons are gone; **Operate** asks for the token
  and keeps it in that browser.

A port forward that relays LAN connections to `127.0.0.1` (for example `netsh portproxy` with
`connectaddress=127.0.0.1`) makes them look local. Forward to the WSL address instead.

## Launch profiles

![Profile editor with recommended settings](../assets/kai-console-profile-editor.png)

A profile is a model, a port and the flags for `ft serve`, kept in `~/.freetoken/mgr/profiles.json`.

- **Model**: picked from `~/models`, the directories in `$FREETOKEN_MODELS_DIRS` and the Hugging
  Face cache. A bare folder name is resolved against the local directories on save, so it is never
  sent to the hub as a repository id.
- **Flags**: picked from `ft serve`'s own argument parser, grouped by what they touch, with the
  description of the selected flag in the right pane. A `ft serve` line from a launch script can be
  pasted in.
- **Recommended**: the flags this fork's measurements point to for this PC's GPUs, RAM and cores
  and the checkpoint's `config.json`, each with its reason. They are compared with the profile:
  **not set**, **different value** (current -> recommended) and **already set**. **Apply all**
  applies only the first two. The context length it proposes is one that starts on the card, not
  a prediction of the most that fits.
- **From measurements** (while the server runs): changes read off the running server, each
  applicable with one click: collecting expert statistics, more RAM for the expert banks, more
  expert slots from free VRAM, a longer context once the free VRAM is known, or a shorter one to
  give the expert cache more room.
- A profile row shows start for a stopped profile, and stop and restart for the running one. A
  profile edited after its server started is marked, and restarting applies the edit.

## Benchmark

**Benchmark** in the header measures this PC with a model and settles the flags on the numbers,
drawn live on a speedometer while it runs. It needs the GPU: the server `ft mgr` runs is stopped
first and started again with its own settings at the end, whether the run finished, failed or was
cancelled. A server started outside `ft mgr` is not touched, and the benchmark refuses to start
while one holds the GPU.

1. **Hardware** (a few minutes, `webui/hwbench.py` in a child process): PCIe transfer per GPU
   against the link's theoretical rate, the memory read rate, reading the model's own file from the
   SSD with its page cache dropped, the model's experts computed on the CPU at each thread count,
   the experts moved to each GPU, and both at once. These are upstream's `ft bench bw` kernels
   driven with the model's own expert geometry.
   - `--moe-strategy`: `hybrid` when computing on the CPU is more than twice the transfer to the
     slowest GPU, `offload` otherwise (upstream's rule).
   - `--moe-cpu-threads`: the fewest threads within 95% of the best, not the maximum.
   - The figures are also merged into the GPU's `ft bench bw` profile, which the engine reads for
     the hybrid fetch split (`--moe-hybrid-max-fetch -1`, the default).
2. **Real runs** (optional; about an hour on two RTX 3060s in the standard mode, two in the
   thorough one). Setting A is the recommended settings plus the above. Each later run changes one
   thing from the best so far (`webui/search.py`) and is kept only when it measured faster; what is
   kept is carried on and can open or close later candidates (a chunk budget of 0.9 is tried only
   after 0.75 was kept). Each run is a fresh `ft serve`:
   - prompt processing: two prompts of 16,384 tokens cut from this repository's own documents and
     code, each opened with a unique line so nothing comes from the prefix cache;
   - generation: the median of three 300-token generations continuing prose, and, for a model with
     MTP weights, three more continuing Python, since MTP drafts far better on code.
   - **Standard** tries what has helped on some machine: offload instead of hybrid, the CPU thread
     count, each expert kernel, a 16-bit KV cache, prefill overlap, a chunk budget of 0.75, four
     prefill pieces, moving the pipeline split one layer, pinned PLE, MTP 3 and 5.
   - **Thorough** adds what has not helped so far: no hybrid fetch, the linear kernels, a q4_0 KV
     cache, device-side copies of cache hits, chunk budget 0.9, one piece, a 16k prefill length,
     no host embedding, bf16 dense weights, the split moved two layers, `--pp-send-ahead` and
     `--pp-prefill-group`.
   - **Main use** (prose, code or both) decides a change that helps one and hurts the other, such
     as MTP.
   - Longer context tiers come last, on everything kept, and are kept while generation stays
     within 5% and prompt processing within 10%. A pre-Ampere card that cannot load the model in
     float16 is retried without `--dtype`.
   - Not searched, with the reason on the result page: `--memory-ratio` (the risk is later VRAM use
     by other programs, which a benchmark cannot see), `--max-running-req` and
     `--cuda-graph-max-bs` (throughput, not one person's speed), `--moe-cpu-layers` and
     `--attention-backend` (auto already picks what starts), and the settings that decide what fits
     rather than how fast.

The result lists each flag as **measured** or **rule** with its reason, and can be saved as a new
profile, merged into a profile for the same model, or started directly. The last result per model
is kept in `~/.freetoken/mgr/tune/`.

## How experts are served

![Expert heatmap](../assets/kai-console-heatmap.png)

- **Per layer** (with `--moe-collect-stats`, about 2% slower generation): the share of routed
  experts that were not on the GPU over the last 5 minutes, one cell per layer and one strip per
  GPU. In hybrid mode it can show the share computed on the CPU instead. The colour scale follows
  the worst layer. Below it: the average, the worst and best layers, and what to change: grow the
  cache, move layers between GPUs (`--pp-layers`), or give the CPU side more threads.
- **Per expert** (only with `--moe-stats-out <file>` and `--disable-cuda-graph`, a measurement
  setup): how often each expert was picked, one row per layer, sorted most picked first with
  today's cache size drawn as a line. Under it, what a cache of today's size and of twice that
  would hold if filled with the most picked experts, next to the measured rate, and how many
  experts per layer cover 90% of the picks.

## Where the numbers come from

Every rank writes a small JSON file every 2 seconds to `$TMPDIR/freetoken-webstats-<port>/`
(`scheduler/webstats.py`). A file rather than the reply stream, because under `--pp-size` only
rank 0 talks to the HTTP server. Device counters are copied without blocking decode, behind a
CUDA event, and read on a later tick. The server (`server/kai_api.py`) sums the ranks into
`/v1/stats`'s `kai` block and `/v1/kai/experts`. The per-expert counts go to a separate file every
10 seconds and are returned only with `?freq=true`.

## Limits

- Nothing measures how long a token waits for expert reads, so a suggestion does not come with a
  predicted tok/s gain.
- The per-expert view needs eager decode: under CUDA graphs the routing histogram is not counted.
- The cache-size estimate is the best case of filling the cache with the most picked experts; an
  LRU replay is not simulated.
- Editing a profile does not change a running server until it is restarted.
