# Freeing VRAM did not make it faster — and how to tell whether it would on your machine

This is a negative result, published because the reasoning behind it is what decides how you
should spend VRAM on a small card, and because it is easy to spend a week on the wrong lever.
Three plausible optimisations were measured here. None of them moved decode. The explanation
is a bandwidth ratio, and on a different machine that ratio points the other way.

## The machine these numbers come from

Everything below is one configuration, and **the conclusion does not transfer without
checking the same two numbers on yours**:

| | |
|---|---|
| GPU | RTX 2060 6 GB, PCIe 3.0 x16 |
| CPU | **Core i7-13700K, 16 cores / 24 threads**, AVX2+VNNI (no AVX-512) |
| RAM | 32 GB, 24 GB given to WSL2 |
| host RAM → GPU (pinned, linear copy) | **10.4 GB/s** measured (12.2 GB/s by `ft bench bw`) |
| CPU reading its own DRAM | **46.6 GB/s** measured (54.3 GB/s by `ft bench bw`) |
| model | Ornith-1.5-35B-A3B-NVFP4 — 40 layers, 256 experts each, top-k 8 |

That is a **fast CPU behind a slow bus**. Hold onto that; it is the whole story.

## How a miss is paid for

The experts do not fit in 6 GB, so they live in host RAM (16.93 GiB of banks) and the GPU
keeps a cache of the hottest ones — 358 slots, 3.5% of the 10,240 experts. On a cache miss
the engine can pull the expert over PCIe into a slot, or leave it in RAM and let the CPU
compute that expert instead. It benchmarks both at startup:

```
--moe-hybrid-max-fetch auto: fetching 13.7% of each decode step's expert misses over PCIe
(benched PCIe/CPU bandwidth ratio), the rest on the CPU executor
```

Over 872,176 measured routed accesses the realized split was `fetched_per_layer=0.065`
against `cpu_per_layer=4.561`: **98.6% of misses are computed on the CPU, not fetched.**

One decode step routes to 8 experts in each of 40 layers — about **310 MB of expert weights
per token** at 1.6875 MiB each. Over PCIe that is ~30 ms. The CPU reading the same bytes from
its own DRAM is ~6.7 ms, and then it computes on them with 6-16 AVX2/VNNI threads.

**So a cache hit is only worth what a miss would have cost, and here a miss is cheap.**

## Three levers, none of which moved decode

**More slots.** `--kv-cache-dtype q4_0` frees 0.9 GiB of KV at 64k, which `--moe-cache-auto`
turns into expert slots: 358 → 902. Warm-cache hit rate 42.2% → 50.7%. Decode: **+2%**.

**More layers reachable by the cache.** When the banks exceed the pin budget the engine locks
whole layers to the CPU — 21 of 40 here. Raising `FREETOKEN_PIN_BUDGET_GB` took that to 17,
then 12. Decode: **+2-3%, inside the run-to-run noise.**

**More CPU threads.** The daily setting was `--moe-cpu-threads 6` on a 16-core CPU. Tried 6,
12 and 16, two runs each:

| threads | decode @8k | decode @32k |
|---|---|---|
| 6 | 23.8 / 22.2 | 19.8 / 15.9 |
| 12 | 27.2 / 24.6 | 19.9 / 18.0 |
| 16 | 22.8 | 16.2 / 18.4 |

12 looks best, but the same configuration measured 19.8 and 15.9 on two runs — **the spread
within one setting is larger than the difference between settings.** No claim survives that.
(16 threads also pins to fewer cores than 12 does, `pinned to cores 0..7`, which is worth
looking into separately — this CPU has 8 P-cores and 8 E-cores.)

**And all together** — four configurations at a 30k context, two runs each:

| | CPU-locked layers | slots | decode @32k |
|---|---|---|---|
| baseline | 21 | 358 | 20.2 / 20.3 tok/s |
| more slots (`q4_0`) | 21 | 902 | 13.6 / 13.5 |
| more layers (budget 11) | 17 | 358 | 20.4 / 19.1 |
| both | 17 | 902 | 13.6 / 13.6 |

(The `q4_0` rows are slower for an unrelated reason: dequantizing a 30k context on every
decode step. See [kv-cache-quant.md](kv-cache-quant.md).)

## Why

The expert cache is worth exactly as much as your bus is slow. Here the bus is **4.5x slower
than the CPU's own path to the same bytes**, so the engine routes almost everything to the
CPU, and the CPU is fast enough that avoiding it saves little. Growing the cache cut CPU work
from 4.56 to 3.92 experts per layer per token — 14% of a term that is itself only a third of
the total, because 21 of 40 layers never consult the cache at all.

## When it would come out differently

This is the part to read if your machine is not this machine.

- **A weaker CPU.** Six to sixteen AVX2/VNNI threads on a 13700K is what makes the fallback
  competitive. On a laptop CPU, or four cores, a miss gets expensive and the cache starts to
  pay for itself. **`--kv-cache-dtype` exists for exactly that case**: it is the cheapest way
  to buy slots.
- **A faster bus.** 10-12 GB/s is PCIe 3.0. The other machine this fork is developed on — two
  RTX 3060s on PCIe 4.0 — measures 24.3 GB/s host-to-device, twice this. Fetching becomes
  worth more than 13.7% there, and the cache with it.
- **Enough RAM to pin the whole bank.** The 21 CPU-locked layers exist because 16.93 GiB of
  banks exceed the pin budget (40% of RAM under WSL). Pin all of it and every layer reaches
  the cache, which changes the arithmetic before you even start.
- **Native Linux.** FreeToken caps host pinning at 40% of RAM on WSL and does not cap it at
  all on native Linux, so the CPU/GPU layer split — and therefore all of the above — can
  differ on identical hardware.

## How to tell, on your machine, in two log lines

```
--moe-cpu-layers auto: banks 16.93 GiB > pin budget 8.44 GiB; locking 21 head+tail MoE layers
--moe-hybrid-max-fetch auto: fetching 13.7% of each decode step's expert misses over PCIe
```

The first says how much of the model the slot cache cannot touch. The second says how much
the engine trusts your bus. **Many locked layers and a small percentage means cache size is
not your lever** — do not spend VRAM there. Few locked layers and a large percentage means
the opposite, and freeing VRAM for slots should pay.

## A caveat about all of these numbers

This was measured on a desktop that is also used as a desktop. Free VRAM moved by ~150 MB
between runs depending on what was on screen, and decode medians for one fixed configuration
varied by up to 20%. Every difference reported above as "no effect" is a difference smaller
than that spread — which is a real finding for a machine like this, but it is not the same as
proving the effect is zero. The one result that survived the noise cleanly, reproducing on
eight runs out of eight, is the `q4_0` decode cost at long context.

## What was not measured

- The gather path. 10.4 GB/s is a linear copy; the real fetch gathers scattered experts, and
  on the other machine that measures 11.8 GB/s against a 24.3 GB/s linear ceiling — roughly
  half. The same ratio here would put the effective fetch path near 5-6 GB/s.
- Any of this on a machine that can pin its whole bank.
- Whether WSL's PCIe path is materially slower than native Linux on the same card.
