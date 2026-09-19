// ?demo: fake documents in the shapes the serve returns, to look at the pages without a server.
"use strict";

if (FT.demo) {
  const GiB = 2 ** 30, t0 = Date.now();
  let s = 7;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const wave = (p, a) => 1 + a * Math.sin((Date.now() - t0) / p);
  const L = 24, E = 256;
  const layers = [];
  for (let r = 0; r < 2; r++) for (let i = 0; i < L; i++) {
    const active = 30000, rate = (r ? 0.075 : 0.042) + 0.02 * Math.sin(i * 1.3) + rnd() * 0.015;
    layers.push({ rank: r, gpu: r, local: i, mtp: false, active, miss: Math.round(active * rate), calls: 4000 });
  }
  layers.push({ rank: 1, gpu: 1, local: L, mtp: true, active: 8000, miss: 900, calls: 4000 });
  const docs = {
    "/health": () => ({ status: "ok", model: "Qwen3.8-Flash-Next", uptime_s: 11520 + (Date.now() - t0) / 1000 }),
    "/v1/stats": () => ({
      model: { id: "Qwen3.8-Flash-Next", ctx: 131072 },
      kv: { used_pages: 2576, total_pages: 8192, page_size: 16 },
      mamba: { used_slots: 3, total_slots: 8 },
      vram_bytes: 11.0 * GiB,
      gpus: [{ index: 0, name: "NVIDIA GeForce RTX 3060", total_bytes: 12 * GiB }],
      throughput: { decode_tps: 18.4 * wave(9000, 0.08), prefill_tps: 412 * wave(13000, 0.3) },
      requests: { active: 1, completed: 57, ttft_mean_ms: 2100 },
      host: { total_bytes: 128 * GiB, used_bytes: 71 * GiB, reclaimable_bytes: 38 * GiB, free_bytes: 19 * GiB,
              swap_total_bytes: 32 * GiB, swap_used_bytes: 0.4 * GiB, mlocked_bytes: 0, engine_bytes: 58 * GiB },
      kai: {
        window_s: 60, kv_cache_dtype: "q8_0", prefix_reuse_rate: 0.61,
        moe: { collect_stats: true, gpu_hit_rate: 0.938, major_faults_per_min: 12400, ssd_read_bytes_per_s: 1.4 * GiB },
        spec: { k: 2, tokens_per_step: 2.42, accept_rate: 0.71 },
        prefill: { chunk: 4096, auto: true },
        decode_sample: [0, 1].map((r) => ({
          rank: r, interval_s: 30, samples: 10, rows: 3,
          total_ms: r ? 126 : 120, ms: { route: r ? 1.6 : 1.5, fetch: r ? 30.2 : 22.4, gpu_experts: r ? 19.1 : 18.3, cpu: 0, other: r ? 75.1 : 77.8 },
        })),
        gpus: [
          { rank: 0, index: 0, name: "NVIDIA GeForce RTX 3060", total_bytes: 12 * GiB, used_bytes: 11.2 * GiB, reserved_bytes: 10.6 * GiB, layers: [0, 23],
            pools: { kv: 1.2 * GiB, moe: 4.6 * GiB, mamba: 0.3 * GiB } },
          { rank: 1, index: 1, name: "NVIDIA GeForce RTX 3060", total_bytes: 12 * GiB, used_bytes: 10.8 * GiB, reserved_bytes: 10.7 * GiB, layers: [24, 47],
            pools: { kv: 1.2 * GiB, moe: 5.1 * GiB, mamba: 0.3 * GiB } },
        ],
      },
    }),
    "/v1/kai/slots": () => ({ ranks: [0, 1].map((r) => {
      const cap = L * E, cur = 700;
      const hit = (x) => 1 - Math.exp(-x / cap * (r ? 9 : 11));
      const curve = Array.from({ length: 24 }, (_, i) => Math.round(cap * (i + 1) / 24)).concat([cur]).sort((a, b) => a - b)
        .map((x) => ({ slots: x, hit: hit(x) }));
      return { rank: r, steps: 1024, accesses: 220000, gpu_layers: L, num_experts: E, capacity: cap, cache_size: cur,
        decode_target: "gpu", hit_at_current: hit(cur), measured_hit_60s: hit(cur) - 0.012, bytes_per_slot: 6.9 * 2 ** 20, curve };
    }) }),
    "/v1/cache/status": () => ({ geometry: {
      num_pages: 8192, page_size: 16, moe_cache_size: 1400, num_mamba_slots: 8, num_experts: E, num_moe_layers: L,
      unit_bytes: { kv_per_token: 20000, moe_per_expert: 1.6 * 2 ** 20, mamba_per_slot: 60 * 2 ** 20 },
    } }),
    "/v1/requests": (q) => (/since=[1-9]/.test(q || "") ? { next_cursor: 3, entries: [] } : { next_cursor: 3, entries: [
      { ts: new Date(t0 - 660e3).toISOString(), path: "/v1/chat/completions", status: 200, duration_ms: 36200, ttft_ms: 400, prompt_tokens: 2113, completion_tokens: 846 },
      { ts: new Date(t0 - 240e3).toISOString(), path: "/v1/chat/completions", status: 200, duration_ms: 69800, ttft_ms: 1200, prompt_tokens: 38560, completion_tokens: 1204 },
      { ts: new Date(t0 - 5e3).toISOString(), path: "/v1/chat/completions", status: 200, duration_ms: 3000, ttft_ms: 3000, prompt_tokens: 41208, completion_tokens: null },
    ] }),
    "/v1/kai/experts": (q) => ({
      expert_freq: /freq=(1|true)/.test(q || "") ? [0, 1].map((r) => ({ rank: r, layer_range: r ? [24, 48] : [0, 24], seconds: 300,
        freq: Array.from({ length: L }, (_, i) => {
          const order = Array.from({ length: E }, (_, k) => k).sort(() => rnd() - 0.5);
          const row = new Array(E).fill(0);
          order.forEach((id, k) => { row[id] = Math.round(2400 / Math.pow(k + 1, 1.35 + 0.15 * Math.sin(i + r)) * (0.8 + rnd() * 0.4)); });
          return row;
        }) })) : null,
      window: "300", layers, host_memory: { mem_total: 128 * GiB, mem_available: 41 * GiB },
      config: { moe_bank_ram: "24G", kv_cache_dtype: "q8_0", pp_size: 2, spec_mtp: 2 },
      ranks: [0, 1].map((r) => ({
        rank: r, layer_range: r ? [24, 48] : [0, 24], moe: { collect_stats: true, decode_target: "gpu", cache_size: 700 },
        gpu: { index: r, name: "NVIDIA GeForce RTX 3060", total_bytes: 12 * GiB, free_bytes: (r ? 1.2 : 0.8) * GiB, reserved_bytes: 10.6 * GiB },
        window: { seconds: 300, decode_tokens: 5520, major_faults: r ? 38000 : 24000, prefill_new_tokens: 41208, prefill_cached_tokens: 38000 },
      })),
    }),
  };
  // ten minutes of made-up samples, so the chart is not empty when the page opens
  FT.demoHistory = () => Array.from({ length: 300 }, (_, i) => {
    const ts = t0 - (300 - i) * 2000, x = i / 300;
    const busy = x < 0.35 || (x > 0.55 && x < 0.8) || x > 0.9;
    return [ts, busy ? 18.4 + Math.sin(i / 5) * 1.2 + (rnd() - 0.5) : 0, busy && (i % 60 < 12) ? 380 + rnd() * 120 : 0];
  });
  FT.serveGet = async (path, query) => {
    const f = docs[path];
    if (!f) throw new FT.HttpError(404, null);
    return JSON.parse(JSON.stringify(f(query)));
  };
}
