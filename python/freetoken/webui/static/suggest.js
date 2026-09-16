// What the measurements say to change, as data. Each suggestion carries `set`: the flags it would
// write, so the profile editor can apply it instead of leaving the reader to edit by hand.
"use strict";

const FTSuggest = (() => {
  const GiB = 2 ** 30;

  async function load() {
    const [exp, stats, geo] = await Promise.all([
      FT.serveGet("/v1/kai/experts", "?window=300"),
      FT.serveGet("/v1/stats"),
      FT.serveGet("/v1/cache/status").then((d) => d.geometry).catch(() => ({})),
    ]);
    return { exp, stats, geo: geo || {} };
  }

  // -> {state, facts, suggestions[]}; state says why there is nothing to suggest
  // modelMax: the checkpoint's own context limit (max_position_embeddings), when known
  function compute({ exp, stats, geo, modelMax }) {
    const { fmt, t } = FT;
    const ranks = exp.ranks || [];
    if (!ranks.length) return { state: "no_data", suggestions: [] };

    const collect = ranks.some((r) => r.moe?.collect_stats);
    const isMoe = ranks.some((r) => r.moe);
    const layers = (exp.layers || []).filter((l) => !l.mtp);
    const A = layers.reduce((s, l) => s + l.active, 0), M = layers.reduce((s, l) => s + l.miss, 0);
    const hit = collect && A ? 1 - M / A : null;
    const secs = Math.max(0, ...ranks.map((r) => r.window?.seconds || 0));
    const decodeTok = Math.max(0, ...ranks.map((r) => r.window?.decode_tokens || 0));
    const faults = ranks.reduce((s, r) => s + (r.window?.major_faults || 0), 0);
    const faultsPerTok = decodeTok ? faults / decodeTok : null;
    const mem = exp.host_memory || {};
    const ub = geo.unit_bytes || {};
    const hybrid = ranks.some((r) => r.moe?.decode_target === "hybrid");
    const allExperts = ranks.reduce((s, r) => s + (r.moe?.num_layers || 0) * (r.moe?.num_experts || 0), 0);
    const cached = ranks.reduce((s, r) => s + (r.moe?.cache_size || 0), 0);
    const slotImpact = (slots) => {
      const share = allExperts > cached ? slots / (allExperts - cached) : 0;
      return share >= 0.15 && hit < 0.9 ? 3 : share >= 0.05 ? 2 : 1;
    };

    const facts = {
      collect, isMoe, hybrid, hit, decodeTok, faultsPerTok, secs,
      missPerTok: collect && decodeTok ? M / decodeTok : null,
      faultsPerMin: secs ? faults * 60 / secs : null,
      ram: mem.mem_available, ramTotal: mem.mem_total, layers: exp.layers || [], allExperts, cached,
    };
    const out = [];

    if (isMoe && !collect) {
      out.push({ impact: 0, title: t("tun_sug_collect"), detail: t("tun_sug_collect_d"),
        flag: "--moe-collect-stats", set: [{ flag: "--moe-collect-stats", value: "" }] });
    }
    if (faultsPerTok != null && faultsPerTok >= 2) {
      const avail = mem.mem_available || 0;
      const bankRam = exp.config?.moe_bank_ram;
      if (avail > 4 * GiB) {
        const cur = bankRam ? parseFloat(String(bankRam)) : 0;
        const next = Math.max(8, Math.round((cur || 0) + avail / GiB * 0.6));
        out.push({
          impact: faultsPerTok >= 30 ? 3 : 2, title: t(bankRam ? "tun_sug_ram" : "tun_sug_ram_new"),
          detail: t("tun_sug_ram_d", { n: faultsPerTok.toFixed(1), avail: fmt.gib(avail) })
            + (bankRam ? " " + t("tun_sug_ram_cur", { v: bankRam }) : ""),
          flag: "--moe-bank-ram", set: [{ flag: "--moe-bank-ram", value: `${next}G` }],
        });
      } else {
        out.push({ impact: 2, title: t("tun_sug_noram"), detail: t("tun_sug_noram_d", { n: faultsPerTok.toFixed(1), avail: fmt.gib(avail) }) });
      }
    }
    if (hit != null && hit < 0.97 && ub.moe_per_expert) {
      const spare = ranks.map((r) => Math.max(0, (r.gpu?.free_bytes || 0) - 0.6 * GiB));
      const extra = Math.floor(Math.min(...spare) / ub.moe_per_expert);
      if (extra > 0) {
        out.push({
          impact: slotImpact(extra), title: t("tun_sug_slots", { n: fmt.num(extra) }),
          detail: t("tun_sug_slots_d", { miss: fmt.pct(1 - hit), free: fmt.gib(Math.min(...spare) + 0.6 * GiB), cur: fmt.num(geo.moe_cache_size) }),
          flag: "--moe-cache-size", set: [{ flag: "--moe-cache-size", value: String((geo.moe_cache_size || 0) + extra) }],
        });
      } else if (geo.num_pages && ub.kv_per_token) {
        const halfKv = geo.num_pages * geo.page_size * ub.kv_per_token / 2;
        const slots = Math.floor(halfKv / ub.moe_per_expert);
        const half = Math.max(4096, Math.floor(geo.num_pages * geo.page_size / 2 / 4096) * 4096);
        out.push({
          impact: Math.min(2, slotImpact(slots)), title: t("tun_sug_ctx"),
          detail: t("tun_sug_ctx_d", { freed: fmt.gib(halfKv), slots: fmt.num(slots), cur: fmt.num(cached), all: fmt.num(allExperts) }),
          flag: "--max-seq-len-override",
          set: [{ flag: "--max-seq-len-override", value: String(half) }, { flag: "--kv-reserve-tokens", value: String(half) }],
        });
      }
    }
    // The recommended context is a value that starts, sized by the card. Once the server runs, the
    // free VRAM is measured: grow the KV pool into it, or, when the expert cache sizes itself
    // (--moe-cache-auto), give up at most a tenth of that cache for it and say so.
    const halving = out.some((s) => s.flag === "--max-seq-len-override");
    if (!halving && geo.num_pages && ub.kv_per_token) {
      const now = geo.num_pages * (geo.page_size || 1);
      const cap = Math.min(modelMax || 262144, geo.limits?.kv_tokens?.max || Infinity);
      const measured = Math.min(...ranks.map((r) => r.gpu?.free_bytes || 0));
      const free = Math.max(0, measured - 0.6 * GiB);
      const autoCache = isMoe && !exp.config?.moe_cache_size && geo.moe_cache_size > 0 && ub.moe_per_expert > 0;
      let best = null;
      for (const ctx of [32768, 65536, 131072, 262144]) {
        if (ctx < now * 1.5 || ctx > cap) continue;
        const need = (ctx - now) * ub.kv_per_token;
        const lost = need <= free ? 0 : autoCache ? Math.ceil((need - free) / ub.moe_per_expert) : null;
        if (lost == null) continue;
        // experts the GPU already misses are not worth trading away
        if (lost > 0 && (lost > geo.moe_cache_size * 0.1 || (hit != null && hit < 0.9))) continue;
        best = { ctx, need, lost };
      }
      if (best) {
        const vars = { ctx: fmt.num(best.ctx), now: fmt.num(now), need: fmt.gib(best.need), free: fmt.gib(measured) };
        out.push({
          impact: best.lost ? 1 : 2, title: t("tun_sug_ctx_up", vars),
          detail: best.lost
            ? t("tun_sug_ctx_up_trade", { ...vars, lost: fmt.num(best.lost), pct: fmt.pct(best.lost / geo.moe_cache_size, 0), hit: hit != null ? fmt.pct(hit) : "—" })
            : t("tun_sug_ctx_up_free", vars) + (out.some((s) => s.flag === "--moe-cache-size") ? " " + t("tun_sug_ctx_up_or") : ""),
          flag: "--kv-reserve-tokens",
          set: [{ flag: "--kv-reserve-tokens", value: String(best.ctx) }, { flag: "--max-seq-len-override", value: String(best.ctx) }],
        });
      }
    }
    if (collect && ranks.length > 1) {
      const rate = ranks.map((r) => {
        const ls = layers.filter((l) => l.rank === r.rank);
        const a = ls.reduce((s, l) => s + l.active, 0), m = ls.reduce((s, l) => s + l.miss, 0);
        return a ? m / a : null;
      });
      const valid = rate.filter((x) => x != null);
      if (valid.length > 1) {
        const hi = Math.max(...valid), lo = Math.min(...valid);
        if (hi > 0.02 && hi > lo * 1.3) {
          const worst = rate.indexOf(hi);
          out.push({
            impact: 1, title: t("tun_sug_pp", { gpu: ranks[worst].gpu?.index ?? worst }),
            detail: t("tun_sug_pp_d", { rates: valid.map((x) => fmt.pct(x)).join(FTI18N.lang === "ja" ? " と " : ", ") }),
            flag: "--pp-layers",  // the split itself is a judgement call: no automatic value
          });
        }
      }
    }
    out.sort((a, b) => b.impact - a.impact);

    const state = !isMoe ? "not_moe" : !decodeTok ? "no_generation"
      : decodeTok && hybrid && hit != null && hit < 0.5 && !out.some((s) => s.impact >= 2) ? "cpu_bound"
      : out.some((s) => s.impact >= 3) ? "slow" : out.some((s) => s.impact === 2) ? "some"
      : !collect ? "no_stats" : "fine";
    return { state, facts, suggestions: out };
  }

  // the headline for a state, as {kind, title, body}
  function verdict(state, facts) {
    const { fmt, t } = FT;
    switch (state) {
      case "no_data": return { kind: "info", title: t("tun_no_values"), body: t("tun_no_values_b") };
      case "not_moe": return { kind: "info", title: t("tun_no_moe"), body: t("tun_no_moe_b") };
      case "no_generation": return { kind: "info", title: t("tun_no_gen"), body: t("tun_no_gen_b", { win: t("tun_win_5_short") }) };
      case "cpu_bound": return { kind: "info", title: t("tun_cpu_bound"), body: t("tun_cpu_bound_b", { pct: fmt.pct(1 - facts.hit, 0) }) };
      case "slow": return { kind: "warn", title: t("tun_slow"), body: "" };
      case "some": return { kind: "info", title: t("tun_some"), body: "" };
      case "no_stats": return { kind: "info", title: t("tun_no_stats"), body: t("tun_no_stats_b") };
      default: return { kind: "ok", title: t("tun_fine"), body: t("tun_fine_b", { miss: fmt.pct(1 - (facts.hit ?? 1)), n: (facts.faultsPerTok ?? 0).toFixed(1) }) };
    }
  }

  return { load, compute, verdict };
})();
