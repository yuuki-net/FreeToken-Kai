// Where the experts a token needs are not on the GPU: one cell per MoE layer (from
// --moe-collect-stats), and, when the routing counts exist (--moe-stats-out), one row per layer
// of how often each expert was picked, with what an ideally filled cache of today's size would
// have held. Both say what to change, not only what happened.
"use strict";

const FTHeatmap = (() => {
  let mode = "sorted";  // per-expert grid: "sorted" (most picked first) or "id"
  let metric = "miss";  // layer strip: "miss" or "cpu" (hybrid only)
  let last = null;

  function hex(c) {
    const m = /^#?([0-9a-f]{6})$/i.exec(c.trim());
    if (!m) return [128, 128, 128];
    const n = parseInt(m[1], 16);
    return [n >> 16, (n >> 8) & 255, n & 255];
  }
  const mix = (a, b, f) => a.map((x, i) => Math.round(x + (b[i] - x) * f));

  // layers with their global index, rank by rank
  function layerRows(doc) {
    const start = new Map((doc.ranks || []).map((r) => [r.rank, (r.layer_range || [0])[0] || 0]));
    return (doc.layers || []).filter((l) => !l.mtp).map((l) => ({ ...l, index: (start.get(l.rank) || 0) + l.local }));
  }

  function renderStrip(doc) {
    const { esc, fmt, t } = FT;
    const rows = layerRows(doc);
    const ranks = doc.ranks || [];
    const collect = ranks.some((r) => r.moe?.collect_stats);
    const hybrid = ranks.some((r) => r.moe?.decode_target === "hybrid");
    if (!ranks.some((r) => r.moe)) return `<div class="empty small">${esc(t("heat_not_moe"))}</div>`;
    if (!collect) return `<div class="empty small">${esc(t("heat_needs_collect"))}</div>`;
    const A = rows.reduce((s, l) => s + l.active, 0);
    if (!A) return `<div class="empty small">${esc(t("heat_no_gen"))}</div>`;
    if (!hybrid) metric = "miss";

    const value = (l) => !l.active ? null
      : metric === "cpu" ? Math.max(0, l.miss - (l.fetched || 0)) / l.active : l.miss / l.active;
    // 0-100% would paint a 5% layer and a 10% layer the same pale: scale to the worst layer
    const top = Math.max(0, ...rows.map((l) => value(l) ?? 0));
    const scale = Math.min(1, Math.max(0.2, Math.ceil(top * 10) / 10));
    const byRank = new Map();
    for (const l of rows) { if (!byRank.has(l.rank)) byRank.set(l.rank, []); byRank.get(l.rank).push(l); }
    const groups = [...byRank.entries()].map(([rank, ls]) => {
      const gpu = ranks.find((r) => r.rank === rank)?.gpu;
      const label = byRank.size > 1 ? `<div class="small muted">${esc(t("heat_gpu", { gpu: gpu?.index ?? rank, from: ls[0].index, to: ls[ls.length - 1].index }))}</div>` : "";
      const cells = ls.map((l) => {
        const v = value(l);
        const tip = t("heat_cell", { layer: l.index, hit: fmt.pct(l.active ? 1 - l.miss / l.active : null),
          cpu: hybrid && l.active ? fmt.pct(Math.max(0, l.miss - (l.fetched || 0)) / l.active) : "—",
          per: l.calls ? (l.active / l.calls).toFixed(1) : "—" });
        const bg = v == null ? "var(--tile)" : `color-mix(in srgb, var(--c-ssd) ${Math.round(Math.min(1, v / scale) * 100)}%, var(--tile))`;
        return `<div class="heat-cell" title="${esc(tip)}" style="background:${bg}"></div>`;
      }).join("");
      const ticks = ls.map((l, i) => `<span style="flex:1;min-width:0">${i === 0 || l.index % 10 === 0 ? l.index : ""}</span>`).join("");
      return `<div style="flex:${ls.length};min-width:0">${label}<div class="heat-strip">${cells}</div><div class="heat-ticks">${ticks}</div></div>`;
    }).join("");

    const vals = rows.filter((l) => l.active).map((l) => ({ index: l.index, v: value(l) }));
    const worst = [...vals].sort((a, b) => b.v - a.v).slice(0, 3);
    const best = [...vals].sort((a, b) => a.v - b.v)[0];
    const mean = metric === "cpu" ? rows.reduce((s, l) => s + Math.max(0, l.miss - (l.fetched || 0)), 0) / A
      : rows.reduce((s, l) => s + l.miss, 0) / A;
    const spread = worst.length && best ? worst[0].v - best.v : 0;
    let advice;
    if (metric === "cpu") advice = t("heat_adv_cpu");
    else if (byRank.size > 1) {
      const rate = [...byRank.values()].map((ls) => ls.reduce((s, l) => s + l.miss, 0) / Math.max(1, ls.reduce((s, l) => s + l.active, 0)));
      advice = Math.max(...rate) > Math.min(...rate) * 1.3 && Math.max(...rate) > 0.02 ? t("heat_adv_pp") : t("heat_adv_flat");
    } else advice = mean < 0.03 ? t("heat_adv_fine") : spread > 0.25 ? t("heat_adv_uneven") : t("heat_adv_flat");

    const toggle = hybrid ? `<div class="seg small" style="margin-bottom:8px">
        <button type="button" data-metric="miss" class="${metric === "miss" ? "on" : ""}">${esc(t("heat_metric_miss"))}</button>
        <button type="button" data-metric="cpu" class="${metric === "cpu" ? "on" : ""}">${esc(t("heat_metric_cpu"))}</button></div>` : "";
    return `${toggle}
      <div style="display:flex;gap:10px;align-items:flex-end">${groups}</div>
      <div class="legend" style="align-items:center"><span>0%</span><span class="heat-scale"></span><span>${fmt.pct(scale, 0)}</span>
        <span class="muted">${esc(t(metric === "cpu" ? "heat_legend_cpu" : "heat_legend_miss"))}</span></div>
      <div class="small" style="margin-top:8px"><b>${esc(t(metric === "cpu" ? "heat_mean_cpu" : "heat_mean_miss", { v: fmt.pct(mean) }))}</b>
        ${worst.length ? " · " + esc(t("heat_worst", { list: worst.map((w) => `${w.index} (${fmt.pct(w.v, 0)})`).join(", "), best: `${best.index} (${fmt.pct(best.v, 0)})` })) : ""}</div>
      <div class="small muted" style="margin-top:4px">${esc(advice)}</div>`;
  }

  // what a cache of `slots` per layer holds when filled with the most picked experts
  function oracle(rows, slots) {
    let sum = 0, n = 0;
    for (const f of rows) {
      const tot = f.reduce((a, b) => a + b, 0);
      if (!tot) continue;
      const s = [...f].sort((a, b) => b - a);
      sum += s.slice(0, Math.max(1, Math.round(slots))).reduce((a, b) => a + b, 0) / tot; n++;
    }
    return n ? sum / n : null;
  }
  function cover(rows, share) {
    let sum = 0, n = 0;
    for (const f of rows) {
      const tot = f.reduce((a, b) => a + b, 0);
      if (!tot) continue;
      const s = [...f].sort((a, b) => b - a);
      let acc = 0, k = 0;
      while (k < s.length && acc < share * tot) acc += s[k++];
      sum += k; n++;
    }
    return n ? sum / n : null;
  }

  function renderExperts(doc) {
    const { esc, fmt, t } = FT;
    const freq = doc.expert_freq;
    if (!freq?.length) return `<div class="small muted">${esc(t("heat_freq_missing"))}</div>`;
    const rows = freq.flatMap((r) => r.freq);
    const total = rows.reduce((s, f) => s + f.reduce((a, b) => a + b, 0), 0);
    if (!total) return `<div class="small muted">${esc(t("heat_no_gen"))}</div>`;
    const ranks = doc.ranks || [];
    const slots = ranks.reduce((s, r) => s + (r.moe?.cache_size || 0), 0) / Math.max(1, rows.length);
    const layers = layerRows(doc);
    const A = layers.reduce((s, l) => s + l.active, 0), M = layers.reduce((s, l) => s + l.miss, 0);
    const hit = A ? 1 - M / A : null;
    const o1 = oracle(rows, slots), o2 = oracle(rows, slots * 2), n90 = cover(rows, 0.9);
    const verdict = o2 - o1 >= 0.1 ? t("heat_freq_more_slots")
      : o1 >= 0.8 ? t("heat_freq_fits")
      : hit != null && o1 - hit >= 0.1 ? t("heat_freq_policy") : t("heat_freq_flat");
    return `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <div class="seg small">
          <button type="button" data-mode="sorted" class="${mode === "sorted" ? "on" : ""}">${esc(t("heat_sorted"))}</button>
          <button type="button" data-mode="id" class="${mode === "id" ? "on" : ""}">${esc(t("heat_by_id"))}</button></div>
        <span class="small muted">${esc(t("heat_freq_axes", { layers: rows.length, experts: rows[0].length }))}</span></div>
      <canvas id="heat-canvas" style="width:100%;margin-top:8px;display:block"></canvas>
      ${mode === "sorted" && slots ? `<div class="small muted">${esc(t("heat_slot_line", { n: slots.toFixed(0) }))}</div>` : ""}
      <div class="small" style="margin-top:8px"><b>${esc(verdict)}</b></div>
      <div class="small muted" style="margin-top:4px">${esc(t("heat_freq_detail", {
        slots: slots.toFixed(0), o1: fmt.pct(o1), hit: hit != null ? fmt.pct(hit) : "—", o2: fmt.pct(o2), n90: n90 != null ? n90.toFixed(0) : "—" }))}</div>`;
  }

  function drawExperts(root, doc) {
    const cv = root.querySelector("#heat-canvas");
    if (!cv) return;
    const rows = doc.expert_freq.flatMap((r) => r.freq);
    const E = rows[0].length, L = rows.length;
    const w = cv.clientWidth, rowH = Math.max(3, Math.min(8, Math.floor(320 / L))), h = rowH * L;
    const dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const lo = hex(css.getPropertyValue("--tile")), hi = hex(css.getPropertyValue("--c-ssd"));
    const cw = w / E;
    rows.forEach((f, y) => {
      const s = mode === "sorted" ? [...f].sort((a, b) => b - a) : f;
      const max = Math.max(1, ...s);
      s.forEach((v, x) => {
        const [r, g, b] = mix(lo, hi, Math.sqrt(v / max));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * cw, y * rowH, Math.ceil(cw), rowH);
      });
    });
    const ranks = doc.ranks || [];
    const slots = ranks.reduce((s, r) => s + (r.moe?.cache_size || 0), 0) / Math.max(1, L);
    if (mode === "sorted" && slots) {
      ctx.strokeStyle = css.getPropertyValue("--accent").trim();
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(slots * cw, 0); ctx.lineTo(slots * cw, h); ctx.stroke();
    }
  }

  function render(root, doc) {
    last = { root, doc };
    root.innerHTML = `<div>${renderStrip(doc)}</div>
      <h3 class="small" style="margin:16px 0 6px">${FT.esc(FT.t("heat_experts_title"))}</h3>
      <div>${renderExperts(doc)}</div>`;
    if (doc.expert_freq?.length) drawExperts(root, doc);
    root.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { mode = b.dataset.mode; render(root, doc); }));
    root.querySelectorAll("[data-metric]").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.metric; render(root, doc); }));
  }

  addEventListener("resize", () => { if (last?.doc.expert_freq?.length) drawExperts(last.root, last.doc); });

  return { render, oracle, cover };
})();
