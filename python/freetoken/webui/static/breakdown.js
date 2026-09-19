// Dashboard cards fed by --moe-collect-stats: where a decode step's time goes (utils/decode_sample.py,
// through /v1/stats kai.decode_sample) and how the GPU expert cache would do at other sizes
// (webui/slot_estimate.py, through /v1/kai/slots).
"use strict";

const FTBreakdown = (() => {
  const { fmt, esc, t } = FT;
  const PARTS = [
    ["fetch", "--c-h2d"], ["cpu", "--c-ssd"], ["gpu_experts", "--c-moe"], ["route", "--c-mamba"], ["other", "--c-other"],
  ];
  const gpuOf = (gpus, rank) => (gpus || []).find((g) => g.rank === rank)?.index ?? rank;
  const msv = (v) => (v >= 10 ? Math.round(v) : v.toFixed(1)) + " ms";

  // ---------------------------------------------------------------- per-token breakdown
  function renderTokens(root, kai, decodeTps) {
    const samples = kai?.decode_sample;
    if (!samples?.length) {
      root.innerHTML = `<div class="muted small">${esc(t(kai?.moe && !kai.moe.collect_stats ? "tok_off" : "tok_none"))}</div>`;
      return;
    }
    // compare like with like: under --spec-mtp a step yields several tokens
    const perStep = kai.spec?.tokens_per_step || 1;
    const real = decodeTps > 0 ? 1000 * perStep / decodeTps : null;
    const many = samples.length > 1;
    const rows = samples.map((s) => {
      const tot = Math.max(1e-6, s.total_ms);
      const seg = PARTS.map(([k, c]) => {
        const v = s.ms[k] || 0;
        return v > 0 ? `<div style="width:${(100 * v / tot).toFixed(2)}%;background:var(${c})" title="${esc(t("tok_part_" + k))} ${esc(msv(v))}"></div>` : "";
      }).join("");
      const parts = PARTS.filter(([k]) => (s.ms[k] || 0) >= 0.05 * tot)
        .map(([k]) => `${t("tok_part_" + k)} ${msv(s.ms[k])} (${fmt.pct(s.ms[k] / tot, 0)})`).join(" · ");
      return `<div>
        <div class="small" style="display:flex;gap:8px">${many ? `<b>${esc(t("tok_gpu", { gpu: gpuOf(kai.gpus, s.rank) }))}</b>` : ""}<span class="grow"></span>
          <span class="muted">${esc(t("tok_step", { v: msv(s.total_ms), n: s.samples }))}</span></div>
        <div class="bar">${seg}</div>
        <div class="small muted" style="margin-top:4px">${esc(parts)}</div>
      </div>`;
    });
    const worst = advice(samples);
    root.innerHTML = `<div class="stack">${rows.join("")}</div>
      <div class="legend one-line">${PARTS.map(([k, c]) => `<span><span class="sw" style="background:var(${c})"></span><span>${esc(t("tok_part_" + k))}</span></span>`).join("")}</div>
      ${real ? `<div class="small muted" style="margin-top:6px">${esc(t(perStep > 1 ? "tok_real_spec" : "tok_real", { v: msv(real), n: perStep.toFixed(2) }))}</div>` : ""}
      ${worst ? `<p class="hint" style="margin:8px 0 0">${esc(worst)}</p>` : ""}`;
  }

  function advice(samples) {
    // the slowest rank sets the pace under --pp-size; judge on it
    const s = samples.reduce((a, b) => (b.total_ms > a.total_ms ? b : a));
    const tot = s.total_ms || 1, share = (k) => (s.ms[k] || 0) / tot;
    const top = ["fetch", "cpu", "gpu_experts", "other"].reduce((a, b) => (share(b) > share(a) ? b : a));
    if (top === "fetch" && share("fetch") >= 0.25) return t("tok_adv_fetch");
    if (top === "cpu" && share("cpu") >= 0.25) return t("tok_adv_cpu");
    if (top === "other" && share("other") >= 0.5) return t("tok_adv_other");
    return null;
  }

  // ---------------------------------------------------------------- cache size estimate
  function renderSlots(root, doc, kai) {
    const ranks = doc?.ranks || [];
    if (!ranks.length) {
      root.innerHTML = `<div class="muted small">${esc(t(kai?.moe && !kai.moe.collect_stats ? "tok_off" : "slot_none"))}</div>`;
      return;
    }
    const many = ranks.length > 1;
    root.innerHTML = ranks.map((r, i) => `<div style="${i ? "margin-top:14px" : ""}">
        ${many ? `<div class="small"><b>${esc(t("tok_gpu", { gpu: gpuOf(kai?.gpus, r.rank) }))}</b></div>` : ""}
        <canvas data-rank="${i}" style="display:block;width:100%;height:150px"></canvas>
        <div class="small muted">${esc(summary(r))}</div>
      </div>`).join("")
      + `<div class="legend one-line"><span><span class="sw" style="background:var(--c-h2d)"></span><span>${esc(t("slot_lg_est"))}</span></span>`
      + `<span><span class="sw" style="background:var(--c-ssd)"></span><span>${esc(t("slot_lg_now"))}</span></span>`
      + (ranks.some((r) => r.measured_hit_60s != null) ? `<span><span class="sw" style="background:var(--fg)"></span><span>${esc(t("slot_lg_meas"))}</span></span>` : "")
      + `</div><p class="hint" style="margin:8px 0 0">${esc(t("slot_hint", { steps: fmt.num(Math.min(...ranks.map((r) => r.steps))) }))}${ranks.some((r) => r.decode_target === "hybrid") ? " " + esc(t("slot_hybrid")) : ""}</p>`;
    // after layout: a canvas inserted this frame has no width to draw into yet
    requestAnimationFrame(() => root.querySelectorAll("canvas").forEach((cv) => draw(cv, ranks[+cv.dataset.rank])));
    lastSlots = [root, doc, kai];
  }

  let lastSlots = null;
  addEventListener("resize", () => { if (lastSlots) renderSlots(...lastSlots); });

  const gibOf = (r, slots) => (r.bytes_per_slot ? fmt.gib(slots * r.bytes_per_slot) : t("slot_n", { n: fmt.num(slots) }));

  function at(r, slots) {
    // the curve's points are exact hit rates; between them, read the next point up (conservative)
    const p = r.curve.find((c) => c.slots >= slots) || r.curve[r.curve.length - 1];
    return p.hit;
  }

  function summary(r) {
    const now = r.cache_size;
    const parts = [t(r.measured_hit_60s != null ? "slot_now_meas" : "slot_now", {
      n: fmt.num(now), size: gibOf(r, now), est: fmt.pct(r.hit_at_current), meas: fmt.pct(r.measured_hit_60s),
    })];
    for (const x of [1.5, 2]) {
      const s = Math.round(now * x);
      if (s <= r.capacity) parts.push(t("slot_times", { x, size: gibOf(r, s), est: fmt.pct(at(r, s)) }));
    }
    parts.push(t("slot_all", { size: gibOf(r, r.capacity) }));
    return parts.join(" · ");
  }

  function draw(cv, r) {
    const ctx = cv.getContext("2d");
    const w = cv.clientWidth, h = 150, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
    const L = 40, R = 10, T = 8, B = 22;
    const X = (s) => L + (w - L - R) * s / r.capacity, Y = (v) => T + (h - T - B) * (1 - v);
    ctx.font = "11px system-ui"; ctx.strokeStyle = col("--line"); ctx.fillStyle = col("--fg3"); ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(fmt.pct(v, 0), L - 6, Y(v) + 4);
    }
    ctx.textAlign = "center";
    for (const f of [0, 0.5, 1]) {
      const s = r.capacity * f;
      ctx.fillText(r.bytes_per_slot ? fmt.gib(s * r.bytes_per_slot) : fmt.num(s), Math.min(w - R - 20, Math.max(L + 16, X(s))), h - 6);
    }
    ctx.strokeStyle = col("--c-h2d"); ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    for (const p of r.curve) ctx.lineTo(X(p.slots), Y(p.hit));
    ctx.stroke();
    ctx.strokeStyle = col("--c-ssd"); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(r.cache_size), T); ctx.lineTo(X(r.cache_size), h - B); ctx.stroke();
    ctx.setLineDash([]);
    if (r.measured_hit_60s != null) {
      ctx.fillStyle = col("--fg");
      ctx.beginPath(); ctx.arc(X(r.cache_size), Y(r.measured_hit_60s), 3.5, 0, 2 * Math.PI); ctx.fill();
    }
  }

  return { renderTokens, renderSlots };
})();
