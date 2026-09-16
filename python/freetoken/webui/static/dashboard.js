"use strict";

(async () => {
  const { fmt, esc, $, homePath, modelName, t } = FT;
  await FT.init();
  FT.header("dash");
  const managed = FT.mode === "mgr";

  const hist = FT.demoHistory ? FT.demoHistory() : []; // [t, decode, prefill], last 10 minutes
  let geometry = null, reqCursor = 0, requests = [], engineConfig = null, lastStatsHost = false;
  let engineRunning = null;  // the manager's own engine: drives which buttons a profile row shows

  // ---------------------------------------------------------------- engine + tiles
  async function tick() {
    let health;
    try { health = await FT.serveGet("/health"); }
    catch (e) { FT.setState("bad", t(managed ? "dash_no_daemon" : "dash_no_serve")); return; }

    const external = !!health.external;
    const running = managed ? !!health.running || external : true;
    const reachable = health.reachable !== false;
    const status = !running ? "stopped" : !reachable ? "unreachable" : health.status;

    $("#eng-model").textContent = modelName(health.model || engineConfig?.model) || t("none");
    const label = {
      ok: ["ok", t("dash_running")], loading: ["warn", t("dash_loading")], error: ["bad", t("dash_error")],
      stopped: ["mute", t("dash_stopped")], unreachable: ["warn", t("dash_starting")],
    }[status] || ["mute", status || t("none")];
    if (external && status === "ok") label[1] = t("dash_external");
    FT.setState(label[0], label[1]);
    $("#eng-state").innerHTML = `<span class="pill ${label[0]}"><span class="dot"></span>${esc(label[1])}</span>`;
    $("#eng-uptime").textContent = status === "ok" ? fmt.dur(health.uptime_s) : t("none");

    // the state pill already says "loading": a bar only when the serve reports bytes to count
    const { done_bytes: done = 0, total_bytes: tot = 0 } = health.progress || {};
    const load = $("#eng-load");
    load.hidden = !(status === "loading" && tot > 0);
    if (!load.hidden) {
      $("#eng-load-bar").style.width = Math.min(100, 100 * done / tot).toFixed(1) + "%";
      $("#eng-load-l").textContent = t("dash_load_bytes", { done: fmt.gib(done), total: fmt.gib(tot), pct: fmt.pct(done / tot, 0) });
    }
    const msg = $("#eng-msg");
    msg.hidden = !(external || status === "error" || (status === "stopped" && health.last_exit_code != null));
    if (status === "error") msg.innerHTML = `<span style="color:var(--bad)">${esc(homePath(health.message))}</span>`;
    else if (external) msg.innerHTML = `<span class="muted">${esc(t("dash_external_note", { port: health.port }))}</span>`;
    else if (!msg.hidden) msg.innerHTML = `<span class="muted">${esc(t("dash_last_exit", { code: health.last_exit_code }))}</span>`;

    if (managed) {
      const nowRunning = !!health.running && !external;
      if (nowRunning !== engineRunning) {
        engineRunning = nowRunning;
        loadConfig().then(loadProfiles);
      }
    }
    if (status !== "ok") { lastStatsHost = false; return; }

    let stats;
    try { stats = await FT.serveGet("/v1/stats"); } catch { lastStatsHost = false; return; }
    lastStatsHost = !!stats.host;
    if (stats.host) renderRam(stats.host);
    const tp = stats.throughput || {};
    $("#t-dec").textContent = (tp.decode_tps ?? 0).toFixed(1);
    $("#t-pre").textContent = fmt.num(tp.prefill_tps ?? 0);
    const kai = stats.kai || {};
    $("#t-dec-s").textContent = kai.spec?.accept_rate != null ? t("dash_accept_rate", { rate: fmt.pct(kai.spec.accept_rate, 0) })
      : kai.moe?.gpu_hit_rate != null ? t("dash_hit_rate", { rate: fmt.pct(kai.moe.gpu_hit_rate) }) : " ";
    $("#t-pre-s").textContent = kai.prefill?.chunk ? t(kai.prefill.auto ? "dash_chunk_auto" : "dash_chunk", { n: kai.prefill.chunk }) : " ";
    if (stats.kv) {
      const ps = stats.kv.page_size || 1;
      $("#t-kv").textContent = fmt.tok(stats.kv.used_pages * ps);
      $("#t-kv-tot").textContent = t("dash_tokens_of", { total: fmt.tok(stats.kv.total_pages * ps) });
    }
    $("#t-kv-s").textContent = stats.mamba ? t("dash_gdn_slots", { used: stats.mamba.used_slots, total: stats.mamba.total_slots }) : " ";
    const rq = stats.requests || {};
    $("#t-req").textContent = rq.active ?? t("none");
    $("#t-req-s").textContent = rq.ttft_mean_ms ? t("dash_ttft_mean", { ms: fmt.ms(rq.ttft_mean_ms) }) : t("dash_completed", { n: fmt.num(rq.completed) });

    const now = Date.now();
    hist.push([now, tp.decode_tps || 0, tp.prefill_tps || 0]);
    while (hist.length && hist[0][0] < now - 600e3) hist.shift();
    drawChart();
    renderGpus(stats);
    renderKai(stats);
  }

  // ---------------------------------------------------------------- speed chart
  function drawChart() {
    const cv = $("#tps-chart"), ctx = cv.getContext("2d");
    const w = cv.clientWidth, h = 160, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const col = (n) => css.getPropertyValue(n).trim();
    const L = 36, R = 44, T = 8, B = 20, now = Date.now();
    const maxD = Math.max(5, ...hist.map((p) => p[1])) * 1.15, maxP = Math.max(50, ...hist.map((p) => p[2])) * 1.15;
    ctx.font = "11px system-ui"; ctx.fillStyle = col("--fg3"); ctx.strokeStyle = col("--line");
    for (let i = 0; i <= 2; i++) {
      const y = T + (h - T - B) * i / 2;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText((maxD * (1 - i / 2)).toFixed(0), L - 6, y + 4);
      ctx.textAlign = "left"; ctx.fillText(fmt.num(maxP * (1 - i / 2)), w - R + 6, y + 4);
    }
    ctx.textAlign = "center";
    [t("dash_t_10min"), t("dash_t_5min"), t("dash_t_now")].forEach((s, i) => ctx.fillText(s, L + (w - L - R) * i / 2, h - 4));
    const x = (ts) => L + (w - L - R) * (1 - (now - ts) / 600e3);
    const line = (idx, max, c) => {
      ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
      hist.forEach((p, i) => { const y = T + (h - T - B) * (1 - p[idx] / max); i ? ctx.lineTo(x(p[0]), y) : ctx.moveTo(x(p[0]), y); });
      ctx.stroke();
    };
    line(2, maxP, col("--c-ssd")); line(1, maxD, col("--c-h2d"));
    ctx.lineWidth = 1;
  }

  // ---------------------------------------------------------------- GPUs
  function renderGpus(stats) {
    const root = $("#gpus");
    const kaiGpus = stats.kai?.gpus;
    let rows;
    const geo = geometry || {}, ub = geo.unit_bytes || {};
    const kv = (geo.num_pages || 0) * (geo.page_size || 1) * (ub.kv_per_token || 0);
    const mamba = (geo.num_mamba_slots || 0) * (ub.mamba_per_slot || 0);
    const moe = (geo.moe_cache_size || 0) * (ub.moe_per_expert || 0);
    if (kaiGpus?.length) {
      // pools are known for rank 0's allocation; under pp the other ranks size theirs the same way
      rows = kaiGpus.map((g) => ({ ...g, pools: g.rank === 0 ? { kv, moe, mamba } : {} }));
      $("#gpu-hint").textContent = t(kaiGpus.length > 1 ? "dash_gpu_hint_multi" : "dash_gpu_hint_measured");
    } else {
      const g = (stats.gpus || [])[0];
      if (!g) { root.innerHTML = `<div class="muted small">${esc(t("dash_gpu_none"))}</div>`; return; }
      rows = [{ index: g.index, name: g.name, total_bytes: g.total_bytes, reserved_bytes: stats.vram_bytes, pools: { kv, moe, mamba } }];
      $("#gpu-hint").textContent = t((stats.gpus || []).length > 1 || geo.num_pages == null ? "dash_gpu_hint_first" : "dash_gpu_hint_calc");
    }
    $("#lg-engine").hidden = !rows.some((g) => g.pools && !Object.keys(g.pools).length);
    root.innerHTML = rows.map((g) => {
      const tot = g.total_bytes || 1, p = g.pools || {};
      const known = (p.kv || 0) + (p.moe || 0) + (p.mamba || 0);
      const weights = Math.max(0, (g.reserved_bytes || 0) - known);
      const other = g.used_bytes != null ? Math.max(0, g.used_bytes - (g.reserved_bytes || 0)) : 0;
      const used = g.used_bytes ?? g.reserved_bytes ?? 0;
      const noSplit = g.pools && !Object.keys(g.pools).length;
      const seg = (v, c) => (v > 0 ? `<div style="width:${(100 * v / tot).toFixed(2)}%;background:var(${c})"></div>` : "");
      const layers = g.layers ? t("dash_gpu_layers", { from: g.layers[0], to: g.layers[1] }) : "";
      return `<div>
        <div class="small" style="display:flex;gap:8px"><b>GPU ${esc(g.index)} · ${esc(g.name)}${esc(layers)}</b><span class="grow"></span>
        <span class="muted">${esc(t("dash_gpu_used", { used: fmt.gib(used), total: fmt.gib(g.total_bytes), free: fmt.gib(tot - used) }))}</span></div>
        <div class="bar">${seg(weights, noSplit ? "--c-engine" : "--c-weights")}${seg(p.kv, "--c-kv")}${seg(p.moe, "--c-moe")}${seg(p.mamba, "--c-mamba")}${seg(other, "--c-other")}</div>
      </div>`;
    }).join("");
  }

  // ---------------------------------------------------------------- host RAM
  // used / reclaimable cache / free add up to the total; the engine's PSS is shown beside the bar,
  // not in it (its mapped checkpoint pages are also counted as cache).
  function renderRam(m) {
    const root = $("#ram"), legend = $("#ram-legend");
    if (!m || !m.total_bytes) { root.innerHTML = ""; legend.hidden = true; return; }
    const tot = m.total_bytes;
    const seg = (v, c) => (v > 0 ? `<div style="width:${(100 * v / tot).toFixed(2)}%;background:var(${c})"></div>` : "");
    const extra = [
      m.engine_bytes != null ? t("dash_ram_engine", { v: fmt.gib(m.engine_bytes) }) : null,
      m.swap_total_bytes ? t("dash_ram_swap", { v: fmt.gib(m.swap_used_bytes), total: fmt.gib(m.swap_total_bytes) }) : null,
    ].filter(Boolean).join(" · ");
    root.innerHTML = `<div class="small" style="display:flex;gap:8px;flex-wrap:wrap"><b>RAM</b><span class="grow"></span>
        <span class="muted">${esc(t("dash_ram_used", { used: fmt.gib(m.used_bytes), total: fmt.gib(tot), avail: fmt.gib(tot - m.used_bytes) }))}</span></div>
      <div class="bar">${seg(m.used_bytes, "--c-ram-used")}${seg(m.reclaimable_bytes, "--c-ram-cache")}</div>
      ${extra ? `<div class="small muted" style="margin-top:4px">${esc(extra)}</div>` : ""}`;
    legend.hidden = false;
  }

  async function pollHost() {
    // the manager answers even with no engine: RAM stays visible while the server is stopped
    if (!managed) return;
    if (lastStatsHost) return;  // the running server already reported it this tick
    try { renderRam((await FT.raw("/host")).memory); } catch {}
  }

  // ---------------------------------------------------------------- kai block
  function renderKai(stats) {
    const k = stats.kai, geo = geometry || {}, rows = [];
    const add = (label, value, cls = "") => rows.push(`<div class="kv"><span class="k">${esc(label)}</span><span class="${cls}">${value}</span></div>`);
    if (k?.moe) {
      add(t("dash_kai_hit"), k.moe.gpu_hit_rate != null ? fmt.pct(k.moe.gpu_hit_rate)
        : `<span class="muted">${esc(t(k.moe.collect_stats ? "dash_kai_no_gen" : "dash_kai_needs_flag"))}</span>`);
      if (k.moe.major_faults_per_min != null)
        add(t("dash_kai_faults"), esc(t("dash_per_min", { n: fmt.num(k.moe.major_faults_per_min) })), k.moe.major_faults_per_min > 100 ? "warn-text" : "");
      if (k.moe.ssd_read_bytes_per_s) add(t("dash_kai_ssd"), esc(t("dash_per_sec", { v: fmt.gib(k.moe.ssd_read_bytes_per_s) })));
    }
    if (k?.spec?.tokens_per_step != null)
      add(t("dash_kai_mtp"), esc(t("dash_kai_mtp_v", { n: k.spec.tokens_per_step.toFixed(2), max: k.spec.k + 1 })));
    if (geo.num_experts && geo.num_moe_layers && geo.moe_cache_size) {
      const all = geo.num_experts * geo.num_moe_layers;
      add(t("dash_kai_slots"), esc(t("dash_kai_slots_v", { n: fmt.num(geo.moe_cache_size), all: fmt.num(all), pct: fmt.pct(geo.moe_cache_size / all, 0) })));
    }
    if (k?.kv_cache_dtype) add(t("dash_kai_kv_dtype"), esc(k.kv_cache_dtype));
    if (k?.prefix_reuse_rate != null) add(t("dash_kai_reuse"), fmt.pct(k.prefix_reuse_rate, 0));
    if (k?.window_s) rows.push(`<div class="small muted" style="padding-top:6px">${esc(t("dash_kai_window", { n: Math.round(k.window_s) }))}</div>`);
    if (!k) rows.push(`<div class="small muted" style="padding-top:6px">${esc(t("dash_kai_missing"))}</div>`);
    $("#kai").innerHTML = rows.join("") || `<div class="muted small">${esc(t("dash_not_moe"))}</div>`;
  }

  // ---------------------------------------------------------------- requests
  async function pollRequests() {
    try {
      const doc = await FT.serveGet("/v1/requests", `?since=${reqCursor}&limit=100`);
      reqCursor = doc.next_cursor ?? reqCursor;
      requests = [...requests, ...(doc.entries || [])].slice(-20);
    } catch { return; }
    const body = $("#reqs");
    if (!requests.length) return;
    body.innerHTML = requests.slice().reverse().slice(0, 10).map((r) => {
      // generation only when the first token's time is known (streamed requests); otherwise the
      // whole request, which includes prompt processing, marked so it is not read as decode speed
      const out = r.completion_tokens || 0;
      let tps = t("none"), title = "";
      if (r.ttft_ms != null && out > 1 && r.duration_ms > r.ttft_ms) {
        tps = ((out - 1) / ((r.duration_ms - r.ttft_ms) / 1000)).toFixed(1);
        if (r.prompt_tokens && r.ttft_ms > 0) title = t("dash_tps_prompt", { v: fmt.num(r.prompt_tokens / (r.ttft_ms / 1000)) });
      } else if (out > 0 && r.duration_ms > 0) {
        tps = `${(out / (r.duration_ms / 1000)).toFixed(1)}*`;
        title = t("dash_tps_total_hint");
      }
      return `<tr>
      <td>${fmt.time(r.ts)}</td>
      <td>${esc(r.path.replace(/^\/v1\//, ""))}${r.status >= 400 ? ` <span class="pill bad">${r.status}</span>` : ""}</td>
      <td class="num">${fmt.num(r.prompt_tokens)}</td><td class="num">${fmt.num(r.completion_tokens)}</td>
      <td class="num">${fmt.ms(r.ttft_ms)}</td><td class="num">${fmt.ms(r.duration_ms)}</td>
      <td class="num" title="${esc(title)}">${esc(tps)}</td></tr>`;
    }).join("");
  }

  // ---------------------------------------------------------------- experts heatmap
  let heatBusy = false;
  async function pollHeat() {
    if (heatBusy || $("#eng-uptime").textContent === t("none")) return;
    heatBusy = true;
    try {
      const doc = await FT.serveGet("/v1/kai/experts", "?window=300&freq=true");
      const card = $("#heat-card");
      card.hidden = !(doc.ranks || []).some((r) => r.moe);
      if (!card.hidden) FTHeatmap.render($("#heat"), doc);
    } catch {} finally { heatBusy = false; }
  }

  async function pollGeometry() { try { geometry = (await FT.serveGet("/v1/cache/status")).geometry; } catch {} }

  // ---------------------------------------------------------------- ft mgr: lifecycle, profiles, logs
  async function lifecycle(label, fn) {
    const msg = $("#eng-msg");
    msg.hidden = false; msg.innerHTML = `<span class="muted">${esc(t("dash_lc_running", { label }))}</span>`;
    document.querySelectorAll("#profiles button").forEach((b) => (b.disabled = true));
    try { await fn(); msg.innerHTML = `<span class="muted">${esc(t("dash_lc_done", { label }))}</span>`; }
    catch (e) {
      const detail = e.body?.error || e.body?.detail || e.message;
      msg.innerHTML = `<span style="color:var(--bad)">${esc(t("dash_lc_failed", { label, detail: homePath(detail) }))}</span>`;
    } finally {
      await loadConfig(); await loadProfiles(); tick();
    }
  }

  // small inline icons: no icon font to fetch, so the page works offline
  const ICONS = {
    apply: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l12-7.5z"/></svg>',
    edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    del: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>',
    stop: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>',
    restart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>',
  };
  const iconBtn = (act, i, label, extra = "") =>
    `<button class="icon-btn ${act === "apply" ? "go" : ""} ${act === "del" || act === "stop" ? "danger" : ""} ${act === "stop" ? "stop" : ""}" data-act="${act}" data-i="${i}" title="${esc(label)}" aria-label="${esc(label)}" ${extra}>${ICONS[act]}</button>`;

  async function loadConfig() { try { engineConfig = await FT.raw("/engine/config"); } catch {} }

  async function loadProfiles() {
    let list = [];
    try { list = (await FT.raw("/profiles")).profiles || []; }
    catch { $("#profiles").innerHTML = `<div class="muted small">${esc(t("dash_profiles_failed"))}</div>`; return; }
    const cur = engineConfig ? JSON.stringify([engineConfig.model, engineConfig.port, engineConfig.args]) : "";
    $("#profiles").innerHTML = list.length ? list.map((p, i) => {
      // only while the manager's engine actually runs: its last config survives a stop
      const sameServer = !!engineRunning && engineConfig?.model === p.model && (engineConfig?.port ?? null) === (p.port ?? null);
      const inUse = sameServer && JSON.stringify(p.args) === JSON.stringify(engineConfig.args);
      // same model and port but different flags: the profile was edited after the server started
      const stale = sameServer && !inUse;
      // one quiet line, not the whole command: the flags belong in the editor
      const flags = (p.args || []).filter((a) => a.startsWith("--"));
      const shown = flags.slice(0, 3).map((f) => f.replace(/^--/, "")).join(" · ");
      const more = flags.length > 3 ? ` +${flags.length - 3}` : "";
      return `<div class="profile">
        <div style="flex:1;min-width:0">
          <div><b>${esc(p.name)}</b> ${inUse ? `<span class="pill info">${esc(t("dash_in_use"))}</span>` : ""}${stale ? `<span class="pill warn">${esc(t("dash_stale"))}</span>` : ""}</div>
          <div class="small muted" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
               title="${esc(homePath((p.args || []).join(" ")))}">${esc(modelName(p.model))} · ${esc(p.port ? t("dash_port", { port: p.port }) : t("dash_port_default"))}${flags.length ? ` · ${esc(shown + more)}` : ""}</div>
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;align-self:center" ${FT.canWrite ? "" : "hidden"}>
          ${sameServer
            ? iconBtn("stop", i, t("dash_stop")) + iconBtn("restart", i, t(stale ? "dash_restart_apply" : "dash_restart"))
            : iconBtn("apply", i, t("dash_apply_tip"))}
          ${iconBtn("edit", i, t("dash_edit"))}
          ${iconBtn("copy", i, t("dash_copy"))}
          ${iconBtn("del", i, t("dash_delete"))}
        </div></div>`;
    }).join("") : `<div class="empty small">${esc(t("dash_profiles_none"))}</div>`;
    $("#profiles").onclick = async (ev) => {
      const b = ev.target.closest("button[data-act]"); if (!b) return;
      const p = list[+b.dataset.i];
      if (b.dataset.act === "apply") {
        if (!await FT.ask(t("dash_confirm_apply", { name: p.name }), { title: t("dash_apply_tip"), ok: t("dash_ok_start") })) return;
        lifecycle(t("dash_lc_apply"), () => FT.raw("/engine/switch", { method: "POST", json: { model: p.model, port: p.port, args: p.args } }));
      } else if (b.dataset.act === "stop") {
        if (!await FT.ask(t("dash_confirm_stop"), { title: t("dash_stop"), ok: t("dash_stop"), danger: true })) return;
        lifecycle(t("dash_lc_stop"), () => FT.raw("/engine/stop", { method: "POST", json: {} }));
      } else if (b.dataset.act === "restart") {
        // restarting from the row uses the profile's flags, so an edited profile is applied here
        if (!await FT.ask(t("dash_confirm_restart"), { title: t("dash_restart"), ok: t("dash_restart") })) return;
        lifecycle(t("dash_lc_restart"), () => FT.raw("/engine/switch", { method: "POST", json: { model: p.model, port: p.port, args: p.args } }));
      } else if (b.dataset.act === "edit") editProfile(p);
      else if (b.dataset.act === "copy") {
        const taken = new Set(list.map((x) => x.name));
        let name = t("dash_copy_of", { name: p.name });
        for (let n = 2; taken.has(name); n++) name = t("dash_copy_of", { name: p.name }) + ` ${n}`;
        editProfile({ name, model: p.model, port: p.port, args: p.args, _new: true });
      }
      else if (b.dataset.act === "del") {
        if (!await FT.ask(t("dash_confirm_delete", { name: p.name }), { title: t("dash_delete"), ok: t("dash_delete"), danger: true })) return;
        FT.raw(`/profiles/${encodeURIComponent(p.name)}`, { method: "DELETE" }).then(loadProfiles, (e) => FT.notice(e.body?.detail || e.body?.error || e.message, t("dash_delete")));
      }
    };
  }

  function editProfile(p) {
    FTProfileEditor.open(p, { onSaved: loadProfiles });
  }

  async function followLogs() {
    const el = $("#log");
    let since = 0;
    for (;;) {
      try {
        await FT.sse(`/engine/logs?since=${since}`, (ev) => {
          if (!ev.data) return;
          let rec; try { rec = JSON.parse(ev.data); } catch { return; }
          if (rec.kind === "gap") { el.append(t("dash_log_gap", { n: rec.dropped }) + "\n"); return; }
          since = rec.seq + 1;
          if (rec.kind === "progress") return;
          el.append(homePath(rec.text) + "\n");
          while (el.childNodes.length > 1500) el.removeChild(el.firstChild);
          if ($("#log-follow").checked) el.scrollTop = el.scrollHeight;
        });
      } catch {}
      await new Promise((r) => setTimeout(r, 3000));
    }
  }

  if (managed) {
    $("#daemon-only").hidden = false;
    $("#btn-save-profile").hidden = !FT.canWrite;
    $("#btn-save-profile").onclick = () => editProfile(engineConfig?.model ? { model: engineConfig.model, port: engineConfig.port, args: engineConfig.args } : {});
    await loadConfig();
    $("#btn-save-profile").textContent = t(engineConfig?.model ? "dash_profile_save_current" : "dash_profile_add");
    loadProfiles();
    followLogs();
  }

  await pollGeometry();
  tick(); pollRequests();
  setInterval(tick, 2000);
  setInterval(pollRequests, 5000);
  setInterval(pollGeometry, 15000);
  setTimeout(pollHeat, 1500);
  setInterval(pollHeat, 10000);
  pollHost();
  setInterval(pollHost, 3000);
  addEventListener("resize", drawChart);
})();
