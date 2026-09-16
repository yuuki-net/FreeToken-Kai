// The benchmark page: run the measurement job in ft mgr (webui/tuner.py), draw every reading as it
// arrives, then show what was measured and the flags it settled on, ready to put in a profile.
"use strict";

FTI18N.add({
  bm_title: "ベンチマークで設定を決める",
  bm_hint: "この PC とモデルで実際に測り、その結果から ft serve のフラグを決めます。PCIe・メモリ・SSD の速さ、CPU と GPU でのエキスパート処理の速さを測り、必要ならモデルを起動して、効きそうな設定を 1 つずつ実測で比べます。終わったら結果をそのままプロファイルにできます。",
  bm_trials: "モデルを起動して実測する（おすすめ）",
  bm_trials_hint: "プロンプト処理と生成の速さを実際に測り、プリフィルのチャンク予算やコンテキスト長を 1 つずつ変えて比べます。モデルの読み込みが 2〜4 回あるので、全体で 10〜25 分ほどかかります。外すとハードウェアだけ測ります（数分）。",
  bm_start: "測定を始める", bm_last: "前回の結果を見る", bm_last_when: "前回: {when}",
  bm_running_title: "測定中", bm_cancel: "中止",
  bm_confirm: "測定を始めます。GPU を使うので、動いているサーバ（{model}）は止まり、終わったら同じ設定で起動し直します。",
  bm_confirm_none: "測定を始めます。測定中は GPU を使います。",
  bm_confirm_time: "かかる時間の目安: {min}。途中で中止できます。",
  bm_time_hw: "数分", bm_time_trials: "10〜25 分",
  bm_view_only: "この画面から測定を始めるには、この PC で開くかトークンが必要です。",
  bm_external: "ft mgr の管理外で起動したサーバが動いていて、GPU を使っています。止めてから測定してください。",
  bm_busy: "ほかの測定が動いています。",
  bm_ph_stop: "動いているサーバを止めています", bm_ph_hw: "ハードウェアを測っています", bm_ph_trial: "モデルを起動して測っています",
  bm_ph_restore: "元のサーバを起動し直しています", bm_ph_done: "終わりました", bm_ph_cancelled: "中止しました", bm_ph_error: "エラーで止まりました",
  bm_s_gpu: "GPU を調べる", bm_s_pcie: "PCIe の転送速度（GPU {gpu}）", bm_s_ram: "メモリの読み出し速度", bm_s_ssd: "SSD からモデルを読む速度",
  bm_s_cpu_moe: "CPU でのエキスパート計算（スレッド数ごと）", bm_s_gather: "GPU へのエキスパート転送（GPU {gpu}）", bm_s_overlap: "CPU 計算と GPU 転送を同時に",
  bm_s_load: "設定 {trial} で起動（{what}）", bm_s_prefill: "設定 {trial}: プロンプト処理（{n} トークン）", bm_s_decode: "設定 {trial}: 生成（{n} トークン）",
  bm_skip_failed: "測れませんでした", bm_skip_no_weights: "重みファイルがありません",
  bm_threads: "{n} スレッド", bm_loading_pct: "読み込み {v}%", bm_loading: "モデルを読み込み中", bm_working: "処理中", bm_working_hint: "最初の区切りが終わるまで数値は出ません", bm_progress: "{done} / {total} トークン",
  bm_result: "測定結果", bm_took: "{when} · {min} 分 {sec} 秒",
  bm_new_profile: "新しいプロファイルとして保存", bm_apply_to: "既存のプロファイルに反映", bm_start_now: "この設定で起動",
  bm_pick_profile: "反映するプロファイル", bm_profile_suffix: "{name}（ベンチマーク）", bm_no_same_model: "同じモデルのプロファイルがありません",
  bm_again: "もう一度測る",
  bm_hw_title: "ハードウェア", bm_link: "PCIe {gen}.0 x{width} の理論値 {max}", bm_link_down: "x{width} で動いています（カードは x{max} まで）",
  bm_ram: "メモリの読み出し", bm_ram_sub: "{n} コアで並列に読んだ合計",
  bm_ssd: "SSD の読み出し", bm_ssd_sub: "モデルのファイルをキャッシュなしで読んだ速さ",
  bm_ssd_ra: "read_ahead_kb が {n} です。256 にすると SSD から冷えたエキスパートを読む速さが上がることがあります（docs/bank-ram.md）。",
  bm_cpu: "CPU でのエキスパート計算", bm_cpu_sub: "{n} スレッドで最大の 95%（{cores} コア中）",
  bm_gather: "GPU へのエキスパート転送（GPU {gpu}）", bm_gather_sub: "エキスパートの重みをまとめて送る速さ",
  bm_overlap: "同時に動かしたとき", bm_overlap_sub: "足りないエキスパートの {pct} を GPU に送り、残りを CPU で計算",
  bm_sweep_title: "スレッド数ごとの CPU 計算速度",
  bm_trials_title: "実測（モデルを起動して）", bm_col_setting: "設定", bm_col_ctx: "コンテキスト長", bm_col_slots: "GPU のエキスパート枠",
  bm_col_load: "起動", bm_col_prefill: "プロンプト処理", bm_col_decode: "生成", bm_col_hit: "GPU で足りた割合", bm_chosen: "採用",
  bm_trial_failed: "起動か測定に失敗", bm_col_change: "変えたところ", bm_rejected: "試して不採用",
  bm_trials_hint2: "設定 A から 1 つずつ変えて測り、速くなったものだけを次に引き継ぎます。",
  bm_change_base: "基本の設定", bm_change_ctx: "コンテキスト長 {n}", bm_change_without: "{flag} なし",
  bm_flags_title: "決まったフラグ", bm_src_measured: "実測", bm_src_rule: "目安", bm_removed: "外した",
  bm_flags_note: "「実測」はこの測定の数字から、「目安」は GPU・RAM・モデルの構成から決めた値です。",
  bm_saved: "プロファイルを保存しました。",
  bm_empty_hint: "モデルを選んで「測定を始める」を押してください。", bm_loading_models: "モデルを読み込み中…",
}, {
  bm_title: "Benchmark to pick the settings",
  bm_hint: "Measure this PC with the model and pick ft serve's flags from the numbers: PCIe, memory and SSD rates, how fast experts are computed on the CPU and moved to the GPU, and, if you like, the model itself started to compare the settings that matter one at a time. The result becomes a profile in one click.",
  bm_trials: "Start the model and measure it (recommended)",
  bm_trials_hint: "Measures prompt processing and generation for real and changes the prefill chunk budget and the context length one at a time. The model loads two to four times, so this takes about 10-25 minutes. Unticked, only the hardware is measured (a few minutes).",
  bm_start: "Start", bm_last: "Show the last result", bm_last_when: "Last: {when}",
  bm_running_title: "Measuring", bm_cancel: "Cancel",
  bm_confirm: "The measurement uses the GPU, so the running server ({model}) stops and is started again with the same settings afterwards.",
  bm_confirm_none: "The measurement uses the GPU.",
  bm_confirm_time: "Expected time: {min}. You can cancel at any point.",
  bm_time_hw: "a few minutes", bm_time_trials: "10-25 minutes",
  bm_view_only: "Starting a benchmark from this page needs this PC or the token.",
  bm_external: "A server started outside ft mgr is running and holds the GPU. Stop it first.",
  bm_busy: "Another benchmark is running.",
  bm_ph_stop: "Stopping the running server", bm_ph_hw: "Measuring the hardware", bm_ph_trial: "Measuring the model",
  bm_ph_restore: "Starting the previous server again", bm_ph_done: "Done", bm_ph_cancelled: "Cancelled", bm_ph_error: "Stopped on an error",
  bm_s_gpu: "Look at the GPUs", bm_s_pcie: "PCIe transfer (GPU {gpu})", bm_s_ram: "Memory read", bm_s_ssd: "Reading the model from the SSD",
  bm_s_cpu_moe: "Experts computed on the CPU, by thread count", bm_s_gather: "Experts moved to the GPU (GPU {gpu})", bm_s_overlap: "CPU compute and GPU transfer together",
  bm_s_load: "Start setting {trial} ({what})", bm_s_prefill: "Setting {trial}: prompt processing ({n} tokens)", bm_s_decode: "Setting {trial}: generation ({n} tokens)",
  bm_skip_failed: "could not measure", bm_skip_no_weights: "no weight files",
  bm_threads: "{n} threads", bm_loading_pct: "loading {v}%", bm_loading: "loading the model", bm_working: "working", bm_working_hint: "no figure until the first chunk is done", bm_progress: "{done} / {total} tokens",
  bm_result: "Result", bm_took: "{when} · {min} min {sec} s",
  bm_new_profile: "Save as a new profile", bm_apply_to: "Apply to a profile", bm_start_now: "Start with these settings",
  bm_pick_profile: "Profile to update", bm_profile_suffix: "{name} (benchmark)", bm_no_same_model: "No profile uses this model",
  bm_again: "Measure again",
  bm_hw_title: "Hardware", bm_link: "PCIe {gen}.0 x{width} carries up to {max}", bm_link_down: "running at x{width} (the card does x{max})",
  bm_ram: "Memory read", bm_ram_sub: "summed over {n} cores in parallel",
  bm_ssd: "SSD read", bm_ssd_sub: "the model's own file, page cache dropped",
  bm_ssd_ra: "read_ahead_kb is {n}. 256 can make reading cold experts from the SSD faster (docs/bank-ram.md).",
  bm_cpu: "Experts on the CPU", bm_cpu_sub: "{n} threads reach 95% of the best ({cores} cores)",
  bm_gather: "Experts to the GPU (GPU {gpu})", bm_gather_sub: "expert weights moved in bulk",
  bm_overlap: "Both at once", bm_overlap_sub: "fetch {pct} of the missing experts, compute the rest on the CPU",
  bm_sweep_title: "CPU compute by thread count",
  bm_trials_title: "Measured with the model", bm_col_setting: "Setting", bm_col_ctx: "Context", bm_col_slots: "Experts on the GPU",
  bm_col_load: "Start", bm_col_prefill: "Prompt processing", bm_col_decode: "Generation", bm_col_hit: "Served from the GPU", bm_chosen: "chosen",
  bm_trial_failed: "failed to start or measure", bm_col_change: "Changed", bm_rejected: "tried, not kept",
  bm_trials_hint2: "Each run changes one thing from the best so far; only what measured faster is carried on.",
  bm_change_base: "base settings", bm_change_ctx: "context {n}", bm_change_without: "without {flag}",
  bm_flags_title: "The flags", bm_src_measured: "measured", bm_src_rule: "rule", bm_removed: "dropped",
  bm_flags_note: "“measured” values come from this run's numbers, “rule” values from the GPU, RAM and the model's config.",
  bm_saved: "The profile was saved.",
  bm_empty_hint: "Pick a model and press Start.", bm_loading_models: "Loading models…",
});

(async () => {
  const { fmt, esc, $, t, modelName, homePath } = FT;
  await FT.init();
  FT.header("bench");
  const managed = FT.mode === "mgr", demo = FT.mode === "demo";
  const lang = () => FTI18N.lang;

  // ---------------------------------------------------------------- run view state
  let steps = new Map(), order = [], current = null, run = null, seq = 0, startedAt = null, trialCtx = {};
  const gauge = { shown: 0, target: 0, max: 10, unit: "GB/s", label: "" };

  function resetRun() {
    steps = new Map(); order = []; current = null; seq = 0; startedAt = null; trialCtx = {};
    gauge.shown = gauge.target = 0;
    $("#bm-steps").innerHTML = "";
  }

  function addStep(id, label, unit, max) {
    if (steps.has(id)) { const s = steps.get(id); s.label = label; renderSteps(); return s; }
    const s = { id, label, unit, max, state: "wait", value: null, samples: [], note: "" };
    steps.set(id, s); order.push(id);
    renderSteps();
    return s;
  }

  const unitOf = (id) => /\.(prefill|decode)$/.test(id) ? "tok/s" : /\.load$/.test(id) ? "%" : "GB/s";
  const fmtVal = (v, unit) => v == null ? "—" : unit === "%" ? `${Math.round(v)}%` : unit === "tok/s" ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} tok/s` : `${v.toFixed(1)} GB/s`;

  function hwLabel(id) {
    const m = /^(pcie|gather)(\d+)$/.exec(id);
    return m ? t(`bm_s_${m[1]}`, { gpu: m[2] }) : t(`bm_s_${id}`);
  }

  function renderSteps() {
    $("#bm-steps").innerHTML = order.map((id) => {
      const s = steps.get(id);
      const ico = { wait: "", run: "●", done: "✓", skip: "!", failed: "!" }[s.state];
      const running = s.state === "run" && s.since ? `${Math.floor((Date.now() - s.since) / 1000)} s` : "";
      const val = s.state === "done" && s.value != null ? fmtVal(s.value, s.unit) : (s.note || running);
      return `<li class="${s.state}"><span class="ico">${ico}</span><span>${esc(s.label)}</span><span class="val">${esc(val || "")}</span></li>`;
    }).join("");
  }

  function focus(id) {
    current = id;
    const s = steps.get(id);
    gauge.unit = s.unit; gauge.label = s.label;
    gauge.target = 0;
    gauge.max = niceMax(Math.max(s.max || 0, ...s.samples.map((x) => x.v)) || (s.unit === "tok/s" ? 50 : s.unit === "%" ? 100 : 5), s.unit);
    $("#bm-now").textContent = s.label;
    drawSpark();
  }

  function niceMax(v, unit) {
    if (unit === "%") return 100;
    const steps = [1, 2, 2.5, 5, 10];
    const exp = Math.pow(10, Math.floor(Math.log10(Math.max(v * 1.15, 1e-3))));
    for (const m of steps) if (m * exp >= v * 1.15) return m * exp;
    return 10 * exp;
  }

  function sample(id, v, extra = {}) {
    const s = steps.get(id) || addStep(id, id, unitOf(id));
    s.samples.push({ v, ...extra });
    if (current !== id) focus(id);
    gauge.target = v;
    if (v > gauge.max * 0.95) gauge.max = niceMax(v, s.unit);
    $("#bm-now").textContent = extra.threads ? `${s.label} · ${t("bm_threads", { n: extra.threads })}` : s.label;
    if (s.unit === "%") s.note = t("bm_loading_pct", { v: Math.round(v) });
    renderSteps();
    drawSpark();
  }

  // ---------------------------------------------------------------- events from the job
  function onEvent(e) {
    startedAt = startedAt || e.t;
    if (e.k === "phase") {
      $("#bm-phase").textContent = t(`bm_ph_${e.phase}`) + (e.message ? ` — ${e.message}` : "");
      if (["done", "cancelled", "error"].includes(e.phase)) for (const s of steps.values()) if (s.state === "run") s.state = e.phase === "done" ? "done" : "failed";
      renderSteps();
    } else if (e.k === "hw") {
      if (e.kind === "plan") for (const s of e.steps) addStep(s.id, hwLabel(s.id), s.unit || "GB/s", s.max);
      else if (e.kind === "step") { const s = steps.get(e.id) || addStep(e.id, hwLabel(e.id), "GB/s"); s.state = "run"; focus(e.id); renderSteps(); }
      else if (e.kind === "sample") sample(e.id, e.value, e.threads ? { threads: e.threads } : {});
      else if (e.kind === "done") {
        const s = steps.get(e.id); if (!s) return;
        s.state = "done"; s.value = e.id === "gpu" ? null : e.value;
        if (e.id === "gpu") s.note = (e.gpus || []).map((g) => g.name.replace(/^NVIDIA\s+/, "")).join(", ");
        if (e.id === "cpu_moe") s.note = t("bm_threads", { n: e.threads });
        if (current === e.id && e.value != null) gauge.target = e.value;
        renderSteps();
      } else if (e.kind === "skip") {
        const s = steps.get(e.id) || addStep(e.id, hwLabel(e.id), "GB/s");
        s.state = "skip"; s.note = t(`bm_skip_${e.reason}`) || e.reason; renderSteps();
      }
    } else if (e.k === "trial") {
      const T = e.trial;
      if (e.step === "load" && e.state === "start") {
        trialCtx[T] = e.change || null;
        addStep(`${T}.load`, t("bm_s_load", { trial: T, what: changeText(e.change) }), "%", 100);
        addStep(`${T}.prefill`, t("bm_s_prefill", { trial: T, n: "…" }), "tok/s");
        addStep(`${T}.decode`, t("bm_s_decode", { trial: T, n: "…" }), "tok/s");
      }
      if (e.state === "failed" && !steps.has(`${T}.${e.step}`)) {
        // failed between steps (a request that errored mid-measurement): mark whatever was running
        for (const k of ["load", "prefill", "decode"]) {
          const st = steps.get(`${T}.${k}`);
          if (st && st.state !== "done") { st.state = "failed"; st.note = t("bm_trial_failed"); }
        }
        renderSteps();
        return;
      }
      const id = `${T}.${e.step}`;
      const s = steps.get(id);
      if (!s) return;
      if (e.step === "prefill" && e.tokens) s.label = t("bm_s_prefill", { trial: T, n: fmt.num(e.tokens) });
      if (e.step === "decode" && e.tokens) s.label = t("bm_s_decode", { trial: T, n: fmt.num(e.tokens) });
      if (e.state === "start") { s.state = "run"; s.since = Date.now(); s.progress = null; focus(id); }
      else if (e.state === "done") { s.state = "done"; s.value = e.step === "load" ? null : e.value; if (e.step === "load") s.note = `${e.seconds} s`; }
      else if (e.state === "failed") { s.state = "failed"; s.note = t("bm_trial_failed"); }
      renderSteps();
    } else if (e.k === "sample") { const s = steps.get(e.id); if (s) s.progress = null; sample(e.id, e.value); }
    else if (e.k === "progress") {
      const s = steps.get(e.id);
      if (!s) return;
      // a new request starts from zero again: keep the samples, restart the count
      s.progress = { done: e.done, total: e.total, rate: e.rate, at: Date.now() };
      if (current !== e.id) focus(e.id);
      if (e.rate) { gauge.target = e.rate; if (e.rate > gauge.max * 0.95) gauge.max = niceMax(e.rate, s.unit); }
      renderSteps();
    }
  }

  const argValue = (args, flag) => { const i = (args || []).indexOf(flag); return i >= 0 ? args[i + 1] : null; };
  // what a run changed against the best one before it
  function changeText(change) {
    if (!change) return t("bm_change_base");
    if (change.flag === "--kv-reserve-tokens" || change.flag === "--max-seq-len-override") return t("bm_change_ctx", { n: fmt.num(+change.value) });
    if (change.value == null) return t("bm_change_without", { flag: change.flag });
    return `${change.flag} ${change.value}`;
  }

  // ---------------------------------------------------------------- drawing
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function drawGauge() {
    const cv = $("#bm-gauge");
    if (!cv || $("#bm-run").hidden) return;
    const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    gauge.shown += (gauge.target - gauge.shown) * 0.12;
    const r = Math.min(w * 0.4, h * 0.42), cx = w / 2, cy = 16 + r;
    const valueY = cy + r * 0.52, unitY = valueY + 22, barY = unitY + 20;
    const a0 = Math.PI * 0.8, a1 = Math.PI * 2.2, frac = Math.max(0, Math.min(1, gauge.shown / gauge.max));
    const cur = current && steps.get(current);
    const waiting = cur && cur.state === "run" && !cur.samples.length && !(cur.progress && cur.progress.rate);
    if (waiting && (cur.unit === "%" || cur.unit === "tok/s")) {
      // a load that reports no bytes: a sweeping arc and the seconds so far, not a needle stuck at 0
      const sec = Math.floor((Date.now() - (cur.since || Date.now())) / 1000);
      const head = a0 + ((Date.now() / 1400) % 1) * (a1 - a0);
      ctx.lineCap = "round"; ctx.lineWidth = 16;
      ctx.strokeStyle = css("--tile"); ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1); ctx.stroke();
      ctx.strokeStyle = css("--accent"); ctx.beginPath(); ctx.arc(cx, cy, r, Math.max(a0, head - 0.6), head); ctx.stroke();
      ctx.fillStyle = css("--fg"); ctx.textAlign = "center"; ctx.font = "600 34px system-ui";
      ctx.fillText(`${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`, cx, valueY);
      ctx.font = "13px system-ui"; ctx.fillStyle = css("--fg2");
      ctx.fillText(t(cur.unit === "%" ? "bm_loading" : "bm_working"), cx, unitY);
      if (cur.unit !== "%" && !cur.progress) {
        ctx.font = "12px system-ui"; ctx.fillStyle = css("--fg3");
        ctx.fillText(t("bm_working_hint"), cx, barY + 4);
      }
      drawProgress(ctx, cur, cx, r, barY);
      return;
    }
    ctx.lineCap = "round";
    ctx.lineWidth = 16; ctx.strokeStyle = css("--tile");
    ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1); ctx.stroke();
    const grad = ctx.createLinearGradient(cx - r, 0, cx + r, 0);
    grad.addColorStop(0, css("--c-h2d")); grad.addColorStop(0.6, css("--c-kv")); grad.addColorStop(1, css("--c-moe"));
    ctx.strokeStyle = grad;
    if (frac > 0.002) { ctx.beginPath(); ctx.arc(cx, cy, r, a0, a0 + (a1 - a0) * frac); ctx.stroke(); }
    ctx.fillStyle = css("--fg3"); ctx.font = "11px system-ui"; ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
      const a = a0 + (a1 - a0) * i / 5, x = cx + Math.cos(a) * (r - 26), y = cy + Math.sin(a) * (r - 26);
      const v = gauge.max * i / 5;
      ctx.fillText(gauge.unit === "%" ? `${Math.round(v)}` : v >= 100 ? Math.round(v) : +v.toFixed(1), x, y + 4);
    }
    const na = a0 + (a1 - a0) * frac;
    ctx.strokeStyle = css("--fg"); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(na) * (r - 40), cy + Math.sin(na) * (r - 40)); ctx.stroke();
    ctx.fillStyle = css("--fg"); ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.font = "600 34px system-ui";
    const shown = gauge.unit === "%" ? `${Math.round(gauge.shown)}` : gauge.shown >= 100 ? Math.round(gauge.shown) : gauge.shown.toFixed(1);
    ctx.fillText(current ? shown : "—", cx, valueY);
    ctx.font = "13px system-ui"; ctx.fillStyle = css("--fg2");
    ctx.fillText(gauge.unit, cx, unitY);
    drawProgress(ctx, cur, cx, r, barY);
  }

  // how far the prompt has got: the count arrives per chunk, so between updates it moves on at the
  // measured rate and never past the next report's worth
  function drawProgress(ctx, s, cx, r, y) {
    const p = s && s.state === "run" ? s.progress : null;
    if (!p || !p.total) return;
    const est = p.rate ? Math.min(p.total, p.done + p.rate * (Date.now() - p.at) / 1000) : p.done;
    const frac = Math.max(0, Math.min(1, est / p.total));
    // as wide as the dial itself, stroke included, with the same round ends
    const half = r + 8, x0 = cx - half, x1 = cx + half;
    ctx.lineCap = "round"; ctx.lineWidth = 6;
    ctx.strokeStyle = css("--tile"); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    if (frac > 0.002) { ctx.strokeStyle = css("--c-h2d"); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + (x1 - x0) * frac, y); ctx.stroke(); }
    ctx.font = "12px system-ui"; ctx.fillStyle = css("--fg3"); ctx.textAlign = "center";
    ctx.fillText(t("bm_progress", { done: fmt.num(est), total: fmt.num(p.total) }), cx, y + 20);
  }

  function drawSpark() {
    const cv = $("#bm-spark");
    const s = current && steps.get(current);
    const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // a load percentage only ever climbs: a line of it says nothing the dial does not
    if (!s || !s.samples.length || s.unit === "%") return;
    const vals = s.samples.map((x) => x.v), max = Math.max(...vals) * 1.1 || 1;
    if (s.samples[0].threads) {
      // the thread sweep: one bar per thread count
      const bw = w / vals.length;
      s.samples.forEach((x, i) => {
        const bh = (h - 16) * x.v / max;
        ctx.fillStyle = css("--accent"); ctx.fillRect(i * bw + 4, h - 14 - bh, bw - 8, bh);
        ctx.fillStyle = css("--fg3"); ctx.font = "10px system-ui"; ctx.textAlign = "center";
        ctx.fillText(x.threads, i * bw + bw / 2, h - 2);
      });
      return;
    }
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, i) => { const x = vals.length > 1 ? w * i / (vals.length - 1) : w / 2, y = h - 4 - (h - 8) * v / max; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }

  (function loop() { drawGauge(); requestAnimationFrame(loop); })();
  setInterval(() => {
    if (!$("#bm-run").hidden && [...steps.values()].some((s) => s.state === "run" && s.since && !s.note)) renderSteps();
    if (startedAt && !$("#bm-run").hidden) {
      const s = Math.max(0, Math.round(Date.now() / 1000 - startedAt));
      $("#bm-elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }
  }, 500);

  // ---------------------------------------------------------------- the job
  function show(view) {
    $("#bm-setup").hidden = view !== "setup";
    $("#bm-run").hidden = view !== "run";
    $("#bm-result").hidden = view !== "result";
    if (view === "result") $("#bm-setup").hidden = false;
  }

  let polling = false;
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const since = run === null ? 0 : seq;
      const st = demo ? DemoJob.status(since) : await FT.raw(`/tune/status?since=${since}`);
      if (st.run !== run) { resetRun(); run = st.run; if (st.state === "running") show("run"); }
      for (const e of st.events) { onEvent(e); seq = e.seq; }
      if (st.state === "running") { if ($("#bm-run").hidden) show("run"); }
      else if (!$("#bm-run").hidden) {
        if (st.state === "done" && st.result) { show("result"); renderResult(st.result); }
        else if (st.state !== "idle") { $("#bm-phase").textContent = t(`bm_ph_${st.state}`) + (st.error ? ` — ${st.error}` : ""); setTimeout(() => show("setup"), 4000); }
      }
    } catch {} finally { polling = false; }
  }

  // ---------------------------------------------------------------- setup
  if (!managed && !demo) { $("#bm-setup").innerHTML = `<div class="empty small">${esc(t("bm_view_only"))}</div>`; return; }
  // a job already running shows up without waiting for the model list
  poll();
  setInterval(poll, 600);

  let models = [], engineCfg = null;
  $("#bm-model").innerHTML = `<option>${esc(t("bm_loading_models"))}</option>`;
  $("#bm-model").disabled = $("#bm-start").disabled = true;
  if (managed) {
    [models, engineCfg] = await Promise.all([
      FT.raw("/models").then((d) => d.models || [], () => []),
      FT.raw("/engine/config").catch(() => null),
    ]);
  } else models = [{ value: "/home/demo/models/Qwen3.8-Flash-Next-NVFP4", name: "Qwen3.8-Flash-Next-NVFP4", size_bytes: 70 * 2 ** 30 }];
  $("#bm-model").innerHTML = models.map((m) => `<option value="${esc(m.value)}">${esc(m.name)}（${fmt.gib(m.size_bytes)}）</option>`).join("");
  if (engineCfg?.model && models.some((m) => m.value === engineCfg.model)) $("#bm-model").value = engineCfg.model;
  $("#bm-model").disabled = false;
  $("#bm-start").disabled = !FT.canWrite || !models.length;
  if (!FT.canWrite) { const w = $("#bm-warn"); w.hidden = false; w.textContent = t("bm_view_only"); }

  let lastResult = null;
  async function loadLast() {
    lastResult = null;
    $("#bm-last").hidden = true; $("#bm-last-when").textContent = "";
    if (demo) return;
    try { lastResult = (await FT.raw(`/tune/last?model=${encodeURIComponent($("#bm-model").value)}`)).result; } catch {}
    if (lastResult) {
      $("#bm-last").hidden = false;
      $("#bm-last-when").textContent = t("bm_last_when", { when: new Date(lastResult.finished * 1000).toLocaleString(lang() === "ja" ? "ja-JP" : "en-US") });
    }
  }
  $("#bm-model").onchange = loadLast;
  $("#bm-last").onclick = () => { show("result"); renderResult(lastResult); };
  loadLast();

  $("#bm-start").onclick = async () => {
    const trials = $("#bm-trials").checked;
    let running = null;
    $("#bm-start").disabled = true;  // the health check can take a second or two
    try { running = (await FT.serveGet("/health")).running ? engineCfg?.model : null; } catch {}
    $("#bm-start").disabled = false;
    const msg = [running ? t("bm_confirm", { model: modelName(running) }) : t("bm_confirm_none"),
      t("bm_confirm_time", { min: t(trials ? "bm_time_trials" : "bm_time_hw") })].join("\n\n");
    if (!await FT.ask(msg, { title: t("bm_title"), ok: t("bm_start") })) return;
    if (demo) { DemoJob.start(trials); run = null; show("run"); return; }
    try {
      await FT.raw("/tune/start", { method: "POST", json: { model: $("#bm-model").value, trials } });
      run = null; show("run");
    } catch (e) {
      const code = e.body?.code;
      FT.notice(code === "external_serve" ? t("bm_external") : code === "busy" ? t("bm_busy") : (e.body?.detail || e.body?.error || e.message), t("bm_title"));
    }
  };
  $("#bm-cancel").onclick = () => (demo ? DemoJob.cancel() : FT.raw("/tune/cancel", { method: "POST", json: {} }).catch(() => {}));

  // ---------------------------------------------------------------- result
  function renderResult(r) {
    const hw = r.hw?.measurements || {};
    const took = Math.max(0, Math.round((r.finished || 0) - (r.started || 0)));
    const tiles = [];
    const tile = (label, value, sub, extra = "") => tiles.push(`<div class="tile"><div class="l">${esc(label)}</div><div class="v">${value}</div><div class="s">${esc(sub)}</div>${extra}</div>`);
    const bar = (v, max, color = "var(--accent)") => max ? `<div class="bar" style="margin-top:6px;height:8px"><div style="width:${Math.min(100, 100 * v / max).toFixed(1)}%;background:${color}"></div></div>` : "";
    const gbs = (v) => v == null ? "—" : `${v.toFixed(1)} <small>GB/s</small>`;
    for (const g of hw.gpus || []) {
      const p = (hw.pcie || {})[g.index];
      if (!p) continue;
      const sub = g.pcie_gen && g.pcie_width ? t("bm_link", { gen: g.pcie_gen, width: g.pcie_width, max: `${g.link_gbs} GB/s` }) : g.name;
      const down = g.pcie_width && g.pcie_width_max && g.pcie_width < g.pcie_width_max
        ? `<div class="s warn-text">${esc(t("bm_link_down", { width: g.pcie_width, max: g.pcie_width_max }))}</div>` : "";
      tile(`PCIe · GPU ${g.index} · ${g.name.replace(/^NVIDIA\s+/, "")}`, gbs(p.h2d_gbs), sub, bar(p.h2d_gbs, g.link_gbs) + down);
    }
    if (hw.ram) tile(t("bm_ram"), gbs(hw.ram.read_gbs), t("bm_ram_sub", { n: hw.ram.threads }));
    if (hw.ssd) tile(t("bm_ssd"), gbs(hw.ssd.read_gbs), t("bm_ssd_sub"),
      hw.ssd.read_ahead_kb > 512 ? `<div class="s warn-text" style="margin-top:4px">${esc(t("bm_ssd_ra", { n: hw.ssd.read_ahead_kb }))}</div>` : "");
    if (hw.cpu_moe) tile(t("bm_cpu"), gbs(hw.cpu_moe.best_gbs), t("bm_cpu_sub", { n: hw.cpu_moe.threads, cores: hw.cpu_moe.cores }));
    for (const [gpu, g] of Object.entries(hw.gather || {})) if (g) tile(t("bm_gather", { gpu }), gbs(g.gbs), t("bm_gather_sub"));
    if (hw.overlap?.fetch_fraction != null) tile(t("bm_overlap"), `${(hw.overlap.pcie_gbs + hw.overlap.cpu_gbs).toFixed(1)} <small>GB/s</small>`,
      t("bm_overlap_sub", { pct: fmt.pct(hw.overlap.fetch_fraction, 0) }));

    const trials = r.trials || [];
    const chosen = r.chosen;
    const okTrials = trials;
    const trialTable = okTrials.length ? `<div class="card"><h2>${esc(t("bm_trials_title"))}</h2>
      <p class="hint">${esc(t("bm_trials_hint2"))}</p>
      <div class="tablewrap"><table class="bm-trials"><thead><tr><th>${esc(t("bm_col_setting"))}</th><th>${esc(t("bm_col_change"))}</th>
        <th class="num">${esc(t("bm_col_ctx"))}</th><th class="num">${esc(t("bm_col_slots"))}</th><th class="num">${esc(t("bm_col_load"))}</th>
        <th class="num">${esc(t("bm_col_prefill"))}</th><th class="num">${esc(t("bm_col_decode"))}</th><th class="num">${esc(t("bm_col_hit"))}</th></tr></thead>
      <tbody>${okTrials.map((x) => {
        const win = x.ok && x.label === chosen ? "win" : "";
        return `<tr><td class="${win}"><b>${esc(x.label)}</b> ${win ? `<span class="pill ok">${esc(t("bm_chosen"))}</span>` : ""}${x.ok ? "" : ` <span class="pill warn">${esc(t("bm_trial_failed"))}</span>`}</td>
          <td class="${win}"><code style="font-size:12px">${esc(changeText(x.change))}</code></td>
          <td class="num ${win}">${x.kv_tokens ? fmt.num(x.kv_tokens) : esc(argValue(x.args, "--kv-reserve-tokens") || "—")}</td>
          <td class="num ${win}">${x.expert_slots != null ? fmt.num(x.expert_slots) : "—"}</td>
          <td class="num ${win}">${x.load_s != null ? `${x.load_s} s` : "—"}</td>
          <td class="num ${win}">${fmtVal(x.prefill_tps, "tok/s")}</td>
          <td class="num ${win}">${fmtVal(x.decode_tps, "tok/s")}</td>
          <td class="num ${win}">${fmt.pct(x.hit_rate)}</td></tr>`;
      }).join("")}</tbody></table></div></div>` : "";

    const notes = (r.notes || []).slice().sort((a, b) => (a.source === "measured" ? 0 : 1) - (b.source === "measured" ? 0 : 1));
    const flags = `<div class="card"><h2>${esc(t("bm_flags_title"))}</h2><p class="hint">${esc(t("bm_flags_note"))}</p>
      ${notes.map((n) => `<div class="bm-flag">
        <span style="flex:1;min-width:0"><span style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <code${n.removed || n.rejected ? ' style="text-decoration:line-through"' : ""}>${esc(n.flag)}${n.value != null ? " " + esc(homePath(n.value)) : ""}</code>
          <span class="pill ${n.rejected ? "warn" : n.source === "measured" ? "ok" : "mute"}">${esc(t(n.rejected ? "bm_rejected" : n.removed ? "bm_removed" : n.source === "measured" ? "bm_src_measured" : "bm_src_rule"))}</span></span>
          <div class="small muted" style="margin-top:2px">${esc(lang() === "en" && n.why_en ? n.why_en : n.why)}</div></span></div>`).join("")}</div>`;

    const sweep = hw.cpu_moe?.sweep;
    $("#bm-result").innerHTML = `
      <div class="card" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <div><div class="small muted">${esc(t("bm_result"))}</div>
          <div style="font-weight:600;font-size:16px">${esc(modelName(r.model))}</div>
          <div class="small muted">${esc(t("bm_took", { when: new Date((r.finished || Date.now() / 1000) * 1000).toLocaleString(lang() === "ja" ? "ja-JP" : "en-US"), min: Math.floor(took / 60), sec: took % 60 }))}</div></div>
        <span class="grow"></span>
        <div style="display:flex;gap:8px;flex-wrap:wrap" ${FT.canWrite && !demo ? "" : "hidden"}>
          <button id="bm-save-new" class="primary">${esc(t("bm_new_profile"))}</button>
          <button id="bm-save-into">${esc(t("bm_apply_to"))}</button>
          <button id="bm-run-now">${esc(t("bm_start_now"))}</button>
        </div>
      </div>
      <div class="card"><h2>${esc(t("bm_hw_title"))}</h2><div class="bm-score" style="margin-top:8px">${tiles.join("")}</div>
        ${sweep ? `<h2 style="margin-top:16px">${esc(t("bm_sweep_title"))}</h2><canvas id="bm-sweep" style="width:100%;height:150px"></canvas>` : ""}</div>
      ${trialTable}
      ${flags}`;
    if (sweep) drawSweep(sweep, hw.cpu_moe.threads);
    const b1 = $("#bm-save-new"), b2 = $("#bm-save-into"), b3 = $("#bm-run-now");
    if (b1) b1.onclick = () => FTProfileEditor.open({ name: t("bm_profile_suffix", { name: modelName(r.model) }), model: r.model, port: r.port, args: r.args, _new: true },
      { onSaved: () => FT.notice(t("bm_saved"), t("bm_new_profile")) });
    if (b2) b2.onclick = () => pickProfile(r);
    if (b3) b3.onclick = async () => {
      if (!await FT.ask(t("dash_confirm_apply", { name: modelName(r.model) }), { title: t("bm_start_now"), ok: t("dash_ok_start") })) return;
      FT.raw("/engine/switch", { method: "POST", json: { model: r.model, port: r.port, args: r.args } }).then(() => { location.href = "./"; }, (e) => FT.notice(e.body?.detail || e.body?.error || e.message));
    };
  }

  function drawSweep(sweep, chosen) {
    const cv = $("#bm-sweep");
    const entries = Object.entries(sweep).map(([k, v]) => [+k, v]).sort((a, b) => a[0] - b[0]);
    const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const max = Math.max(...entries.map((e) => e[1])) * 1.15, bw = Math.min(70, w / entries.length);
    entries.forEach(([th, v], i) => {
      const x = i * bw + (w - bw * entries.length) / 2, bh = (h - 34) * v / max;
      ctx.fillStyle = th === chosen ? css("--c-ssd") : css("--accent");
      ctx.fillRect(x + 6, h - 16 - bh, bw - 12, bh);
      ctx.fillStyle = css("--fg2"); ctx.font = "11px system-ui"; ctx.textAlign = "center";
      ctx.fillText(v.toFixed(1), x + bw / 2, h - 20 - bh);
      ctx.fillStyle = css("--fg3");
      ctx.fillText(t("bm_threads", { n: th }), x + bw / 2, h - 2);
    });
  }

  // flags from the benchmark over a profile's own: replaced in place, new ones appended
  function mergeArgs(base, over) {
    const parse = (a) => { const out = []; for (let i = 0; i < a.length; i++) { if (!a[i].startsWith("--")) continue; const v = a[i + 1] != null && !a[i + 1].startsWith("--") ? a[++i] : null; out.push([a[i - (v != null ? 1 : 0)], v]); } return out; };
    const b = parse(base || []), o = new Map(parse(over || []));
    const merged = b.map(([k, v]) => (o.has(k) ? [k, o.get(k)] : [k, v]));
    for (const [k, v] of o) if (!b.some(([bk]) => bk === k)) merged.push([k, v]);
    return merged.flatMap(([k, v]) => (v == null ? [k] : [k, v]));
  }

  async function pickProfile(r) {
    let list = [];
    try { list = ((await FT.raw("/profiles")).profiles || []).filter((p) => p.model === r.model); } catch {}
    if (!list.length) return FT.notice(t("bm_no_same_model"), t("bm_apply_to"));
    const d = document.createElement("dialog");
    d.className = "ask";
    d.innerHTML = `<form method="dialog" class="stack"><b>${esc(t("bm_pick_profile"))}</b>
      <select id="bm-pick">${list.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("")}</select>
      <div style="display:flex;gap:8px;justify-content:flex-end"><button value="cancel">${esc(t("cancel"))}</button><button value="ok" class="primary">${esc(t("dash_edit"))}</button></div></form>`;
    document.body.append(d);
    d.addEventListener("close", () => {
      const p = list[+$("#bm-pick", d).value];
      d.remove();
      if (d.returnValue !== "ok" || !p) return;
      FTProfileEditor.open({ ...p, args: mergeArgs(p.args, r.args) }, { onSaved: () => FT.notice(t("bm_saved"), t("bm_apply_to")) });
    });
    d.showModal();
  }

  // ---------------------------------------------------------------- ?demo: a made-up run to look at the page
  const DemoJob = (() => {
    let events = [], s = 0, state = "idle", result = null, id = 0, timer = null;
    const push = (k, f) => events.push({ seq: ++s, k, t: Date.now() / 1000, ...f });
    function start(trials) {
      id++; events = []; s = 0; state = "running"; result = null;
      const script = [];
      const at = (ms, fn) => script.push([ms, fn]);
      let ms = 0;
      at(ms, () => push("phase", { phase: "hw" }));
      at(ms += 300, () => push("hw", { kind: "plan", steps: [{ id: "gpu" }, { id: "pcie0", max: 7.88 }, { id: "pcie1", max: 7.88 }, { id: "ram" }, { id: "ssd" }, { id: "cpu_moe" }, { id: "gather0" }, { id: "gather1" }, { id: "overlap" }] }));
      at(ms += 400, () => { push("hw", { kind: "step", id: "gpu" }); push("hw", { kind: "done", id: "gpu", value: 2, gpus: [{ name: "NVIDIA GeForce RTX 3060" }, { name: "NVIDIA GeForce RTX 3060" }] }); });
      const series = (sid, base, n, gap, extra) => {
        at(ms += gap, () => push("hw", { kind: "step", id: sid }));
        for (let i = 0; i < n; i++) at(ms += gap, () => push("hw", { kind: "sample", id: sid, value: +(base * (0.9 + Math.random() * 0.15)).toFixed(2) }));
        at(ms += gap, () => push("hw", { kind: "done", id: sid, value: base, ...(extra || {}) }));
      };
      series("pcie0", 6.2, 8, 350); series("pcie1", 3.1, 8, 350); series("ram", 21.4, 5, 450); series("ssd", 3.3, 10, 300);
      at(ms += 400, () => push("hw", { kind: "step", id: "cpu_moe" }));
      [[1, 2.1], [2, 4.0], [4, 7.4], [6, 9.9], [8, 11.2], [10, 11.6], [12, 11.8]].forEach(([th, v]) => at(ms += 700, () => push("hw", { kind: "sample", id: "cpu_moe", value: v, threads: th })));
      at(ms += 400, () => push("hw", { kind: "done", id: "cpu_moe", value: 11.8, threads: 8 }));
      series("gather0", 5.6, 4, 500); series("gather1", 2.9, 4, 500);
      at(ms += 300, () => push("hw", { kind: "step", id: "overlap" }));
      at(ms += 1500, () => push("hw", { kind: "done", id: "overlap", value: 14.1 }));
      if (trials) {
        const runs = [["A", null, 18.1, 548], ["B", { flag: "--prefill-chunk-budget", value: "0.75" }, 18.3, 702], ["C", { flag: "--kv-reserve-tokens", value: "131072" }, 18.0, 689]];
        for (const [T, change, dec, pre] of runs) {
          at(ms += 500, () => { push("phase", { phase: "trial" }); push("trial", { trial: T, step: "load", state: "start", change }); });
          for (let p = 5; p <= 100; p += 5) at(ms += 150, () => push("sample", { id: `${T}.load`, value: p }));
          at(ms += 300, () => push("trial", { trial: T, step: "load", state: "done", seconds: 94 }));
          at(ms += 200, () => push("trial", { trial: T, step: "prefill", state: "start", tokens: 16384 }));
          for (let i = 0; i < 2; i++) {
            at(ms += 1200, () => {});
            for (let c = 1; c <= 3; c++) at(ms += 900, () => push("progress", { id: `${T}.prefill`, done: Math.min(16384, 5120 * c), total: 16384, rate: pre * (0.9 + Math.random() * 0.1) }));
            at(ms += 900, () => push("sample", { id: `${T}.prefill`, value: pre * (0.95 + Math.random() * 0.06) }));
          }
          at(ms += 200, () => push("trial", { trial: T, step: "prefill", state: "done", value: pre }));
          at(ms += 200, () => push("trial", { trial: T, step: "decode", state: "start", tokens: 300 }));
          for (let i = 0; i < 18; i++) at(ms += 250, () => push("sample", { id: `${T}.decode`, value: dec * (0.93 + Math.random() * 0.1) }));
          at(ms += 200, () => push("trial", { trial: T, step: "decode", state: "done", value: dec }));
        }
      }
      at(ms += 500, () => {
        push("phase", { phase: "done" });
        state = "done";
        result = demoResult(trials);
      });
      const t0 = Date.now();
      clearInterval(timer);
      timer = setInterval(() => { while (script.length && Date.now() - t0 >= script[0][0]) script.shift()[1](); if (!script.length) clearInterval(timer); }, 50);
    }
    function demoResult(trials) {
      const now = Date.now() / 1000;
      return {
        model: "/home/demo/models/Qwen3.8-Flash-Next-NVFP4", port: 1919, started: now - 1180, finished: now, chosen: "C",
        args: ["--pp-size", "2", "--gpu", "0,1", "--moe-strategy", "hybrid", "--moe-cpu-threads", "8", "--moe-cache-auto", "--moe-bank-ram", "44G", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "131072", "--max-seq-len-override", "131072"],
        hw: { measurements: {
          gpus: [0, 1].map((i) => ({ index: i, name: "NVIDIA GeForce RTX 3060", pcie_gen: 4, pcie_width: i ? 4 : 8, pcie_width_max: 16, link_gbs: i ? 7.88 : 15.75 })),
          pcie: { 0: { h2d_gbs: 6.2 }, 1: { h2d_gbs: 3.1 } }, ram: { read_gbs: 21.4, threads: 10 }, ssd: { read_gbs: 3.3, read_ahead_kb: 4096 },
          cpu_moe: { best_gbs: 11.8, threads: 8, cores: 10, sweep: { 1: 2.1, 2: 4.0, 4: 7.4, 6: 9.9, 8: 11.2, 10: 11.6 } },
          gather: { 0: { gbs: 5.6 }, 1: { gbs: 2.9 } }, overlap: { cpu_gbs: 9.8, pcie_gbs: 4.3, fetch_fraction: 0.305 },
        } },
        trials: trials ? [
          { label: "A", change: null, ok: true, kv_tokens: 65536, expert_slots: 1520, load_s: 101, prefill_tps: 548, decode_tps: 18.1, hit_rate: 0.581, args: [] },
          { label: "B", change: { flag: "--prefill-chunk-budget", value: "0.75" }, ok: true, kv_tokens: 65536, expert_slots: 1498, load_s: 99, prefill_tps: 702, decode_tps: 18.3, hit_rate: 0.579, args: [] },
          { label: "C", change: { flag: "--kv-reserve-tokens", value: "131072" }, ok: true, kv_tokens: 131072, expert_slots: 1180, load_s: 118, prefill_tps: 689, decode_tps: 18.0, hit_rate: 0.566, args: [] },
        ] : [],
        notes: [
          { flag: "--moe-strategy", value: "hybrid", source: "measured", why: "CPU でのエキスパート計算（11.8 GB/s）が GPU への転送（2.9 GB/s）の 4.1 倍速いので、GPU に無いエキスパートは CPU で計算します。", why_en: "Computing experts on the CPU (11.8 GB/s) is 4.1x the transfer to the GPU (2.9 GB/s), so experts not on the GPU are computed on the CPU." },
          { flag: "--moe-cpu-threads", value: "8", source: "measured", why: "8 スレッドで最大（11.8 GB/s）の 95% 以上が出ます。", why_en: "8 threads reach 95% of the best (11.8 GB/s)." },
          { flag: "--kv-reserve-tokens", value: "131072", source: "measured", why: "32,768 トークンでは生成 18.9 tok/s、131,072 トークンでは 18.4 tok/s。長くしても速さがほぼ変わらないので長い方にします。", why_en: "32,768 tokens gave 18.9 tok/s, 131,072 tokens 18.4. The longer context costs almost nothing, so it is kept." },
          { flag: "--pp-size", value: "2", source: "rule", why: "エキスパート以外の重みが 1 枚に収まらないので、層を 2 枚に分けます。", why_en: "The weights other than the experts do not fit one card, so the layers are split across 2." },
          { flag: "--moe-bank-ram", value: "44G", source: "rule", why: "重みが RAM に収まらないので、入る分だけ RAM に置きます。", why_en: "The weights do not fit RAM: keep what fits in RAM." },
        ],
      };
    }
    return {
      start, cancel() { state = "cancelled"; push("phase", { phase: "cancelled" }); clearInterval(timer); },
      status(since) { return { run: id, state, result, error: null, events: events.filter((e) => e.seq > since) }; },
    };
  })();

  // ---------------------------------------------------------------- go
})();
