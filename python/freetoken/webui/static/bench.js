// The benchmark page: run the measurement job in ft mgr (webui/tuner.py), draw every reading as it
// arrives, then show what was measured and the flags it settled on, ready to put in a profile.
"use strict";

FTI18N.add({
  bm_title: "ベンチマークで設定を決める",
  bm_hint: "この PC とモデルで実際に測り、その結果から ft serve のフラグを決めます。\nPCIe・メモリ・SSD と、CPU・GPU でのエキスパート処理の速さを測り、モデルを起動して設定を比べます。\n終わったら、結果をそのままプロファイルにできます。",
  bm_trials: "モデルを起動して実測する（おすすめ）",
  bm_trials_hint: "モデルを何度も起動して、設定を 1 つずつ変え、速くなったものだけを残します。",
  bm_trials_off: "外すと、ハードウェアだけ測ります（数分）。",
  bm_start: "測定を始める", bm_last: "前回の結果を見る", bm_last_when: "前回: {when}",
  bm_running_title: "測定中", bm_cancel: "中止",
  bm_confirm: "測定を始めます。GPU を使うので、動いているサーバ（{model}）は止まり、終わったら同じ設定で起動し直します。",
  bm_confirm_none: "測定を始めます。測定中は GPU を使います。",
  bm_confirm_time: "かかる時間の目安: {min}。途中で中止できます。",
  bm_time_hw: "数分", bm_time_trials: "10〜25 分",
  bm_view_only: "この画面から測定を始めるには、この PC で開くかトークンが必要です。",
  bm_external: "ft mgr の管理外で起動したサーバが動いていて、GPU を使っています。止めてから測定してください。",
  bm_busy: "ほかの測定が動いています。",
  bm_ph_stop: "動いているサーバを止めています", bm_ph_upstream: "upstream の測定（ft bench bw）をしています", bm_ph_hw: "ハードウェアを測っています",
  bm_s_upstream: "upstream のハードウェア測定（ft bench bw・GPU {gpu}）",
  bm_prof_title: "エンジンが読む upstream の測定値（ft bench bw）", bm_prof_line: "GPU {gpu} · {name}: 版 {version} で測定（{when}）",
  bm_prof_unknown: "GPU {gpu} · {name}: 測った版の記録なし（デスクトップアプリか古い ft bench bw）、{when}",
  bm_prof_none: "まだありません。", bm_prof_stale: "今の版（{current}）では測っていません。",
  bm_prof_rerun: "ベンチマークは毎回、最初に ft bench bw をそのまま実行してこの値を作り直します。",
  bm_upstream_tile: "upstream の測定（ft bench bw）", bm_upstream_sub: "版 {version} で測り直し", bm_ph_trial: "モデルを起動して測っています",
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
  bm_col_load: "起動", bm_col_prefill: "プロンプト処理", bm_col_decode: "生成（文章）", bm_col_hit: "GPU で足りた割合", bm_chosen: "最終",
  bm_trial_failed: "起動か測定に失敗", bm_col_change: "変えたところ", bm_rejected: "試して不採用",
  bm_trials_hint2: "設定 A から 1 つずつ変えて測り、採用したものだけを次に引き継ぎます。生成は文章とコードで測り、主な使い方で判定します。",
  bm_change_base: "基本の設定", bm_change_ctx: "コンテキスト長 {n}", bm_change_without: "{flag} なし",
  bm_flags_title: "決まったフラグ", bm_src_measured: "実測", bm_src_rule: "目安", bm_removed: "外した",
  bm_flags_note: "「実測」はこの測定の数字から、「目安」は GPU・RAM・モデルの構成から決めた値です。",
  bm_saved: "プロファイルを保存しました。",
  bm_empty_hint: "モデルを選んで「測定を始める」を押してください。", bm_loading_models: "モデルを読み込み中…",
  bm_mode: "測り方", bm_mode_time: "かかる時間は GPU とモデルしだいです（例: RTX 3060×2 と Flash-Next で、標準 1.5〜2 時間・徹底 3 時間前後）。測定中は、それまでの 1 回の長さから残り時間を出します。",
  bm_mode_standard: "標準", bm_mode_standard_sub: "エキスパートの置き方とスレッド数、カーネル、KV、プロンプト処理のチャンクと片分け、層の分け方、MTP",
  bm_mode_thorough: "徹底", bm_mode_thorough_sub: "標準に加えて、細かい設定も全部（足りないエキスパートを CPU だけで計算する、KV q4_0、GPU 内のエキスパートの使い回し、埋め込みの置き場、dense の精度、2 枚目への先送りなど）",
  bm_use: "主な使い方", bm_use_sub: "MTP のように、文章の種類で効き方が変わる設定を採用するかどうかに使います。",
  bm_use_both: "両方", bm_use_code: "コード・ツール呼び出し（エージェント）", bm_use_prose: "文章・会話",
  bm_time_standard: "30 分〜2 時間（GPU とモデルしだい）", bm_time_thorough: "1〜3 時間以上（GPU とモデルしだい）",
  bm_planned: "予定", bm_eta: "残り約 {min} 分", bm_eta_soon: "残りわずか", bm_axis_threads: "スレッド数", bm_fig_prefill: "プロンプト", bm_fig_prose: "文章", bm_fig_code: "コード", bm_row_loading: "読み込み", bm_row_prefill: "プロンプト処理", bm_row_decode: "生成（文章）", bm_row_decode_code: "生成（コード）",
  bm_dec_kept: "採用", bm_dec_rejected: "不採用", bm_dec_failed: "失敗", bm_dec_base: "基準", bm_dec_recheck: "速かったのでもう一度測る", bm_dec_rebase: "生成だけ落ちたので基準を測り直す", bm_dec_tiebreak: "結果が割れたので 3 回目を測る",
  bm_col_decode_code: "生成（コード）", bm_col_decision: "判定",
  bm_skipped_title: "測らなかった項目と理由",
  bm_plan_note: "{n} 個の候補を 1 つずつ試します（採用したものによって、あとの候補が増減します）。",
}, {
  bm_title: "Benchmark to pick the settings",
  bm_hint: "Measure this PC with the model and pick ft serve's flags from the numbers.\nPCIe, memory and SSD rates and how fast experts run on the CPU and reach the GPU are measured, then the model is started to compare settings.\nThe result becomes a profile in one click.",
  bm_trials: "Start the model and measure it (recommended)",
  bm_trials_hint: "Starts the model again and again, changes one setting at a time and keeps only what measured faster.",
  bm_trials_off: "Unticked, only the hardware is measured (a few minutes).",
  bm_start: "Start", bm_last: "Show the last result", bm_last_when: "Last: {when}",
  bm_running_title: "Measuring", bm_cancel: "Cancel",
  bm_confirm: "The measurement uses the GPU, so the running server ({model}) stops and is started again with the same settings afterwards.",
  bm_confirm_none: "The measurement uses the GPU.",
  bm_confirm_time: "Expected time: {min}. You can cancel at any point.",
  bm_time_hw: "a few minutes", bm_time_trials: "10-25 minutes",
  bm_view_only: "Starting a benchmark from this page needs this PC or the token.",
  bm_external: "A server started outside ft mgr is running and holds the GPU. Stop it first.",
  bm_busy: "Another benchmark is running.",
  bm_ph_stop: "Stopping the running server", bm_ph_upstream: "Running upstream's measurement (ft bench bw)", bm_ph_hw: "Measuring the hardware",
  bm_s_upstream: "Upstream's hardware measurement (ft bench bw, GPU {gpu})",
  bm_prof_title: "Upstream's measurements the engine reads (ft bench bw)", bm_prof_line: "GPU {gpu} · {name}: measured with {version} ({when})",
  bm_prof_unknown: "GPU {gpu} · {name}: no record of the version (the desktop app or an older ft bench bw), {when}",
  bm_prof_none: "None yet.", bm_prof_stale: "Not measured with this version ({current}).",
  bm_prof_rerun: "Every benchmark first runs ft bench bw as it is and rebuilds these.",
  bm_upstream_tile: "Upstream's measurement (ft bench bw)", bm_upstream_sub: "measured again with {version}", bm_ph_trial: "Measuring the model",
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
  bm_col_load: "Start", bm_col_prefill: "Prompt processing", bm_col_decode: "Generation (prose)", bm_col_hit: "Served from the GPU", bm_chosen: "final",
  bm_trial_failed: "failed to start or measure", bm_col_change: "Changed", bm_rejected: "tried, not kept",
  bm_trials_hint2: "Each run changes one thing from the best so far; only what is kept is carried on. Generation is measured on prose and code, and the main use decides.",
  bm_change_base: "base settings", bm_change_ctx: "context {n}", bm_change_without: "without {flag}",
  bm_flags_title: "The flags", bm_src_measured: "measured", bm_src_rule: "rule", bm_removed: "dropped",
  bm_flags_note: "“measured” values come from this run's numbers, “rule” values from the GPU, RAM and the model's config.",
  bm_saved: "The profile was saved.",
  bm_empty_hint: "Pick a model and press Start.", bm_loading_models: "Loading models…",
  bm_mode: "How much to try", bm_mode_time: "The time depends on the GPUs and the model (two RTX 3060s with Flash-Next: 1.5-2 hours standard, about 3 thorough). While it runs, the time left is worked out from the runs so far.",
  bm_mode_standard: "Standard", bm_mode_standard_sub: "where experts run and CPU threads, kernels, the KV cache, prefill chunks and pieces, the layer split, MTP",
  bm_mode_thorough: "Thorough", bm_mode_thorough_sub: "standard plus every finer setting (computing all missing experts on the CPU, a q4_0 KV cache, reusing experts already on the GPU, where embeddings live, dense precision, sending ahead to the next GPU, and more)",
  bm_use: "Main use", bm_use_sub: "Decides settings like MTP whose effect depends on the kind of text.",
  bm_use_both: "Both", bm_use_code: "Code and tool calls (agents)", bm_use_prose: "Prose and chat",
  bm_time_standard: "30 minutes to 2 hours (depends on the GPUs and the model)", bm_time_thorough: "1 to 3 hours or more (depends on the GPUs and the model)",
  bm_planned: "planned", bm_eta: "about {min} min left", bm_eta_soon: "almost done", bm_axis_threads: "threads", bm_fig_prefill: "prompt", bm_fig_prose: "prose", bm_fig_code: "code", bm_row_loading: "loading", bm_row_prefill: "prompt", bm_row_decode: "generation (prose)", bm_row_decode_code: "generation (code)",
  bm_dec_kept: "kept", bm_dec_rejected: "not kept", bm_dec_failed: "failed", bm_dec_base: "base", bm_dec_recheck: "faster: measuring again", bm_dec_rebase: "generation alone fell: measuring the base again", bm_dec_tiebreak: "the runs disagree: a third run",
  bm_col_decode_code: "Generation (code)", bm_col_decision: "Verdict",
  bm_skipped_title: "Not measured, and why",
  bm_plan_note: "{n} candidates, one at a time (what is kept can open or close later ones).",
});

(async () => {
  const { fmt, esc, $, t, modelName, homePath } = FT;
  await FT.init();
  FT.header("bench");
  const managed = FT.mode === "mgr", demo = FT.mode === "demo";
  const lang = () => FTI18N.lang;

  // ---------------------------------------------------------------- run view state
  let steps = new Map(), order = [], current = null, run = null, seq = 0, startedAt = null, trialCtx = {}, eta = null, done = false;
  const gauge = { shown: 0, target: 0, max: 10, unit: "GB/s", label: "" };

  function resetRun() {
    steps = new Map(); order = []; current = null; seq = 0; startedAt = null; trialCtx = {}; eta = null; done = false;
    gauge.shown = gauge.target = 0;
    $("#bm-steps").innerHTML = "";
  }

  function addStep(id, label, unit, max, hidden = false) {
    if (steps.has(id)) { const s = steps.get(id); s.label = label; renderSteps(); return s; }
    const s = { id, label, unit, max, state: "wait", value: null, samples: [], note: "" };
    steps.set(id, s);
    if (!hidden) order.push(id);
    renderSteps();
    return s;
  }

  const unitOf = (id) => /\.(prefill|decode|decode_code)$/.test(id) ? "tok/s" : /\.load$/.test(id) ? "%" : "GB/s";
  const SUBS = ["load", "prefill", "decode", "decode_code"];
  // a run's row: what it changed, where it is, and at the end its figures and verdict
  function runNote(row) {
    const T = row.id;
    if (row.decision) {
      const p = steps.get(`${T}.prefill`)?.value, d = steps.get(`${T}.decode`)?.value, c = steps.get(`${T}.decode_code`)?.value;
      const figs = [[p, "bm_fig_prefill"], [d, "bm_fig_prose"], [c, "bm_fig_code"]].filter(([v]) => v != null)
        .map(([v, k]) => `${t(k)} ${v >= 100 ? Math.round(v) : v.toFixed(1)}`).join(" · ") + (p != null || d != null ? " tok/s" : "");
      return figs || t(`bm_dec_${row.decision}`);
    }
    for (const k of [...SUBS].reverse()) {
      const s = steps.get(`${T}.${k}`);
      if (!s || s.state === "wait") continue;
      if (k === "load") return s.note || `${t("bm_row_loading")} ${s.since ? Math.floor((Date.now() - s.since) / 1000) + " s" : ""}`;
      if (k === "prefill" && s.progress) return `${t("bm_row_prefill")} ${fmt.num(s.progress.done)} / ${fmt.num(s.progress.total)}`;
      const v = s.samples.length ? s.samples[s.samples.length - 1].v : s.value;
      return `${t(`bm_row_${k}`)}${v != null ? " " + fmtVal(v, "tok/s") : ""}`;
    }
    return "";
  }
  const fmtVal = (v, unit) => v == null ? "—" : unit === "%" ? `${Math.round(v)}%` : unit === "tok/s" ? `${v >= 100 ? Math.round(v) : v.toFixed(1)} tok/s` : `${v.toFixed(1)} GB/s`;

  function hwLabel(id) {
    const m = /^(pcie|gather|upstream)(\d+)$/.exec(id);
    return m ? t(`bm_s_${m[1]}`, { gpu: m[2] }) : t(`bm_s_${id}`);
  }

  function renderSteps() {
    $("#bm-steps").innerHTML = order.map((id) => {
      const s = steps.get(id);
      const ico = { wait: "", run: "●", done: "✓", skip: "!", failed: "!" }[s.state];
      const running = s.state === "run" && s.since ? `${Math.floor((Date.now() - s.since) / 1000)} s` : "";
      const val = s.run ? runNote(s) : s.planned ? t("bm_planned") : s.state === "done" && s.value != null ? fmtVal(s.value, s.unit) : (s.note || running);
      const cls = s.decision === "rejected" || s.decision === "failed" ? "skip" : s.state;
      if (s.run || s.planned) {
        const mark = s.decision ? ` <span class="pill ${{ kept: "ok", base: "info", recheck: "info", rebase: "info", tiebreak: "info", rejected: "mute", failed: "bad" }[s.decision]}">${esc(t(`bm_dec_${s.decision}`))}</span>` : "";
        return `<li class="${cls} two${s.planned ? " planned" : ""}"><span class="ico">${ico}</span><span class="grow"><span class="bm-row-title">${esc(s.label)}${mark}</span>
          ${s.detail ? `<span class="bm-row-detail">${esc(s.detail)}</span>` : ""}<span class="val2">${esc(val || "")}</span></span></li>`;
      }
      return `<li class="${cls}"><span class="ico">${ico}</span><span>${esc(s.label)}</span><span class="val">${esc(val || "")}</span></li>`;
    }).join("");
    const ul = $("#bm-steps"), el = ul.querySelector("li.run");
    if (el && (el.offsetTop < ul.scrollTop || el.offsetTop + el.offsetHeight > ul.scrollTop + ul.clientHeight)) ul.scrollTop = el.offsetTop - ul.clientHeight / 3;
  }

  // "X（Y）" or "X。Y" -> X on the first line, Y under it: the explanations are long and wrap badly inline
  function splitWhat(text) {
    const s = String(text || "");
    let m = /^(.+?)。(.+)$/.exec(s);
    if (m) return { title: m[1], detail: m[2] };
    m = /^(.+?)（(.+)）$/.exec(s) || /^(.+?) \((.+)\)$/.exec(s);
    if (m) return { title: m[1], detail: m[2].replace(/）\s*（/g, " · ").replace(/\)\s*\(/g, " · ") };
    m = /^(.+?): (.+)$/.exec(s);  // English: "one CPU thread fewer (7): the core left free ..."
    return m && lang() === "en" ? { title: m[1], detail: m[2] } : { title: s, detail: "" };
  }

  // the caption under the dial: what is measured now, and for a run what that run changed
  function setNow(main, detail = "") {
    $("#bm-now").innerHTML = `<div>${esc(main)}</div>${detail ? `<div class="bm-now-sub">${esc(detail)}</div>` : ""}`;
  }

  function focus(id) {
    current = id;
    const s = steps.get(id);
    gauge.unit = s.unit; gauge.label = s.label;
    gauge.target = 0;
    gauge.max = niceMax(Math.max(s.max || 0, ...s.samples.map((x) => x.v)) || (s.unit === "tok/s" ? 50 : s.unit === "%" ? 100 : 5), s.unit);
    setNow(s.label, s.caption);
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
    s.samples.push({ v, at: Date.now(), ...extra });
    if (current !== id) focus(id);
    gauge.target = v;
    if (v > gauge.max * 0.95) gauge.max = niceMax(v, s.unit);
    setNow(extra.threads ? `${s.label} · ${t("bm_threads", { n: extra.threads })}` : s.label, s.caption);
    if (s.unit === "%") s.note = id.startsWith("upstream") ? `${Math.round(v)}%${extra.label ? " · " + extra.label : ""}` : t("bm_loading_pct", { v: Math.round(v) });
    renderSteps();
    drawSpark();
  }

  // ---------------------------------------------------------------- events from the job
  function onEvent(e) {
    startedAt = startedAt || e.t;
    if (e.k === "phase") {
      $("#bm-phase").textContent = t(`bm_ph_${e.phase}`) + (e.message ? ` — ${e.message}` : "");
      if (["done", "cancelled", "error"].includes(e.phase)) {
        done = true;
        for (const s of steps.values()) if (s.state === "run") s.state = e.phase === "done" ? "done" : "failed";
        for (const id of order.filter((id) => steps.get(id).planned)) { steps.delete(id); order.splice(order.indexOf(id), 1); }
      }
      renderSteps();
    } else if (e.k === "hw") {
      if (e.kind === "plan") for (const s of e.steps) addStep(s.id, hwLabel(s.id), s.unit || "GB/s", s.max);
      else if (e.kind === "step") { const s = steps.get(e.id) || addStep(e.id, hwLabel(e.id), "GB/s"); s.state = "run"; focus(e.id); renderSteps(); }
      else if (e.kind === "sample") sample(e.id, e.value, e.threads ? { threads: e.threads } : e.label ? { label: e.label } : {});
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
    } else if (e.k === "plan") {
      // what is still to come: rows no longer in the plan go, new candidates are added at the end
      const still = new Set(e.items.map((it) => `plan:${it.key}`));
      for (const id of order.filter((id) => steps.get(id)?.planned && !still.has(id))) {
        steps.delete(id); order.splice(order.indexOf(id), 1);
      }
      for (const it of e.items) {
        if (steps.has(`plan:${it.key}`)) continue;
        const w = splitWhat(lang() === "en" ? it.what_en : it.what);
        const s = addStep(`plan:${it.key}`, w.title, "tok/s");
        Object.assign(s, { planned: true, detail: w.detail });
      }
      if (!e.update) setNow(t("bm_plan_note", { n: e.items.length }));
      eta = e.eta_s != null ? { s: e.eta_s, at: Date.now() } : eta;
      renderSteps();
    } else if (e.k === "decision") {
      const row = steps.get(e.trial);
      if (row) { row.decision = e.decision; row.state = ["kept", "recheck", "rebase", "base", "tiebreak"].includes(e.decision) ? "done" : "skip"; renderSteps(); }
    } else if (e.k === "trial") {
      const T = e.trial;
      if (e.step === "load" && e.state === "start") {
        trialCtx[T] = e.change || null;
        const w = splitWhat((lang() === "en" ? e.what_en : e.what) || changeText(e.change));
        const label = `${t("bm_col_setting")} ${T}: ${w.title}`;
        const planned = e.key ? order.indexOf(`plan:${e.key}`) : -1;
        if (planned >= 0) { steps.delete(`plan:${e.key}`); order.splice(planned, 1, T); }
        const row = steps.get(T) || addStep(T, label, "tok/s", null, true);
        if (!order.includes(T)) {
          let last = -1;
          order.forEach((id, i) => { if (!steps.get(id)?.planned) last = i; });
          order.splice(last + 1, 0, T);
        }
        Object.assign(row, { id: T, label, detail: w.detail, run: true, state: "run", since: Date.now() });
        steps.set(T, row);
        for (const k of SUBS) {
          const sub = addStep(`${T}.${k}`, `${t("bm_col_setting")} ${T} · ${t(`bm_row_${k === "load" ? "loading" : k}`)}`, unitOf(`${T}.${k}`), k === "load" ? 100 : null, true);
          sub.caption = w.title;
        }
      }
      const row = steps.get(T);
      if (e.state === "failed") {
        if (row) { row.decision = "failed"; row.state = "skip"; }
        renderSteps();
        return;
      }
      const s = steps.get(`${T}.${e.step}`);
      if (!s) {
        if (row && e.step === "measured") { row.state = "done"; if (T === "A" && !row.decision) row.decision = "base"; renderSteps(); }
        return;
      }
      if (e.state === "start") { s.state = "run"; s.since = Date.now(); s.progress = null; focus(s.id); }
      else if (e.state === "done") { s.state = "done"; s.value = e.step === "load" ? null : e.value; if (e.step === "load") s.note = `${t("bm_row_loading")} ${e.seconds} s`; }
      renderSteps();
    } else if (e.k === "sample") { const s = steps.get(e.id); if (s) s.progress = null; if (s) sample(e.id, e.value); }
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

  // the readings of the step on the dial, with axes: values on the left, seconds (or threads) below
  function drawSpark() {
    const cv = $("#bm-spark");
    const s = current && steps.get(current);
    const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // a load percentage only ever climbs: a line of it says nothing the dial does not
    if (!s || !s.samples.length || s.unit === "%") return;
    const vals = s.samples.map((x) => x.v);
    const top = niceMax(Math.max(...vals) || 1, s.unit);
    const exp = Math.pow(10, Math.floor(Math.log10(top) + 1e-9));
    const lead = Math.round((top / exp) * 10) / 10;
    const divs = [2.5, 5, 10].includes(lead) ? 5 : 4;
    const L = 46, R = 12, T = 18, B = 22, pw = w - L - R, ph = h - T - B;
    const y = (v) => T + ph * (1 - v / top);
    const num = (v) => (s.unit === "tok/s" ? (v >= 100 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1))
      : v >= 10 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1));
    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 1;
    // horizontal grid and the value axis
    for (let k = 0; k <= divs; k++) {
      const v = (top * k) / divs, yy = Math.round(y(v)) + 0.5;
      ctx.strokeStyle = css(k === 0 ? "--fg3" : "--line");
      ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(w - R, yy); ctx.stroke();
      ctx.fillStyle = css("--fg3"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(num(v), L - 6, yy);
    }
    ctx.strokeStyle = css("--fg3");
    ctx.beginPath(); ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph); ctx.stroke();
    ctx.fillStyle = css("--fg3"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(s.unit, 2, 0);

    if (s.samples[0].threads) {
      // the thread sweep: one bar per thread count, its value on top
      const bw = pw / vals.length;
      s.samples.forEach((x, i) => {
        const x0 = L + i * bw, bh = (ph * x.v) / top;
        ctx.fillStyle = css("--accent"); ctx.fillRect(x0 + bw * 0.18, T + ph - bh, bw * 0.64, bh);
        ctx.fillStyle = css("--fg2"); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        ctx.fillText(x.v < 100 ? x.v.toFixed(1) : String(Math.round(x.v)), x0 + bw / 2, T + ph - bh - 2);
        ctx.fillStyle = css("--fg3"); ctx.textBaseline = "top";
        ctx.fillText(x.threads, x0 + bw / 2, T + ph + 5);
      });
      ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText(t("bm_axis_threads"), w - R, 0);
      return;
    }

    // over time: seconds since the step's first reading
    const t0 = s.samples[0].at || 0;
    const span = Math.max(1, ((s.samples[s.samples.length - 1].at || t0) - t0) / 1000);
    const xAt = (x) => (s.samples.length > 1 ? L + (pw * ((x.at || t0) - t0)) / 1000 / span : L + pw / 2);
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((d) => span / d <= 6) || 600;
    ctx.fillStyle = css("--fg3"); ctx.textBaseline = "top";
    for (let sec = 0; sec <= span + 1e-6; sec += step) {
      const xx = Math.round(L + (pw * sec) / span) + 0.5;
      ctx.strokeStyle = css("--line"); ctx.beginPath(); ctx.moveTo(xx, T + ph); ctx.lineTo(xx, T + ph + 4); ctx.stroke();
      ctx.textAlign = sec === 0 ? "left" : xx > w - R - 20 ? "right" : "center";
      ctx.fillText(sec >= 60 && sec % 60 === 0 ? `${sec / 60} min` : `${sec} s`, xx, T + ph + 6);
    }
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.beginPath();
    s.samples.forEach((x, i) => (i ? ctx.lineTo(xAt(x), y(x.v)) : ctx.moveTo(xAt(x), y(x.v))));
    ctx.stroke();
    if (s.samples.length <= 24) {
      ctx.fillStyle = css("--accent");
      s.samples.forEach((x) => { ctx.beginPath(); ctx.arc(xAt(x), y(x.v), 2.5, 0, Math.PI * 2); ctx.fill(); });
    }
    // the latest reading, next to its point
    const last = s.samples[s.samples.length - 1], lx = xAt(last), ly = y(last.v);
    ctx.fillStyle = css("--fg"); ctx.font = "600 11px system-ui, sans-serif";
    ctx.textAlign = lx > w - R - 60 ? "right" : "left"; ctx.textBaseline = ly < T + 14 ? "top" : "bottom";
    ctx.fillText(`${num(last.v)} ${s.unit}`, lx + (ctx.textAlign === "right" ? -6 : 6), ly + (ctx.textBaseline === "top" ? 4 : -4));
  }

  (function loop() { drawGauge(); requestAnimationFrame(loop); })();
  setInterval(() => {
    if (!$("#bm-run").hidden && [...steps.values()].some((s) => s.state === "run" && s.since && !s.note)) renderSteps();
    if (startedAt && !$("#bm-run").hidden) {
      const s = Math.max(0, Math.round(Date.now() / 1000 - startedAt));
      const left = eta && !done ? Math.max(0, eta.s - (Date.now() - eta.at) / 1000) : null;
      const etaText = left == null ? "" : ` · ${left < 90 ? t("bm_eta_soon") : t("bm_eta", { min: Math.round(left / 60) })}`;
      $("#bm-elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}${etaText}`;
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
  async function loadProfiles() {
    const box = $("#bm-profiles");
    if (demo) { box.hidden = true; return; }
    let doc;
    try { doc = await FT.raw("/tune/profiles"); } catch { box.hidden = true; return; }
    const when = (e) => e ? new Date(e * 1000).toLocaleString(lang() === "ja" ? "ja-JP" : "en-US") : "—";
    const lines = (doc.profiles || []).map((p) => {
      const name = (p.gpu || "").replace(/^NVIDIA\s+/, "");
      const text = p.version ? t("bm_prof_line", { gpu: p.index, name, version: p.version, when: when(p.epoch) })
        : t("bm_prof_unknown", { gpu: p.index, name, when: when(p.epoch) });
      const stale = p.version !== doc.current ? ` <span class="warn-text">${esc(t("bm_prof_stale", { current: doc.current }))}</span>` : "";
      return `<div>${esc(text)}${stale}</div>`;
    });
    box.innerHTML = `<b>${esc(t("bm_prof_title"))}</b><span class="muted">${esc(t("bm_prof_rerun"))}</span>
      ${lines.join("") || `<div>${esc(t("bm_prof_none"))}</div>`}`;
  }
  loadProfiles();

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
  $("#bm-trials").onchange = () => { $("#bm-opts").hidden = !$("#bm-trials").checked; };
  $("#bm-last").onclick = () => { show("result"); renderResult(lastResult); };
  loadLast();

  $("#bm-start").onclick = async () => {
    const trials = $("#bm-trials").checked;
    let running = null;
    $("#bm-opts").hidden = !trials;
    $("#bm-start").disabled = true;  // the health check can take a second or two
    try { running = (await FT.serveGet("/health")).running ? engineCfg?.model : null; } catch {}
    $("#bm-start").disabled = false;
    const mode = document.querySelector('input[name="bm-mode"]:checked')?.value || "standard";
    const use = document.querySelector('input[name="bm-use"]:checked')?.value || "both";
    const msg = [running ? t("bm_confirm", { model: modelName(running) }) : t("bm_confirm_none"),
      t("bm_confirm_time", { min: t(trials ? `bm_time_${mode}` : "bm_time_hw") })].join("\n\n");
    if (!await FT.ask(msg, { title: t("bm_title"), ok: t("bm_start") })) return;
    if (demo) { DemoJob.start(trials); run = null; show("run"); return; }
    try {
      await FT.raw("/tune/start", { method: "POST", json: { model: $("#bm-model").value, trials, mode, use } });
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
    if (r.upstream?.length) tile(t("bm_upstream_tile"), r.upstream.map((u) => `GPU ${u.gpu}`).join(", "),
      t("bm_upstream_sub", { version: r.upstream[0].version }));
    if (hw.overlap?.fetch_fraction != null) tile(t("bm_overlap"), `${(hw.overlap.pcie_gbs + hw.overlap.cpu_gbs).toFixed(1)} <small>GB/s</small>`,
      t("bm_overlap_sub", { pct: fmt.pct(hw.overlap.fetch_fraction, 0) }));

    const trials = r.trials || [];
    const chosen = r.chosen;
    const okTrials = trials;
    const anyCode = trials.some((x) => x.decode_code_tps);
    const decisionPill = (x) => {
      const d = x.decision || (x.ok ? null : "failed");
      if (!d) return "";
      const kind = d === "kept" ? "ok" : ["base", "recheck", "rebase", "tiebreak"].includes(d) ? "info" : d === "failed" ? "bad" : "mute";
      return `<span class="pill ${kind}">${esc(t(`bm_dec_${d}`))}</span>`;
    };
    const whatOf = (x) => (x.what ? (lang() === "en" ? x.what[1] : x.what[0]) : changeText(x.change));
    const trialTable = okTrials.length ? `<div class="card"><h2>${esc(t("bm_trials_title"))}</h2>
      <p class="hint">${esc(t("bm_trials_hint2"))}</p>
      <div class="tablewrap"><table class="bm-trials"><thead><tr><th>${esc(t("bm_col_setting"))}</th><th>${esc(t("bm_col_change"))}</th><th>${esc(t("bm_col_decision"))}</th>
        <th class="num">${esc(t("bm_col_ctx"))}</th><th class="num">${esc(t("bm_col_slots"))}</th><th class="num">${esc(t("bm_col_load"))}</th>
        <th class="num">${esc(t("bm_col_prefill"))}</th><th class="num">${esc(t("bm_col_decode"))}</th>${anyCode ? `<th class="num">${esc(t("bm_col_decode_code"))}</th>` : ""}<th class="num">${esc(t("bm_col_hit"))}</th></tr></thead>
      <tbody>${okTrials.map((x) => {
        const win = x.ok && x.label === chosen ? "win" : "";
        return `<tr><td class="${win}"><b>${esc(x.label)}</b> ${win ? `<span class="pill ok">${esc(t("bm_chosen"))}</span>` : ""}</td>
          <td class="${win}" style="white-space:normal;min-width:220px">${esc(whatOf(x))}<div class="small muted"><code style="font-size:11px">${esc(x.change ? changeText(x.change) : "")}</code></div></td>
          <td class="${win}">${decisionPill(x)}</td>
          <td class="num ${win}">${x.kv_tokens ? fmt.num(x.kv_tokens) : esc(argValue(x.args, "--kv-reserve-tokens") || "—")}</td>
          <td class="num ${win}">${x.expert_slots != null ? fmt.num(x.expert_slots) : "—"}</td>
          <td class="num ${win}">${x.load_s != null ? `${x.load_s} s` : "—"}</td>
          <td class="num ${win}">${fmtVal(x.prefill_tps, "tok/s")}</td>
          <td class="num ${win}">${fmtVal(x.decode_tps, "tok/s")}</td>
          ${anyCode ? `<td class="num ${win}">${fmtVal(x.decode_code_tps, "tok/s")}</td>` : ""}
          <td class="num ${win}">${fmt.pct(x.hit_rate)}</td></tr>`;
      }).join("")}</tbody></table></div></div>` : "";

    const rank = (n) => (n.rejected ? 2 : n.source === "measured" ? 0 : 1);
    const notes = (r.notes || []).slice().sort((a, b) => rank(a) - rank(b));
    const skipped = (r.skipped || []).length ? `<div class="card"><h2>${esc(t("bm_skipped_title"))}</h2>
      ${r.skipped.map((s) => `<div class="bm-flag"><span style="flex:1;min-width:0"><code>${esc(s.flag)}</code>
        <div class="small muted" style="margin-top:2px">${esc(lang() === "en" ? s.why_en : s.why)}</div></span></div>`).join("")}</div>` : "";
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
      ${flags}
      ${skipped}`;
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
  // [label, key, what, what_en, change, prefill, prose, code, decision]
  const DEMO_RUNS = [
    ["A", null, null, null, null, 548, 18.1, 24.0, "base"],
    ["B", "strategy_offload", "エキスパートを CPU で計算せず、GPU に転送して計算する", "compute experts on the GPU (fetched) instead of the CPU", { flag: "--moe-strategy", value: "offload" }, 402, 12.2, 15.8, "rejected"],
    ["C", "kernel_moe_marlin", "エキスパートの計算カーネルを marlin にする", "the marlin expert kernel", { flag: "--quant-backend", value: "moe.nvfp4=marlin" }, 611, 19.0, 25.1, "kept"],
    ["D", "kv_16bit", "KV キャッシュを量子化しない（16 ビット）", "an unquantized (16-bit) KV cache", { flag: "--kv-cache-dtype", value: null }, 598, 18.7, 24.6, "rejected"],
    ["E", "budget_075", "プリフィルのチャンク予算を 0.75 にする", "a prefill chunk budget of 0.75", { flag: "--prefill-chunk-budget", value: "0.75" }, 702, 19.1, 25.0, "kept"],
    ["F", "mtp_3", "MTP で 3 トークン先まで予測する", "MTP drafting 3 tokens ahead", { flag: "--spec-mtp", value: "3" }, 690, 17.2, 38.4, "kept"],
    ["G", "mtp_5", "MTP で 5 トークン先まで予測する", "MTP drafting 5 tokens ahead", { flag: "--spec-mtp", value: "5" }, 688, 16.1, 41.9, "kept"],
    ["H", "context_131072", "コンテキスト長を 131,072 にする", "a context of 131,072 tokens", { flag: "--kv-reserve-tokens", value: "131072" }, 681, 16.0, 41.2, "kept"],
  ];
  const DEMO_PLAN = DEMO_RUNS.slice(1, 7).map(([, key, what, what_en, change]) => ({ key, what, what_en, change }))
    .concat([{ key: "overlap_on", what: "プリフィルで CPU 計算と転送を重ねる", what_en: "overlap CPU compute and transfers in prefill", change: null }]);
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
        const R = DEMO_RUNS;
        at(ms += 500, () => push("phase", { phase: "trial" }));
        for (const [T, key, what, what_en, change, pre, dec, code, decision] of R) {
          at(ms += 500, () => push("trial", { trial: T, step: "load", state: "start", change, key, what, what_en }));
          if (T === "A") at(ms += 10, () => {});
          for (let p = 10; p <= 100; p += 10) at(ms += 100, () => push("sample", { id: `${T}.load`, value: p }));
          at(ms += 200, () => push("trial", { trial: T, step: "load", state: "done", seconds: 90 + Math.round(Math.random() * 20) }));
          at(ms += 150, () => push("trial", { trial: T, step: "prefill", state: "start", tokens: 16384 }));
          for (let c = 1; c <= 3; c++) at(ms += 350, () => push("progress", { id: `${T}.prefill`, done: Math.min(16384, 5460 * c), total: 16384, rate: pre }));
          at(ms += 200, () => { push("sample", { id: `${T}.prefill`, value: pre }); push("trial", { trial: T, step: "prefill", state: "done", value: pre }); });
          for (const [k, v] of [["decode", dec], ["decode_code", code]]) {
            if (v == null) continue;
            at(ms += 150, () => push("trial", { trial: T, step: k, state: "start", tokens: 300 }));
            for (let i = 0; i < 6; i++) at(ms += 150, () => push("sample", { id: `${T}.${k}`, value: v * (0.94 + Math.random() * 0.1) }));
            at(ms += 100, () => push("trial", { trial: T, step: k, state: "done", value: v }));
          }
          at(ms += 100, () => push("trial", { trial: T, step: "measured", state: "done" }));
          if (T === "A") at(ms += 50, () => push("plan", { items: DEMO_PLAN }));
          else at(ms += 100, () => push("decision", { trial: T, key, decision }));
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
        model: "/home/demo/models/Qwen3.8-Flash-Next-NVFP4", port: 1919, started: now - 3900, finished: now, chosen: "H", mode: "standard", use: "both",
        args: ["--pp-size", "2", "--gpu", "0,1", "--moe-strategy", "hybrid", "--moe-cpu-threads", "8", "--moe-cache-auto", "--moe-bank-ram", "44G", "--kv-cache-dtype", "q8_0", "--kv-reserve-tokens", "131072", "--max-seq-len-override", "131072"],
        hw: { measurements: {
          gpus: [0, 1].map((i) => ({ index: i, name: "NVIDIA GeForce RTX 3060", pcie_gen: 4, pcie_width: i ? 4 : 8, pcie_width_max: 16, link_gbs: i ? 7.88 : 15.75 })),
          pcie: { 0: { h2d_gbs: 6.2 }, 1: { h2d_gbs: 3.1 } }, ram: { read_gbs: 21.4, threads: 10 }, ssd: { read_gbs: 3.3, read_ahead_kb: 4096 },
          cpu_moe: { best_gbs: 11.8, threads: 8, cores: 10, sweep: { 1: 2.1, 2: 4.0, 4: 7.4, 6: 9.9, 8: 11.2, 10: 11.6 } },
          gather: { 0: { gbs: 5.6 }, 1: { gbs: 2.9 } }, overlap: { cpu_gbs: 9.8, pcie_gbs: 4.3, fetch_fraction: 0.305 },
        } },
        trials: trials ? DEMO_RUNS.map(([label, key, what, what_en, change, pre, dec, code, decision], i) => ({
          label, key, what: what ? [what, what_en] : null, change, decision, ok: true, kv_tokens: label === "H" ? 131072 : 65536,
          expert_slots: 1520 - i * 20, load_s: 95 + i * 3, prefill_tps: pre, decode_tps: dec, decode_code_tps: code, hit_rate: 0.58, args: [] })) : [],
        skipped: [
          { flag: "--memory-ratio", why: "上げると他のアプリが GPU を使ったときに落ちます。速さではなく余裕の問題なので測りません。", why_en: "Raising it crashes when other apps use the GPU; it is headroom, not speed." },
          { flag: "--max-running-req", why: "同時に何人で使うかで決まる値で、速さの測定では決められません。", why_en: "It depends on how many people use the server at once, not on speed." },
        ],
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
