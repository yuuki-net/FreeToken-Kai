// Profile editor (ft mgr console): model picked from what exists on the host, flags picked from the real
// `ft serve` parser. The flag list scrolls on its own and what a flag does stays open in the right pane.
"use strict";

const FTProfileEditor = (() => {
  const { esc, $, fmt, homePath, modelName, t } = FT;
  let metaPromise = null;
  const RESERVED = new Set(["--model", "--model-path", "--port"]);

  function loadMeta() {
    if (!metaPromise) {
      metaPromise = Promise.all([
        FT.raw("/serve-flags").then((d) => d.flags),
        fetch("flags_ja.json", { cache: "no-store" }).then((r) => r.json()),
        FT.raw("/models").then((d) => d.models).catch(() => []),
      ]).then(([flags, ja, models]) => {
        const byName = new Map();
        for (const f of flags) {
          f.ja = ja.flags[f.flag] || null;
          f.cat = f.ja?.cat || "other";
          for (const n of [f.flag, ...(f.aliases || [])]) byName.set(n, f);
        }
        const cats = [["common", "pe_cat_common"], ...ja.categories.map(([k]) => [k, "pe_cat_" + k]), ["other", "pe_cat_other"]];
        return { flags, byName, ja, cats, models };
      }).catch((e) => { metaPromise = null; throw e; });
    }
    return metaPromise;
  }

  // one line about a flag: Japanese when we have it, otherwise the parser's own help
  const summary = (f) => (FTI18N.lang === "ja" ? f.ja?.ja || f.help : f.help || f.ja?.ja || "");

  // argv -> rows, keeping the spelling the user wrote (--model vs --model-path)
  function parseArgs(args, meta) {
    const rows = [];
    for (let i = 0; i < args.length; i++) {
      let tok = args[i], inline = null;
      if (!tok.startsWith("--")) { if (rows.length) rows[rows.length - 1].extra.push(tok); continue; }
      const eq = tok.indexOf("=");
      if (eq > 0) { inline = tok.slice(eq + 1); tok = tok.slice(0, eq); }
      let f = meta.byName.get(tok);
      if (!f) {  // argparse accepts any unambiguous prefix: --max-running-req is --max-running-requests
        const hits = [...new Set([...meta.byName.entries()].filter(([n]) => n.startsWith(tok)).map(([, m]) => m))];
        if (hits.length === 1) f = hits[0];
      }
      const row = { spelling: tok, meta: f || null, value: "", extra: [] };
      if (inline !== null) row.value = inline;
      else if (f && f.kind === "bool") { /* no value */ }
      else if (f && f.multiple) { const vals = []; while (i + 1 < args.length && !args[i + 1].startsWith("--")) vals.push(args[++i]); row.value = vals.join(" "); }
      else if (i + 1 < args.length && !args[i + 1].startsWith("--")) row.value = args[++i];
      rows.push(row);
    }
    return rows;
  }

  function toArgs(rows) {
    const out = [];
    for (const r of rows) {
      out.push(r.spelling);
      if (r.meta?.kind === "bool") continue;
      const v = r.value.trim();
      if (r.meta?.multiple) out.push(...v.split(/\s+/).filter(Boolean));
      else if (v !== "") out.push(v);
      out.push(...r.extra);
    }
    return out;
  }

  function splitShell(s) {
    const out = []; let cur = "", q = null, has = false;
    for (const ch of s.replace(/\\\n/g, " ")) {
      if (q) { if (ch === q) q = null; else cur += ch; }
      else if (ch === '"' || ch === "'") { q = ch; has = true; }
      else if (/\s/.test(ch)) { if (cur || has) out.push(cur); cur = ""; has = false; }
      else cur += ch;
    }
    if (cur || has) out.push(cur);
    return out;
  }
  const quote = (x) => (/[\s"'$]/.test(x) || x === "" ? `'${x.replace(/'/g, "'\\''")}'` : x);

  function open(profile, { onSaved } = {}) {
    const d = document.createElement("dialog");
    d.className = "pe";
    d.innerHTML = `<form method="dialog" class="stack">
      <div style="display:flex;align-items:center"><b style="font-size:15px">${esc(t(profile.name && !profile._new ? "pe_edit" : "pe_new"))}</b><span class="grow"></span><button value="cancel" formnovalidate>${esc(t("close"))}</button></div>
      <div class="grid" style="grid-template-columns:minmax(0,1fr) 110px;gap:10px">
        <label class="small">${esc(t("pe_name"))}<br><input type="text" id="pe-name" style="width:100%" value="${esc(profile.name || "")}"></label>
        <label class="small">${esc(t("pe_port"))}<br><input type="text" id="pe-port" style="width:100%" value="${esc(profile.port ?? "")}" placeholder="1919"></label>
      </div>
      <div class="small">${esc(t("pe_model"))}
        <select id="pe-model" style="width:100%;margin-top:2px"><option value="">${esc(t("pe_model_loading"))}</option></select>
        <input type="text" id="pe-model-path" style="width:100%;margin-top:6px" hidden placeholder="${esc(t("pe_model_ph"))}">
        <div id="pe-model-info" class="muted" style="margin-top:4px"></div>
      </div>
      <div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <b class="small">${esc(t("pe_flags"))}</b><span class="grow"></span>
          <button type="button" id="pe-recommend">${esc(t("pe_recommend"))}</button>
          <button type="button" id="pe-tune" hidden>${esc(t("pe_tune"))}</button>
          <button type="button" id="pe-add">${esc(t("pe_add"))}</button>
          <button type="button" id="pe-paste">${esc(t("pe_paste"))}</button>
        </div>
        <div id="pe-paste-box" hidden class="stack" style="margin-top:8px;gap:6px">
          <textarea id="pe-paste-text" rows="4" style="width:100%;font-family:var(--mono);font-size:12px" placeholder="${esc(t("pe_paste_ph"))}"></textarea>
          <div style="display:flex;gap:8px;justify-content:flex-end;align-items:center"><span class="small muted grow">${esc(t("pe_paste_note"))}</span><button type="button" id="pe-paste-do">${esc(t("pe_paste_do"))}</button></div>
        </div>
        <div class="pe-split" style="margin-top:8px">
          <div class="pe-left">
            <div id="pe-advice" hidden class="card" style="margin-bottom:12px;padding:10px 12px"></div>
        <div id="pe-picker" hidden style="border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:12px">
              <div style="display:flex;gap:6px;flex-wrap:wrap"><input type="text" id="pe-q" placeholder="${esc(t("pe_search_ph"))}" style="flex:1;min-width:160px"><select id="pe-cat" style="width:auto"></select></div>
              <div id="pe-list" style="max-height:200px;overflow:auto;margin-top:6px"></div>
            </div>
            <div id="pe-rows" class="pe-rows"></div>
          </div>
          <div class="pe-right" id="pe-detail"></div>
        </div>
      </div>
      <div class="small"><span class="muted">${esc(t("pe_cmd"))}</span><pre id="pe-preview" class="log" style="height:auto;max-height:96px;margin-top:4px"></pre></div>
      <div id="pe-err" class="small" style="color:var(--bad)"></div>
      <div style="display:flex;gap:8px;justify-content:flex-end"><button value="cancel" formnovalidate>${esc(t("cancel"))}</button><button type="button" id="pe-save" class="primary">${esc(t("save"))}</button></div>
    </form>`;
    document.body.append(d);
    d.addEventListener("close", () => d.remove());
    d.showModal();

    let meta = null, rows = [], selected = -1;
    const modelSel = $("#pe-model", d), modelPath = $("#pe-model-path", d);
    const modelValue = () => (modelSel.value === "__other" ? modelPath.value.trim() : modelSel.value);
    const catLabel = (cat) => { const c = meta?.cats.find((x) => x[0] === cat); return c ? t(c[1]) : ""; };

    function renderModelInfo() {
      const m = meta?.models.find((x) => x.value === modelSel.value);
      modelPath.hidden = modelSel.value !== "__other";
      $("#pe-model-info", d).textContent = m
        ? [fmt.gib(m.size_bytes), m.quant, m.model_type, m.max_context ? fmt.tok(m.max_context) : null].filter(Boolean).join(" · ")
        : modelSel.value === "__other" ? t("pe_model_other_hint") : "";
      preview();
    }

    // ---------------------------------------------------------------- flag rows (left) and detail (right)
    function control(r, i) {
      const f = r.meta;
      if (f?.kind === "bool") return `<span class="pill ok">${esc(t("pe_bool_on"))}</span>`;
      if (f?.kind === "choice" && !f.multiple)
        return `<select data-i="${i}" class="pe-val">${[...(f.choices.includes(r.value) ? [] : [r.value]), ...f.choices].map((c) => `<option ${c === r.value ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
      const ph = f?.multiple ? t("pe_multi_ph") : f?.default != null && f.default !== "" ? t("pe_default_ph", { v: f.default }) : f?.metavar || t("pe_value_ph");
      return `<input type="text" data-i="${i}" class="pe-val" value="${esc(homePath(r.value))}" placeholder="${esc(ph)}">`;
    }

    function renderRows() {
      const box = $("#pe-rows", d);
      if (!rows.length) { box.innerHTML = `<div class="empty small">${esc(t("pe_flags_none"))}</div>`; renderDetail(); preview(); return; }
      // grouped by category, not by the order they were added: a long list is unreadable otherwise
      const order = new Map((meta?.cats || []).map(([k], i) => [k, i]));
      const sorted = rows.map((r, i) => ({ r, i }))
        .sort((a, b) => (order.get(a.r.meta?.cat ?? "other") ?? 99) - (order.get(b.r.meta?.cat ?? "other") ?? 99)
          || a.r.spelling.localeCompare(b.r.spelling));
      let lastCat = null;
      box.innerHTML = sorted.map(({ r, i }) => {
        const cat = r.meta?.cat ?? "other";
        const head = cat === lastCat ? "" : `<div class="pe-cat">${esc(catLabel(cat) || t("pe_cat_other"))}</div>`;
        lastCat = cat;
        return head + `<div class="pe-row ${i === selected ? "on" : ""}" data-row="${i}">
          <code>${esc(r.spelling)}</code>
          ${RESERVED.has(r.spelling) ? `<span class="pill warn">${esc(t("pe_reserved"))}</span>` : ""}
          ${r.meta ? "" : `<span class="pill bad">?</span>`}
          <span class="grow"></span>${control(r, i)}
          <button type="button" data-rm="${i}" aria-label="${esc(t("pe_remove"))}">✕</button>
        </div>`;
      }).join("");
      box.querySelectorAll(".pe-val").forEach((el) => {
        el.addEventListener("input", () => { rows[+el.dataset.i].value = el.value; preview(); });
        el.addEventListener("focus", () => select(+el.dataset.i));
      });
      box.querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const i = +b.dataset.rm;
        rows.splice(i, 1);
        if (selected >= rows.length) selected = rows.length - 1;
        renderRows(); renderList();
      }));
      box.querySelectorAll("[data-row]").forEach((el) => el.addEventListener("click", () => select(+el.dataset.row)));
      renderDetail(); preview();
    }

    function select(i) {
      selected = i;
      $("#pe-rows", d).querySelectorAll("[data-row]").forEach((el) => el.classList.toggle("on", +el.dataset.row === i));
      renderDetail();
    }

    function renderDetail(flag) {
      const f = flag || rows[selected]?.meta;
      const pane = $("#pe-detail", d);
      if (!f) {
        pane.innerHTML = rows[selected] && !rows[selected].meta
          ? `<div class="small" style="color:var(--warn)">${esc(t("pe_unknown_flag"))}</div>`
          : `<div class="small muted">${esc(t("pe_detail_none"))}</div>`;
        return;
      }
      const ja = FTI18N.lang === "ja" ? f.ja : null;
      const facts = [];
      if (f.default != null && f.default !== "" && f.kind !== "bool") facts.push([t("pe_default"), String(f.default)]);
      if (f.choices) facts.push([t("pe_choices"), f.choices.join(" / ")]);
      if (f.aliases?.length) facts.push(["別名", f.aliases.join(" ")]);
      pane.innerHTML = `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <code style="font-weight:600;color:var(--fg)">${esc(f.flag)}</code><span class="pill mute">${esc(catLabel(f.cat))}</span></div>
        <div class="small" style="margin-top:6px">${esc(summary(f))}</div>
        ${ja?.more ? `<div class="small" style="color:var(--fg2);margin-top:6px">${esc(ja.more)}</div>` : ""}
        ${facts.map(([k, v]) => `<div class="small muted" style="margin-top:6px"><b>${esc(k)}</b> ${esc(v)}</div>`).join("")}`;
    }

    function preview() {
      const port = $("#pe-port", d).value.trim();
      const argv = ["ft", "serve", "--model", modelName(modelValue()) || "…", ...(port ? ["--port", port] : []), ...toArgs(rows)];
      $("#pe-preview", d).textContent = homePath(argv.map(quote).join(" "));
    }

    // ---------------------------------------------------------------- flag picker
    function renderList() {
      if (!meta) return;
      const q = $("#pe-q", d).value.trim().toLowerCase(), cat = $("#pe-cat", d).value;
      const have = new Set(rows.map((r) => r.meta?.flag));
      let list = meta.flags.filter((f) => !RESERVED.has(f.flag));
      if (q) list = list.filter((f) => [f.flag, ...(f.aliases || []), f.ja?.ja, f.ja?.more, f.help].join(" ").toLowerCase().includes(q));
      else if (cat === "common") list = meta.ja.common.map((n) => meta.byName.get(n)).filter(Boolean);
      else if (cat) list = list.filter((f) => f.cat === cat);
      const order = new Map(meta.cats.map(([k], i) => [k, i]));
      if (cat !== "common" || q) list = [...list].sort((a, b) => (order.get(a.cat) ?? 99) - (order.get(b.cat) ?? 99) || a.flag.localeCompare(b.flag));
      let lastCat = null;
      $("#pe-list", d).innerHTML = list.length ? list.map((f) => {
        const head = f.cat === lastCat || cat === "common" ? "" : `<div class="pe-cat">${esc(catLabel(f.cat) || t("pe_cat_other"))}</div>`;
        lastCat = f.cat;
        return head + `<button type="button" class="pe-pick" data-add="${esc(f.flag)}" ${have.has(f.flag) ? "disabled" : ""}>
          <code>${esc(f.flag)}</code> ${have.has(f.flag) ? `<span class="pill mute">${esc(t("pe_added"))}</span>` : ""}
          <div class="small" style="color:var(--fg2)">${esc(summary(f))}</div></button>`;
      }).join("") : `<div class="small muted" style="padding:8px">${esc(t("pe_not_found"))}</div>`;
      $("#pe-list", d).querySelectorAll("[data-add]").forEach((b) => {
        b.addEventListener("mouseenter", () => renderDetail(meta.byName.get(b.dataset.add)));
        b.addEventListener("click", () => {
          const f = meta.byName.get(b.dataset.add);
          rows.push({ spelling: f.flag, meta: f, value: f.kind === "choice" && !f.multiple ? String(f.default ?? f.choices[0]) : "", extra: [] });
          selected = rows.length - 1;
          renderRows(); renderList();
          const last = $("#pe-rows", d).lastElementChild;
          last?.scrollIntoView({ block: "nearest" });
          last?.querySelector(".pe-val")?.focus();
        });
      });
    }

    function importArgs(argv, replace) {
      for (const r of parseArgs(argv, meta)) {
        if (r.spelling === "--model" || r.spelling === "--model-path") { setModel(r.value); continue; }
        if (r.spelling === "--port") { $("#pe-port", d).value = r.value; continue; }
        const idx = rows.findIndex((x) => (x.meta && x.meta === r.meta) || x.spelling === r.spelling);
        if (idx >= 0 && replace) rows[idx] = r; else rows.push(r);
      }
      renderRows(); renderList();
    }

    function setModel(v) {
      if (!v) return;
      if ([...modelSel.options].some((o) => o.value === v)) modelSel.value = v;
      else {
        // a launch script writes $HOME/models/x or ~/models/x: pick the listed model with that folder name
        const hit = meta?.models.find((m) => m.value.startsWith("/") && modelName(m.value) === modelName(v.replace(/^(\$HOME|\$\{HOME\})/, "~")));
        if (hit) modelSel.value = hit.value;
        else { modelSel.value = "__other"; modelPath.value = homePath(v.replace(/^(\$HOME|\$\{HOME\})/, "~")); }
      }
      renderModelInfo();
    }

    // set a flag to a value (adding the row if it is not there), then select it
    function setFlag(flag, value) {
      const f = meta?.byName.get(flag);
      let row = rows.find((r) => (f && r.meta === f) || r.spelling === flag);
      if (value === null) {  // the recommendation drops this flag
        if (row) rows.splice(rows.indexOf(row), 1);
        selected = Math.min(selected, rows.length - 1);
        renderRows(); renderList();
        return;
      }
      if (!row) { row = { spelling: f?.flag || flag, meta: f || null, value: "", extra: [] }; rows.push(row); }
      row.value = value ?? "";
      selected = rows.indexOf(row);
      renderRows(); renderList();
    }

    function advice(title, bodyHtml, redraw = null) {
      const box = $("#pe-advice", d);
      box.hidden = false;
      box.innerHTML = `<div style="display:flex;align-items:center;gap:8px"><b class="small">${esc(title)}</b><span class="grow"></span>
        <button type="button" id="pe-advice-close">${esc(t("close"))}</button></div><div style="margin-top:6px">${bodyHtml}</div>`;
      $("#pe-advice-close", d).onclick = () => { box.hidden = true; };
      box.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => {
        for (const one of JSON.parse(b.dataset.set)) setFlag(one.flag, one.value);
        if (redraw) return redraw();
        b.disabled = true;
        b.textContent = t("pe_applied");
      }));
    }

    // what this host would run: reads GPUs, RAM, cores and the checkpoint's own config
    $("#pe-recommend", d).onclick = async () => {
      const model = modelValue();
      if (!model) return (($("#pe-err", d).textContent = t("pe_err_model")));
      advice(t("pe_recommend"), `<div class="small muted">${esc(t("pe_working"))}</div>`);
      let rec;
      try { rec = await FT.raw(`/recommend?model=${encodeURIComponent(model)}`); }
      catch (e) { return advice(t("pe_recommend"), `<div class="small" style="color:var(--bad)">${esc(e.body?.detail || e.message)}</div>`); }
      renderRecommend(rec);
    };

    // each recommended flag is new, a different value, or already there
    function recState(n) {
      const f = meta?.byName.get(n.flag);
      const row = rows.find((r) => (f && r.meta === f) || r.spelling === n.flag);
      if (n.removed) return { state: row ? "remove" : "set" };
      if (!row) return { state: "add" };
      const want = String(n.value ?? "").trim(), have = String(row.value ?? "").trim();
      if (f?.kind === "bool" || !want || homePath(want) === homePath(have)) return { state: "set" };
      return { state: "change", have };
    }

    function renderRecommend(rec) {
      const h = rec.host || {}, m = h.model || {};
      const host = [
        (h.gpus || []).map((g) => `${g.name} ${fmt.gib(g.total_bytes)}`).join(" + "),
        h.memory?.total ? t("pe_host_ram", { total: fmt.gib(h.memory.total) }) : null,
        h.cores ? t("pe_host_cores", { n: h.cores }) : null,
        m.num_experts ? t("pe_host_moe", { layers: m.num_layers, experts: m.num_experts }) : null,
      ].filter(Boolean).join(" · ");
      const notes = rec.notes.map((n) => ({ ...n, ...recState(n) }));
      const todo = notes.filter((n) => n.state !== "set"), done = notes.filter((n) => n.state === "set");
      const pair = (n) => [{ flag: n.flag, value: n.removed ? null : n.value || "" }];
      const flagText = (n) => (n.removed ? `<s>${esc(n.flag)}</s>` : `${esc(n.flag)}${n.value ? " " + esc(homePath(n.value)) : ""}`);
      const item = (n) => {
        const badge = n.state === "set" ? (n.removed ? "pe_rec_absent" : "pe_rec_set") : n.state === "change" ? "pe_rec_change" : n.state === "remove" ? "pe_rec_remove" : "pe_rec_add";
        const pill = `<span class="pill ${n.state === "set" ? "" : "warn"}" style="font-size:11px;padding:0 8px">${esc(t(badge))}</span>`;
        const change = n.state === "change"
          ? `<div class="small" style="margin-top:2px">${esc(t("pe_rec_now"))} <code>${esc(homePath(n.have))}</code> → <code>${esc(homePath(n.value))}</code></div>` : "";
        const action = n.state === "set" ? ""
          : `<button type="button" style="flex-shrink:0" data-set='${esc(JSON.stringify(pair(n)))}'>${esc(t(n.state === "change" ? "pe_rec_do_change" : n.state === "remove" ? "pe_rec_do_remove" : "pe_apply"))}</button>`;
        return `<div class="kv" style="align-items:flex-start;gap:12px;padding:6px 0;${n.state === "set" ? "opacity:.6" : ""}">
            <span style="flex:1;min-width:0"><span style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><code>${flagText(n)}</code>${pill}</span>
              ${change}<div class="small muted" style="margin-top:2px">${esc(FTI18N.lang === "en" && n.why_en ? n.why_en : n.why)}</div></span>${action}
          </div>`;
      };
      const summary = todo.length
        ? t("pe_rec_count", { todo: todo.length, done: done.length })
        : t("pe_rec_all_set", { n: done.length });
      const applyAll = todo.length
        ? `<button type="button" style="flex-shrink:0" class="primary" data-set='${esc(JSON.stringify(todo.flatMap(pair)))}'>${esc(t("pe_apply_all"))}</button>` : "";
      advice(t("pe_recommend"), `<div class="small muted">${esc(host)}</div>
        <div style="margin-top:10px;display:flex;gap:12px;align-items:center">${applyAll}
          <span class="small"><b>${esc(summary)}</b></span></div>
        <div class="small ${rec.benchmark ? "" : "muted"}" style="margin-top:6px">${esc(rec.benchmark
          ? t("pe_recommend_bench", { when: rec.benchmark.finished ? new Date(rec.benchmark.finished * 1000).toLocaleString(FTI18N.lang === "ja" ? "ja-JP" : "en-US") : "—" })
          : t("pe_recommend_note"))}</div>
        ${rec.benchmark && rec.benchmark.version !== rec.benchmark.current
          ? `<div class="small warn-text" style="margin-top:4px">${esc(t("pe_recommend_bench_old", { then: rec.benchmark.version || "—", now: rec.benchmark.current }))}</div>` : ""}
        <div style="margin-top:8px;max-height:240px;overflow:auto">
          ${todo.map(item).join("")}
          ${done.length && todo.length ? `<div class="pe-cat" style="position:static;padding:6px 0 2px">${esc(t("pe_rec_set_head"))}</div>` : ""}
          ${done.map(item).join("")}
        </div>`, () => renderRecommend(rec));
    }

    // measurements from the running server, turned into flag changes
    $("#pe-tune", d).onclick = async () => {
      advice(t("pe_tune"), `<div class="small muted">${esc(t("pe_working"))}</div>`);
      let data;
      try {
        data = await FTSuggest.load();
        data.modelMax = meta?.models.find((x) => x.value === modelValue())?.max_context || null;
      }
      catch (e) { return advice(t("pe_tune"), `<div class="small" style="color:var(--bad)">${esc(e.body?.detail || e.message)}</div>`); }
      const { state, facts, suggestions } = FTSuggest.compute(data);
      const v = FTSuggest.verdict(state, facts);
      const level = [t("tun_impact_0"), t("tun_impact_1"), t("tun_impact_2"), t("tun_impact_3")];
      const stats = [
        facts?.hit != null ? `${t("tun_title_hit")} ${fmt.pct(facts.hit)}` : null,
        facts?.faultsPerMin != null ? `${t("tun_title_faults")} ${fmt.num(facts.faultsPerMin)}${t("tun_per_min_unit")}` : null,
        facts?.ram != null ? `${t("tun_title_ram")} ${fmt.gib(facts.ram)}` : null,
      ].filter(Boolean).join(" · ");
      advice(t("pe_tune"), `<div class="small"><b>${esc(v.title)}</b>${v.body ? " " + esc(v.body) : ""}</div>
        ${stats ? `<div class="small muted" style="margin-top:4px">${esc(stats)}</div>` : ""}
        <div style="margin-top:8px">${suggestions.length ? suggestions.map((s) => `<div class="kv" style="align-items:flex-start">
            <span style="flex:1;min-width:0"><b class="small">${esc(s.title)}</b>
              <div class="small muted">${esc(s.detail)}</div>
              ${s.flag ? `<div class="small muted">${esc(t("tun_sug_flag"))} <code>${esc(s.flag)}</code></div>` : ""}</span>
            <span class="pill ${s.impact >= 3 ? "ok" : s.impact === 2 ? "info" : "mute"}">${esc(t("tun_impact", { level: level[s.impact] }))}</span>
            ${s.set ? `<button type="button" data-set='${esc(JSON.stringify(s.set))}'>${esc(t("pe_apply"))}</button>` : ""}
          </div>`).join("") : `<div class="empty small">${esc(t("tun_sug_none"))}</div>`}</div>`);
    };

    $("#pe-add", d).onclick = () => { const p = $("#pe-picker", d); p.hidden = !p.hidden; if (!p.hidden) $("#pe-q", d).focus(); };
    $("#pe-paste", d).onclick = () => { $("#pe-paste-box", d).hidden = !$("#pe-paste-box", d).hidden; };
    $("#pe-paste-do", d).onclick = () => {
      let argv = splitShell($("#pe-paste-text", d).value);
      const s = argv.findIndex((tok, i) => tok === "serve" && /(^|\/)ft$/.test(argv[i - 1] || ""));
      if (s >= 0) argv = argv.slice(s + 1);
      const cut = argv.findIndex((tok) => tok.startsWith("|") || tok === "&&" || tok === ";");
      if (cut >= 0) argv = argv.slice(0, cut);
      argv = argv.filter((tok) => !/^(\d?>&?\d?|2>&1|exec|\\)$/.test(tok)).map((tok) => tok.replace(/^(\$HOME|\$\{HOME\})(?=\/)/, "~"));
      importArgs(argv, true);
      $("#pe-paste-text", d).value = ""; $("#pe-paste-box", d).hidden = true;
    };
    $("#pe-q", d).addEventListener("input", renderList);
    $("#pe-cat", d).addEventListener("change", renderList);
    $("#pe-port", d).addEventListener("input", preview);
    modelSel.addEventListener("change", renderModelInfo);
    modelPath.addEventListener("input", preview);

    $("#pe-save", d).onclick = async () => {
      const err = $("#pe-err", d), name = $("#pe-name", d).value.trim(), model = modelValue(), portS = $("#pe-port", d).value.trim();
      err.textContent = "";
      if (!name) return (err.textContent = t("pe_err_name"));
      if (!model) return (err.textContent = t("pe_err_model"));
      if (!/^[\/~]/.test(model) && !model.includes("/")) return (err.textContent = t("pe_err_model_form"));
      if (portS && !/^\d+$/.test(portS)) return (err.textContent = t("pe_err_port"));
      const unknown = rows.filter((r) => !r.meta).map((r) => r.spelling);
      if (unknown.length) return (err.textContent = t("pe_err_unknown", { flags: unknown.join(" ") }));
      const empty = rows.filter((r) => r.meta && r.meta.kind !== "bool" && !r.value.trim()).map((r) => r.spelling);
      if (empty.length) return (err.textContent = t("pe_err_empty", { flags: empty.join(" ") }));
      const reserved = rows.filter((r) => RESERVED.has(r.spelling)).map((r) => r.spelling);
      if (reserved.length) return (err.textContent = t("pe_err_reserved", { flags: reserved.join(" ") }));
      try {
        if (profile.name && profile.name !== name && !profile._new) await FT.raw(`/profiles/${encodeURIComponent(profile.name)}`, { method: "DELETE" });
        await FT.raw(`/profiles/${encodeURIComponent(name)}`, { method: "PUT", json: { model, port: portS ? +portS : null, args: toArgs(rows) } });
        d.close(); onSaved?.();
      } catch (e) { err.textContent = e.body?.detail || e.message; }
    };

    $("#pe-rows", d).innerHTML = `<div class="empty small">${esc(t("pe_flags_loading"))}</div>`;
    loadMeta().then((m) => {
      meta = m;
      modelSel.innerHTML = `<option value="">${esc(t("pe_model_choose"))}</option>` +
        m.models.map((x) => `<option value="${esc(x.value)}">${esc(x.name)}（${fmt.gib(x.size_bytes)}${x.quant ? " · " + esc(x.quant) : ""}）</option>`).join("") +
        `<option value="__other">${esc(t("pe_model_other"))}</option>`;
      if (!m.models.length) $("#pe-model-info", d).textContent = t("pe_model_none");
      $("#pe-cat", d).innerHTML = m.cats.map(([k, key]) => `<option value="${k}">${esc(t(key))}</option>`).join("");
      FT.serveGet("/health").then((h) => { $("#pe-tune", d).hidden = !(h.status === "ok"); }, () => {});
      setModel(profile.model || "");
      rows = [];
      importArgs(profile.args || [], false);
      renderModelInfo();
    }, (e) => {
      $("#pe-rows", d).innerHTML = `<div class="small" style="color:var(--bad)">${esc(t("pe_flags_failed", { detail: e.body?.detail || e.message }))}</div>`;
    });
  }

  return { open, parseArgs, toArgs, splitShell };
})();
