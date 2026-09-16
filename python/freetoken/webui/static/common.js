// Shared by both pages: which process served us, API paths for serve vs mgr, token, formatting, language.
"use strict";

const FT = (() => {
  const TOKEN_KEY = "ft-webui-token";
  const demo = new URLSearchParams(location.search).has("demo");
  const t = FTI18N.t;
  let mode = "serve";
  // ft mgr: anyone may look; this PC or a token holder may operate (webui/auth.py)
  let auth = { local: false, write: false, token_given: false, token_valid: false, token: null };

  // serve path -> manager path. ft mgr proxies the serve and camelCases keys; snake() undoes that.
  const DAEMON_PATHS = {
    "/health": "/engine/health",
    "/v1/stats": "/engine/view/stats",
    "/v1/requests": "/engine/requests",
    "/v1/cache/status": "/engine/cache",
    "/v1/kai/experts": "/engine/kai/experts",
  };

  function token() { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } }
  function setToken(v) { try { v ? localStorage.setItem(TOKEN_KEY, v) : localStorage.removeItem(TOKEN_KEY); } catch {} }

  function snake(o) {
    if (Array.isArray(o)) return o.map(snake);
    if (o && typeof o === "object") {
      const r = {};
      for (const [k, v] of Object.entries(o)) r[k.replace(/([A-Z])/g, (m) => "_" + m.toLowerCase())] = snake(v);
      return r;
    }
    return o;
  }

  class HttpError extends Error { constructor(status, body) { super(`HTTP ${status}`); this.status = status; this.body = body; } }

  async function raw(path, opts = {}) {
    const headers = { Accept: "application/json", ...(opts.headers || {}) };
    const tok = token();
    if (tok) headers["X-FT-Token"] = tok;
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
    const res = await fetch(path, { method: opts.method || "GET", headers, body: opts.body, cache: "no-store" });
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = text; }
    if (!res.ok) { if (res.status === 401) FT.onUnauthorized(); throw new HttpError(res.status, body); }
    return body;
  }

  // GET a serve-side document, whichever process served the page
  async function serveGet(path, query = "") {
    if (mode === "mgr" && path === "/health") {
      // the manager runs nothing: show a serve started outside it on the default port, if any
      const h = snake(await raw("/engine/health"));
      if (h.running) return h;
      const ext = snake(await raw("/engine/external"));
      return ext.external ? { ...ext.health, running: false, external: true, port: ext.port } : h;
    }
    const p = mode === "mgr" ? DAEMON_PATHS[path] : path;
    if (!p) throw new Error(`no route for ${path}`);
    const doc = await raw(p + query);
    return mode === "mgr" ? snake(doc) : doc;
  }

  async function init() {
    FTI18N.applyStatic();
    if (demo) { mode = "demo"; return mode; }
    try { mode = (await raw("/ui/env.json")).mode || "serve"; } catch { mode = "serve"; }
    if (mode === "mgr") {
      try { auth = await raw("/auth"); }
      // a manager from before the check: it takes any write, so do not hide what it would accept
      catch (e) { if (e.status === 404) auth = { ...auth, write: true }; }
    }
    return mode;
  }

  // SSE over fetch so the token header can ride along (EventSource cannot set headers)
  async function sse(path, onEvent, signal) {
    const headers = { Accept: "text/event-stream" };
    const tok = token();
    if (tok) headers["X-FT-Token"] = tok;
    const res = await fetch(path, { headers, signal });
    if (!res.ok) { if (res.status === 401) FT.onUnauthorized(); throw new HttpError(res.status, null); }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = { event: "message", data: "" };
        for (const line of block.split("\n")) {
          if (line.startsWith("data:")) ev.data += line.slice(5).trimStart();
          else if (line.startsWith("event:")) ev.event = line.slice(6).trim();
          else if (line.startsWith("id:")) ev.id = line.slice(3).trim();
        }
        onEvent(ev);
      }
    }
  }

  const locale = () => (FTI18N.lang === "ja" ? "ja-JP" : "en-US");
  const fmt = {
    gib: (b) => (b == null ? "—" : (b / 2 ** 30).toFixed(1) + " GiB"),
    num: (n) => (n == null ? "—" : Math.round(n).toLocaleString(locale())),
    tok: (n) => (n == null ? "—" : n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(Math.round(n))),
    dur: (s) => {
      if (s == null) return "—";
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
      if (FTI18N.lang === "ja") return h ? `${h} 時間 ${m} 分` : m ? `${m} 分` : `${Math.floor(s)} 秒`;
      return h ? `${h} h ${m} min` : m ? `${m} min` : `${Math.floor(s)} s`;
    },
    ms: (v) => {
      if (v == null) return "—";
      if (v < 1000) return Math.round(v) + " ms";
      return (v / 1000).toFixed(1) + (FTI18N.lang === "ja" ? " 秒" : " s");
    },
    pct: (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d) + "%"),
    time: (iso) => { if (!iso) return "—"; const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit", second: "2-digit" }); },
  };

  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  const $ = (sel, root = document) => root.querySelector(sel);

  // display only: a model is shown by its folder name, other text gets the home directory folded to ~
  const homePath = (s) => String(s ?? "").replace(/(^|[\s='"(])\/(?:home\/[^/\s'"]+|root)(?=\/|\s|$)/g, "$1~");
  const modelName = (m) => { const s = String(m ?? ""); return s.startsWith("/") || s.startsWith("~") ? s.replace(/\/+$/, "").split("/").pop() : s; };

  function header(active) {
    const el = document.createElement("header");
    el.className = "top";
    el.innerHTML = `<div class="wrap">
      <span class="brand">FreeToken-Kai</span>
      <nav class="tabs" id="hdr-nav" hidden>
        <a href="./${demo ? "?demo" : ""}" class="${active === "dash" ? "on" : ""}">${esc(t("nav_dash"))}</a>
        <a href="bench.html${demo ? "?demo" : ""}" class="${active === "bench" ? "on" : ""}">${esc(t("nav_bench"))}</a>
      </nav>
      <span class="grow"></span>
      <span id="hdr-state" class="pill mute" hidden><span class="dot"></span>${esc(t("state_connecting"))}</span>
      <span id="hdr-mode" class="pill mute"></span>
      <select id="hdr-lang" aria-label="language" style="width:auto">
        <option value="en" ${FTI18N.lang === "en" ? "selected" : ""}>English</option>
        <option value="ja" ${FTI18N.lang === "ja" ? "selected" : ""}>日本語</option>
      </select>
      <span id="hdr-auth" class="pill warn" hidden></span>
      <button id="hdr-token" hidden>${esc(t("token"))}</button>
    </div>`;
    document.body.prepend(el);
    const m = $("#hdr-mode");
    $("#hdr-nav").hidden = !(mode === "mgr" || mode === "demo");
    if (mode === "mgr") {
      m.hidden = true; $("#hdr-token").hidden = false;
      if (!auth.write) {
        const a = $("#hdr-auth");
        a.hidden = false;
        a.className = "pill " + (auth.token_given ? "bad" : "warn");
        a.textContent = t(auth.token_given ? "auth_wrong" : "auth_view_only");
        $("#hdr-token").textContent = t("auth_enter");
        $("#hdr-token").className = "primary";
      }
    }
    else if (mode === "demo") { m.className = "pill warn"; m.textContent = t("mode_demo"); }
    else m.hidden = true;  // served by ft serve: read-only is the only thing it could be
    $("#hdr-token").onclick = () => (auth.local ? showToken() : askToken());
    $("#hdr-lang").onchange = (ev) => FTI18N.set(ev.target.value);
  }

  function setState(kind, text) {
    const s = $("#hdr-state");
    if (!s) return;
    // the engine card already shows the state: the header speaks up only when something is wrong
    s.hidden = kind !== "bad";
    s.className = "pill " + kind;
    s.innerHTML = `<span class="dot"></span>${esc(text)}`;
  }

  // confirm() and alert() in the page's own look; resolve to true only on the OK button
  function ask(message, { title = "", ok = t("ok"), danger = false, cancel = true } = {}) {
    return new Promise((resolve) => {
      const d = document.createElement("dialog");
      d.className = "ask";
      d.innerHTML = `<form method="dialog" class="stack">
        ${title ? `<b>${esc(title)}</b>` : ""}
        <div class="ask-msg">${esc(message)}</div>
        <div style="display:flex;gap:8px;justify-content:flex-end">
          ${cancel ? `<button value="cancel">${esc(t("cancel"))}</button>` : ""}
          <button value="ok" class="${danger ? "danger-fill" : "primary"}" autofocus>${esc(ok)}</button></div>
      </form>`;
      document.body.append(d);
      d.addEventListener("close", () => { resolve(d.returnValue === "ok"); d.remove(); });
      d.showModal();
    });
  }
  const notice = (message, title = "") => ask(message, { title, cancel: false });

  // on this PC: show the token so it can be typed into a browser on another PC
  function showToken() {
    const d = document.createElement("dialog");
    d.innerHTML = `<form method="dialog" class="stack">
      <b>${esc(t("token_local_title"))}</b>
      <span class="small muted">${esc(t("token_local_hint"))}</span>
      <div style="display:flex;gap:8px"><input type="text" id="tok" readonly value="${esc(auth.token || "")}" style="flex:1;font-family:var(--mono)">
        <button type="button" id="tok-copy">${esc(t("copy"))}</button></div>
      <div style="display:flex;justify-content:flex-end"><button value="close">${esc(t("close"))}</button></div>
    </form>`;
    document.body.append(d);
    $("#tok-copy", d).onclick = async (ev) => {
      const input = $("#tok", d);
      try { await navigator.clipboard.writeText(input.value); } catch { input.select(); document.execCommand("copy"); }
      ev.target.textContent = t("copied");
    };
    d.addEventListener("close", () => d.remove());
    d.showModal();
  }

  let asking = false;
  function askToken() {
    if (asking) return;
    asking = true;
    const d = document.createElement("dialog");
    d.innerHTML = `<form method="dialog" class="stack">
      <b>${esc(t("token_title"))}</b>
      <span class="small muted">${esc(t("token_hint"))}</span>
      <input type="password" id="tok" value="${esc(token())}" autocomplete="off">
      <div style="display:flex;gap:8px;align-items:center">
        <button value="clear" class="danger">${esc(t("clear"))}</button><span class="grow"></span>
        <button value="cancel">${esc(t("close"))}</button><button value="ok" class="primary">${esc(t("save"))}</button></div>
    </form>`;
    document.body.append(d);
    // Enter in the field would press the form's first button, which is "clear": make it save
    $("#tok", d).addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); d.close("ok"); } });
    d.addEventListener("close", () => {
      if (d.returnValue === "ok") setToken($("#tok", d).value.trim());
      if (d.returnValue === "clear") setToken("");
      d.remove(); asking = false;
      if (d.returnValue === "ok" || d.returnValue === "clear") location.reload();
    });
    d.showModal();
  }

  return {
    init, raw, serveGet, ask, notice, sse, fmt, esc, $, header, setState, askToken, HttpError, homePath, modelName, t,
    get mode() { return mode; }, demo,
    get canWrite() { return mode !== "mgr" || auth.write; },
    onUnauthorized: () => { setState("bad", t("token_needed")); },
  };
})();
