// Modül sayfası: el hesabı (a), kod kutuları (b–e), sınama, aşama rayı.
import { progress, saved } from "./store.js";
import { enhance } from "./editor.js";
import { runBrowser, runLocal, stopBrowser, warmBrowser, state, onChange, checkLocal } from "./engines.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v; else if (k === "html") el.innerHTML = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v); else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
};
const cfg = window.PTL || {};

export function initModule() {
  const data = JSON.parse($("#page-data").textContent);
  const mod = data.id;
  initHand(mod);
  for (const box of $$(".codebox")) initCodebox(mod, box, data.stages[box.dataset.stage]);
  initQuiz(mod);
  initRail(mod);
  // tarayıcı aşaması varsa Python'u arka planda ısıt; yerel motoru yokla
  if ($$('.codebox[data-engine="browser"]').length) setTimeout(warmBrowser, 600);
  checkLocal();
}

// ------------------------------------------------------------------ (a) el hesabı
function parseNum(s) {
  s = s.trim().replace(/\s+/g, "").replace(/[−–]/g, "-");
  if (!s) return NaN;
  if (/^-?\d+(?:[.,]\d+)?\/-?\d+(?:[.,]\d+)?$/.test(s)) { const [a, b] = s.replace(/,/g, ".").split("/"); return Number(a) / Number(b); }
  if (/^-?\d{1,3}(\.\d{3})+(,\d+)?$/.test(s)) s = s.replace(/\./g, ""); // 262.144 → binlik ayırıcı
  return Number(s.replace(",", "."));
}

function initHand(mod) {
  const root = $(".hand");
  if (!root) return;
  const key = `${mod}.a`;
  const fields = $$(".hf", root);
  const status = $(".hand-status", root);
  try { const v = JSON.parse(saved.get(key) || "{}"); for (const f of fields) if (v[f.dataset.id] != null) $("input", f).value = v[f.dataset.id]; } catch {}
  const save = () => saved.set(key, JSON.stringify(Object.fromEntries(fields.map((f) => [f.dataset.id, $("input", f).value]))));
  const judge = (f) => {
    const x = parseNum($("input", f).value), want = Number(f.dataset.answer), tol = Number(f.dataset.tol) || 0;
    if (Number.isNaN(x)) return null;
    return Math.abs(x - want) <= Math.max(tol, 1e-9) + 1e-12 * Math.abs(want);
  };
  const mark = (f, ok) => { f.classList.toggle("ok", ok === true); f.classList.toggle("bad", ok === false); $(".hf-mark", f).textContent = ok === true ? "✓" : ok === false ? "✗" : ""; };
  const check = () => {
    let right = 0, empty = 0;
    for (const f of fields) { const ok = judge(f); mark(f, ok); if (ok) right++; if (ok === null) empty++; }
    const all = right === fields.length;
    status.textContent = all ? "Hepsi doğru. Sıradaki aşama: döngüyle yaz." : `${right}/${fields.length} doğru${empty ? `, ${empty} boş` : ""}.`;
    status.className = "hand-status " + (all ? "good" : "");
    if (all) progress.set(mod, "a");
  };
  for (const f of fields) {
    const inp = $("input", f);
    inp.addEventListener("input", () => { mark(f, null); save(); });
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); const ok = judge(f); mark(f, ok); if (ok) { const i = fields.indexOf(f); fields[i + 1] ? $("input", fields[i + 1]).focus() : check(); } } });
  }
  root.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "hand-check") check();
    if (act === "hand-hint") for (const f of fields) $(".hf-hint", f).hidden = judge(f) === true;
    if (act === "hand-reset") { for (const f of fields) { $("input", f).value = ""; mark(f, null); $(".hf-hint", f).hidden = true; } status.textContent = ""; save(); }
  });
  if (fields.some((f) => $("input", f).value)) for (const f of fields) mark(f, judge(f));
}

// ------------------------------------------------------------------ (b–e) kod kutuları
const ENGINE_NAME = { browser: "Tarayıcı · Pyodide", local: "Yerel PyTorch" };

function initCodebox(mod, box, spec) {
  const s = box.dataset.stage, key = `${mod}.${s}`;
  const ta = $(".cb-code", box), out = $(".cb-out", box), engineEl = $('[data-role="engine"]', box);
  const ed = enhance(ta, { onCheck: () => go(true), onRun: () => go(false), onChange: (v) => saved.set(key, v) });
  const prev = saved.get(key);
  if (prev != null && prev !== spec.start) ed.set(prev);
  if (progress.get(mod)[s]) box.classList.add("passed");

  // motor seçimi: tarayıcı aşamaları ikisinde de çalışır; torch aşamaları yalnız yerelde
  const prefKey = `ptl.engine.${mod}.${s}`;
  let choice = spec.engine === "torch" ? "local" : (localStorage.getItem(prefKey) || "browser");
  const sel = h("select", { "aria-label": "Çalıştırma motoru", class: "cb-select" },
    h("option", { value: "browser", disabled: spec.engine === "torch" }, ENGINE_NAME.browser + (spec.engine === "torch" ? " (torch yok)" : "")),
    h("option", { value: "local" }, ENGINE_NAME.local));
  sel.value = choice;
  sel.addEventListener("change", () => { choice = sel.value; try { localStorage.setItem(prefKey, choice); } catch {} paintEngine(); });
  const dot = h("span", { class: "dot" });
  engineEl.append(dot, sel);
  const paintEngine = () => {
    const st = choice === "browser" ? state.browser.status : state.local.status;
    dot.className = "dot " + ({ ready: "on", ok: "on", loading: "busy", checking: "busy", error: "err", auth: "err" }[st] || "off");
    dot.title = choice === "browser" ? state.browser.text : state.local.text;
  };
  onChange(paintEngine);

  let busy = false;
  async function go(withCheck) {
    if (busy) return;
    busy = true;
    box.classList.add("busy");
    const stopBtn = choice === "browser" ? h("button", { class: "btn ghost small", type: "button", onclick: () => stopBrowser() }, "Durdur") : null;
    const st = h("span", {}, choice === "browser" && state.browser.status !== "ready" ? "Python yükleniyor…" : "Çalışıyor…");
    out.replaceChildren(h("div", { class: "cb-running" }, h("span", { class: "spinner", "aria-hidden": "true" }), st, stopBtn));
    const code = ta.value, check = withCheck ? spec.check : null;
    const t0 = performance.now();
    const res = choice === "browser"
      ? await runBrowser(code, check, { onStatus: (t) => (st.textContent = t + "…") })
      : await runLocal(code, check, { tag: `${mod}/${s}` });
    render(res, withCheck, (performance.now() - t0) / 1000);
    busy = false;
    box.classList.remove("busy");
  }

  function render(res, withCheck, wall) {
    const parts = [];
    if (res.offline) {
      const colab = `https://colab.research.google.com/github/${cfg.repo}/blob/${cfg.branch}/notebooks/${mod}.ipynb`;
      parts.push(h("div", { class: "cb-offline", html: `<strong>Yerel PyTorch motoru bağlı değil.</strong> Bir terminalde çalıştır:<pre class="code"><code>uv run ${cfg.site}runner/torchlab.py</code></pre>Açılan bağlantı bu sayfayı motora bağlar. <a href="kurulum.html">Kurulum ayrıntıları</a> · ya da <a href="${colab}" target="_blank" rel="noopener">bu modülü Colab'da aç</a>.` }));
      out.replaceChildren(...parts);
      return;
    }
    if (res.error) parts.push(h("pre", { class: "cb-error" }, res.error));
    if (res.stdout && res.stdout.trim()) parts.push(h("pre", { class: "cb-stdout" }, res.stdout.replace(/\n$/, "")));
    if (withCheck && res.items?.length) {
      parts.push(h("ul", { class: "cb-items" }, res.items.map((it) => h("li", { class: it.ok ? "ok" : "bad" },
        h("span", { class: "mk", "aria-label": it.ok ? "geçti" : "kaldı" }, it.ok ? "✓" : "✗"), h("span", { class: "nm" }, it.name), it.msg ? h("span", { class: "msg" }, it.msg) : null))));
    }
    for (const t of res.tables || []) {
      parts.push(h("figure", { class: "bench" }, h("figcaption", {}, t.title),
        h("table", {}, h("thead", {}, h("tr", {}, t.cols.map((c) => h("th", {}, c)))), h("tbody", {}, t.rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c)))))),
        t.note ? h("p", { class: "note" }, t.note) : null));
    }
    for (const n of res.notes || []) parts.push(h("p", { class: "note" }, n));
    const where = choice === "browser" ? "tarayıcı" : "yerel";
    const verdict = withCheck ? (res.ok ? "Geçti" : "Henüz değil") : res.error ? "Hata" : "Çalıştı";
    parts.unshift(h("div", { class: "cb-verdict " + (withCheck ? (res.ok ? "good" : "bad") : res.error ? "bad" : "") }, h("strong", {}, verdict), h("span", { class: "muted" }, ` · ${where} · ${wall.toFixed(2)} s`)));
    out.replaceChildren(...parts);
    if (withCheck && res.ok) { progress.set(mod, s); box.classList.add("passed"); }
  }

  box.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "check") go(true);
    if (act === "run") go(false);
    if (act === "reset" && confirm("Kodun başlangıç hâline dönsün mü? Yazdıkların silinir.")) { ed.set(spec.start); saved.del(key); out.replaceChildren(); }
    if (act === "solution" && confirm("Model çözüm kodunun yerine yazılsın mı? Önce kendin denemen önerilir; mevcut kodun silinir.")) { ed.set(spec.sol); out.replaceChildren(); }
  });
}

// ------------------------------------------------------------------ sınama
function initQuiz(mod) {
  for (const q of $$(".quiz .q")) {
    q.addEventListener("click", (e) => {
      const b = e.target.closest(".q-opt");
      if (!b) return;
      const ok = b.dataset.ok === "1";
      b.classList.add(ok ? "ok" : "bad");
      b.setAttribute("aria-pressed", "true");
      b.nextElementSibling.hidden = false;
      if (ok) q.classList.add("solved");
    });
  }
}

// ------------------------------------------------------------------ aşama rayı
function initRail(mod) {
  const links = $$("[data-rail]");
  const paint = () => { const p = progress.get(mod); for (const a of links) a.classList.toggle("done", !!p[a.dataset.rail]); };
  paint();
  addEventListener("ptl:progress", paint);
  const secs = $$(".stage");
  const io = new IntersectionObserver((ents) => {
    for (const en of ents) if (en.isIntersecting) for (const a of links) a.classList.toggle("here", a.dataset.rail === en.target.id);
  }, { rootMargin: "-30% 0px -60% 0px" });
  secs.forEach((s) => io.observe(s));
}
