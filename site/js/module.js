// Modül sayfası: el hesabı (a), kod kutuları (b–e), sınama, aşama rayı.
import { progress, saved, kv } from "./store.js";
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
  initPractice(mod);
  initGoals(mod);
  initTools(mod, data);
  initReadiness(data);
  if (Object.keys(data.labs || {}).length) import("./labs/runtime.js").then((L) => L.mountAll(data.labs)).catch((e) => console.error(e));
  paintReview(mod, data);
  addEventListener("ptl:progress", () => paintReview(mod, data));
  // tarayıcı aşaması varsa Python'u arka planda ısıt; yerel motoru yokla
  if ($$('.codebox[data-engine="browser"]').length) setTimeout(warmBrowser, 600);
  checkLocal();
}

// ------------------------------------------------------------------ (a) el hesabı
export function parseNum(s) {
  s = s.trim().replace(/\s+/g, "").replace(/[−–]/g, "-").replace(/[a-zA-ZçğıöşüÇĞİÖŞÜµ%°²³]+$/u, "");
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
  const mark = (f, ok) => {
    f.classList.toggle("ok", ok === true); f.classList.toggle("bad", ok === false); $(".hf-mark", f).textContent = ok === true ? "✓" : ok === false ? "✗" : "";
    const why = $(".hf-why", f);
    const msg = ok === false ? mistakeMsg(f.dataset.mistakes, parseNum($("input", f).value), Number(f.dataset.answer)) : "";
    why.innerHTML = msg; why.hidden = !msg;
  };
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
    if (act === "hand-hint") {
      let any = false;
      for (const f of fields) {
        if (judge(f) === true) continue;
        const next = $$(".hf-hint[hidden]", f)[0];
        if (next) { next.hidden = false; any = true; }
      }
      status.textContent = any ? "Yanlış ya da boş alanlarda bir sonraki ipucu açıldı." : "Açılacak başka ipucu yok — adım adım çözüme bakabilirsin.";
    }
    if (act === "hand-reset") { for (const f of fields) { $("input", f).value = ""; mark(f, null); $$(".hf-hint", f).forEach((x) => (x.hidden = true)); } status.textContent = ""; save(); }
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
    if (act === "next-hint") nextHint(e.target.closest(".cb-hints"));
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
      const qi = q.dataset.q, hist = kv.get(`quiz.${mod}`, {});
      if (!(qi in hist)) { hist[qi] = ok ? 1 : 0; kv.set(`quiz.${mod}`, hist); dispatchEvent(new CustomEvent("ptl:progress")); }
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

// ------------------------------------------------------------------ ortak yardımcılar
export function mistakeMsg(json, x, want) {
  if (!json || Number.isNaN(x)) return "";
  try {
    for (const m of JSON.parse(json)) if (Math.abs(x - m.v) <= 1e-6 * Math.max(1, Math.abs(m.v))) return m.m;
  } catch {}
  if (want !== 0 && Math.abs(x + want) <= 1e-6 * Math.abs(want)) return "İşaret hatası: büyüklük doğru, yön (işaret) ters.";
  if (want !== 0 && [10, 100, 1000, 0.1, 0.01, 0.001].some((k) => Math.abs(x - want * k) <= 1e-6 * Math.abs(want * k))) return "Büyüklük mertebesi kaymış: birim ya da ondalık basamağını kontrol et.";
  return "";
}

function nextHint(box) {
  if (!box) return;
  const items = $$(".hint-list > li", box), n = items.length, shown = items.filter((x) => !x.hidden).length;
  if (shown < n) items[shown].hidden = false;
  const b = $("[data-act]", box);
  b.textContent = shown + 1 >= n ? `Tüm ipuçları açık (${n}/${n})` : `Sonraki ipucu (${shown + 1}/${n})`;
  b.disabled = shown + 1 >= n;
}

// ------------------------------------------------------------------ pratik kartları
function initPractice(mod) {
  const sec = $("#pratik");
  if (!sec) return;
  const st = kv.get(`pr.${mod}`, {});
  const save = () => { kv.set(`pr.${mod}`, st); dispatchEvent(new CustomEvent("ptl:progress")); };
  for (const b of $$(".pr-filter [data-lvl]", sec)) b.addEventListener("click", () => {
    $$(".pr-filter [data-lvl]", sec).forEach((x) => x.classList.toggle("on", x === b));
    for (const a of $$(".pr", sec)) a.hidden = b.dataset.lvl !== "all" && a.dataset.level !== b.dataset.lvl;
  });
  for (const a of $$(".pr", sec)) {
    const pid = a.dataset.pid, s = (st[pid] = st[pid] || {});
    const ans = $(".pr-answer", a);
    if (ans) {
      const inp = $("input", ans), msg = $(".pr-msg", ans);
      if (s.v) inp.value = s.v;
      const check = () => {
        const x = parseNum(inp.value), want = Number(ans.dataset.answer), tol = Number(ans.dataset.tol) || 0;
        s.v = inp.value;
        if (Number.isNaN(x)) { msg.textContent = "Bir sayı yaz (kesir, virgül ve birim kabul edilir)."; msg.className = "pr-msg"; return; }
        const ok = Math.abs(x - want) <= Math.max(tol, 1e-9) + 1e-12 * Math.abs(want);
        if (!("first" in s)) s.first = ok ? 1 : 0;
        s.ok = ok ? 1 : s.ok || 0;
        const why = ok ? "" : mistakeMsg(ans.dataset.mistakes, x, want);
        msg.innerHTML = ok ? "Doğru. Çözümdeki başka yolu da oku." : `Henüz değil.${why ? " " + why : " Bir ipucu açmayı dene."}`;
        msg.className = "pr-msg " + (ok ? "good" : "bad");
        a.classList.toggle("solved", ok);
        save();
      };
      $("[data-act=pr-check]", ans).addEventListener("click", check);
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); check(); } });
      if (s.ok) a.classList.add("solved");
    }
    for (const b of $$(".pr-choices .q-opt", a)) b.addEventListener("click", () => {
      const ok = b.dataset.ok === "1";
      b.classList.add(ok ? "ok" : "bad"); b.setAttribute("aria-pressed", "true"); b.nextElementSibling.hidden = false;
      if (!("first" in s)) s.first = ok ? 1 : 0;
      if (ok) { s.ok = 1; a.classList.add("solved"); }
      save();
    });
    const ta = $(".pr-open textarea", a);
    if (ta) { if (s.text) ta.value = s.text; ta.addEventListener("input", () => { s.text = ta.value; kv.set(`pr.${mod}`, st); }); }
    const rub = $$(".pr-rubric input", a);
    rub.forEach((c) => { c.checked = !!(s.r || [])[c.dataset.r]; c.addEventListener("change", () => { s.r = rub.map((x) => (x.checked ? 1 : 0)); s.ok = s.r.every(Boolean) ? 1 : 0; a.classList.toggle("solved", !!s.ok); save(); }); });
    if (rub.length && s.ok) a.classList.add("solved");
    const hb = $("[data-act=pr-hint]", a);
    hb?.addEventListener("click", () => { nextHint(hb.closest(".pr-hints")); s.hints = (s.hints || 0) + 1; kv.set(`pr.${mod}`, st); });
    $(".pr-sol", a)?.addEventListener("toggle", (e) => { if (e.target.open && !("first" in s) && a.dataset.level !== "0") { s.peek = 1; kv.set(`pr.${mod}`, st); } });
  }
}

// ------------------------------------------------------------------ öz değerlendirme, notlar, yer imi, kaldığın yer
function initGoals(mod) {
  const boxes = $$("input[data-goal]");
  const st = kv.get(`self.${mod}`, {});
  for (const c of boxes) { c.checked = !!st[c.dataset.goal]; c.addEventListener("change", () => { st[c.dataset.goal] = c.checked ? 1 : 0; kv.set(`self.${mod}`, st); dispatchEvent(new CustomEvent("ptl:progress")); }); }
}

function initTools(mod, data) {
  const ta = $(".notes textarea");
  if (ta) {
    ta.value = kv.get(`note.${mod}`, "") || "";
    if (ta.value) $(".notes").open = true;
    ta.addEventListener("input", () => (ta.value ? kv.set(`note.${mod}`, ta.value) : kv.del(`note.${mod}`)));
  }
  const bm = $("[data-act=bookmark]");
  const paintBm = () => { const on = (kv.get("bookmarks", []) || []).some((b) => b.mod === mod && !b.sec); bm.setAttribute("aria-pressed", String(on)); bm.textContent = on ? "Yer iminde ✓" : "Yer imi ekle"; };
  bm?.addEventListener("click", () => {
    let list = kv.get("bookmarks", []) || [];
    if (list.some((b) => b.mod === mod && !b.sec)) list = list.filter((b) => !(b.mod === mod && !b.sec));
    else list.push({ mod, title: document.querySelector(".module h1")?.textContent, code: data.code, t: Date.now() });
    kv.set("bookmarks", list); paintBm();
  });
  if (bm) paintBm();
  // kaldığın yer: görünür bölüm
  const io = new IntersectionObserver((ents) => {
    for (const en of ents) if (en.isIntersecting) kv.set("resume", { mod, code: data.code, title: document.querySelector(".module h1")?.textContent, sec: en.target.id, name: (en.target.querySelector("h2,h3")?.textContent || "").trim().slice(0, 80), t: Date.now() });
  }, { rootMargin: "-40% 0px -55% 0px" });
  $$(".stage, .lab, .practice, .extras, .quiz-sec, .intro").forEach((s) => s.id && io.observe(s));
}

function initReadiness(data) {
  const box = $("#hazir .rc-body");
  if (!box || !data.ready?.length) return;
  box.innerHTML = `<p class="muted small">Bu modülün önkoşullarından birkaç soru. Yanlış yaptığın varsa önce o modüle dönmek zaman kazandırır.</p><ol class="quiz">${data.ready.map((q, i) => `<li class="q" data-r="${i}"><div class="q-text"><span class="tag">${q.code}</span> ${q.q}</div><ul class="q-opts">${q.o.map((o) => `<li><button type="button" class="q-opt" data-ok="${o.ok ? 1 : 0}">${o.t}</button><div class="q-why" hidden>${o.why}</div></li>`).join("")}</ul><p class="rc-back" hidden>Bunu <a href="${q.mod}.html">${q.code}</a> modülünde tekrar etmen önerilir.</p></li>`).join("")}</ol>`;
  box.addEventListener("click", (e) => {
    const b = e.target.closest(".q-opt"); if (!b) return;
    const ok = b.dataset.ok === "1", li = b.closest(".q");
    b.classList.add(ok ? "ok" : "bad"); b.nextElementSibling.hidden = false;
    if (!ok) li.querySelector(".rc-back").hidden = false;
  });
}

function paintReview(mod, data) {
  const sec = $("#geri-don"), ul = $(".review-list", sec || document);
  if (!sec) return;
  const items = [];
  const qh = kv.get(`quiz.${mod}`, {});
  for (const [qi, v] of Object.entries(qh)) if (!v) { const t = $(`.quiz .q[data-q="${qi}"] .q-text`)?.textContent || ""; items.push(`<li><a href="#sinama">Sınama sorusu ${Number(qi) + 1}</a>: ${t.slice(0, 110)}${t.length > 110 ? "…" : ""} <span class="muted small">— ilgili bölümü tekrar oku, sonra karışık tekrar sayfasında yeniden gel</span></li>`); }
  const self = kv.get(`self.${mod}`, {});
  const touched = Object.keys(qh).length || Object.keys(progress.get(mod)).length;
  $$("input[data-goal]").forEach((c, i) => { if (touched && !self[i]) items.push(`<li>Kazanım işaretlenmedi: ${c.closest("label").querySelector("span").textContent.slice(0, 120)}</li>`); });
  const p = progress.get(mod);
  if (touched) for (const s of "abcde") if (!p[s]) items.push(`<li><a href="#${s}">Aşama ${s}</a> henüz geçilmedi.</li>`);
  const pr = kv.get(`pr.${mod}`, {});
  for (const [pid, s] of Object.entries(pr)) if (s.first === 0 && !s.ok) items.push(`<li><a href="#pr-${pid}">Pratik ${pid}</a>: ilk denemede yanlış, henüz çözülmedi.</li>`);
  ul.innerHTML = items.join("");
  sec.hidden = !items.length;
}
