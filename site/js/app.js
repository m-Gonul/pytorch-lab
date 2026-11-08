// Her sayfada çalışan giriş noktası.
import { progress, kv } from "./store.js";
import { onChange, checkLocal, captureRunnerLink, runnerConf, setRunnerConf, warmBrowser } from "./engines.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// --- tema
$(".theme-toggle")?.addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("ptl.theme", next); } catch {}
});

// --- mobil müfredat menüsü
const tog = $(".nav-toggle");
tog?.addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  tog.setAttribute("aria-expanded", String(open));
});

// --- sözlük terimleri: dokunmatikte dokununca aç/kapa
document.addEventListener("click", (e) => {
  const t = e.target.closest("dfn.term");
  $$("dfn.term.open").forEach((x) => x !== t && x.classList.remove("open"));
  if (t) t.classList.toggle("open");
});

// --- yan menüde modül ilerlemesi
function paintNav() {
  for (const a of $$("a.nav-mod[data-mod]")) {
    const n = progress.count(a.dataset.mod);
    const el = $(".nav-prog", a);
    if (el) el.innerHTML = Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("");
    a.classList.toggle("complete", n === 5);
  }
  for (const s of $$("[data-prog]")) { const n = progress.count(s.dataset.prog); s.textContent = n ? `${n}/5` : "hazır"; s.classList.toggle("complete", n === 5); }
  const sum = $("#prog-summary");
  if (sum) {
    const mods = window.PTL.modules;
    const done = mods.filter((m) => progress.count(m) === 5).length;
    const stages = mods.reduce((a, m) => a + progress.count(m), 0);
    sum.textContent = `${done}/${mods.length} hazır modül tamam · ${stages}/${mods.length * 5} aşama`;
  }
}
paintNav();
addEventListener("ptl:progress", paintNav);

// --- ilerleme dışa/içe aktarma (müfredat sayfası)
document.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "prog-export") {
    const blob = new Blob([JSON.stringify({ app: "pytorch-lab", v: 1, progress: progress.all() }, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "pytorch-lab-ilerleme.json" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  if (act === "prog-clear" && confirm("Tüm ilerleme işaretleri silinsin mi? (Yazdığın kodlar kalır.)")) progress.clear();
});
document.addEventListener("change", async (e) => {
  if (e.target.dataset?.act !== "prog-import") return;
  try {
    const j = JSON.parse(await e.target.files[0].text());
    if (j.app !== "pytorch-lab" || typeof j.progress !== "object") throw new Error();
    progress.replace(j.progress);
  } catch { alert("Bu dosya bir pytorch-lab ilerleme dosyası değil."); }
  e.target.value = "";
});

// --- başlıktaki motor göstergesi
const pill = $("#engine-pill");
if (pill) {
  pill.addEventListener("click", () => (location.href = "kurulum.html#motorlar"));
  onChange((s) => {
    const b = s.browser.status === "ready" ? "on" : s.browser.status === "loading" ? "busy" : s.browser.status === "error" ? "err" : "off";
    const l = s.local.status === "ok" ? "on" : s.local.status === "checking" ? "busy" : s.local.status === "auth" ? "err" : "off";
    pill.innerHTML = `<span class="dot ${b}"></span>Python <span class="dot ${l}"></span>PyTorch`;
    pill.title = `Tarayıcı: ${s.browser.text}\nYerel PyTorch: ${s.local.text}`;
  });
}

// --- kurulum sayfası: yerel motor bağlantı paneli
function runnerPanel() {
  const root = $("#runner-connect");
  if (!root) return;
  const captured = captureRunnerLink();
  const c = runnerConf();
  root.innerHTML = `
    <div class="rc-row"><label>Port <input id="rc-port" type="number" min="1" max="65535" value="${c.port}"></label>
    <label class="grow">Token <input id="rc-token" type="password" autocomplete="off" spellcheck="false" placeholder="motor başlarken terminalde yazar" value="${c.token || ""}"></label>
    <button class="btn primary" id="rc-save" type="button">Bağlan</button></div>
    <p class="rc-status" id="rc-status" role="status"></p>
    <div class="rc-row"><button class="btn" id="rc-warm" type="button">Tarayıcı Python'unu şimdi indir</button><span class="muted" id="rc-browser"></span></div>`;
  const st = $("#rc-status"), br = $("#rc-browser");
  onChange((s) => {
    const cls = s.local.status === "ok" ? "good" : s.local.status === "auth" ? "bad" : "";
    st.className = "rc-status " + cls;
    st.textContent = (s.local.status === "ok" ? "Bağlı: " : "Yerel motor: ") + s.local.text;
    br.textContent = s.browser.text;
  });
  $("#rc-save").addEventListener("click", () => { setRunnerConf({ port: Number($("#rc-port").value) || window.PTL.port, token: $("#rc-token").value.trim() }); checkLocal(); });
  $("#rc-warm").addEventListener("click", warmBrowser);
  if (captured) setTimeout(() => $("#motorlar")?.scrollIntoView(), 50);
}
runnerPanel();

// --- arama (başlıkta; "/" ile odaklan)
const fold = (t) => t.toLocaleLowerCase("tr").normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/ı/g, "i");
const qIn = $("#q"), qRes = $("#q-res");
let IDX = null;
async function loadIdx() { if (!IDX) { try { IDX = (await (await fetch("search.json")).json()).map((e) => ({ ...e, f: fold(e.t + " " + e.s + " " + e.k) })); } catch { IDX = []; } } return IDX; }
function closeRes() { qRes.hidden = true; qIn.setAttribute("aria-expanded", "false"); }
if (qIn) {
  let sel = -1;
  const show = async () => {
    const words = fold(qIn.value).split(/\s+/).filter(Boolean);
    if (!words.length) return closeRes();
    const idx = await loadIdx();
    const hits = idx.filter((e) => words.every((w) => e.f.includes(w)))
      .map((e) => ({ e, sc: (fold(e.t).startsWith(words[0]) ? 3 : 0) + (e.k === "modül" ? 2 : 0) + (fold(e.t).includes(words.join(" ")) ? 1 : 0) }))
      .sort((a, b) => b.sc - a.sc).slice(0, 12);
    sel = -1;
    qRes.innerHTML = hits.length ? hits.map(({ e }, i) => `<a role="option" id="qr-${i}" href="${e.u}"><span class="k">${e.k}</span> <b>${e.t.replace(/</g, "&lt;")}</b>${e.s ? `<span class="s">${e.s.replace(/</g, "&lt;").slice(0, 120)}</span>` : ""}</a>`).join("") : `<p class="muted">Sonuç yok. Daha kısa ya da İngilizce terimle dene (ör. “dropout”, “attention”).</p>`;
    qRes.hidden = false; qIn.setAttribute("aria-expanded", "true");
  };
  qIn.addEventListener("input", show);
  qIn.addEventListener("focus", () => { loadIdx(); if (qIn.value) show(); });
  qIn.addEventListener("keydown", (e) => {
    const opts = $$("a", qRes);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + opts.length) % Math.max(opts.length, 1); opts.forEach((o, i) => o.classList.toggle("sel", i === sel)); opts[sel]?.scrollIntoView({ block: "nearest" }); qIn.setAttribute("aria-activedescendant", opts[sel]?.id || ""); }
    if (e.key === "Enter" && opts.length) { e.preventDefault(); location.href = (opts[sel] || opts[0]).getAttribute("href"); }
    if (e.key === "Escape") { closeRes(); qIn.blur(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) closeRes(); });
  document.addEventListener("keydown", (e) => { if (e.key === "/" && !/input|textarea|select/i.test(document.activeElement?.tagName) && !document.activeElement?.isContentEditable) { e.preventDefault(); qIn.focus(); } });
}

// --- ana sayfa: kaldığın yerden devam
const resume = kv.get("resume");
const home = $(".page-home .hero .cta");
if (home && resume?.mod) home.insertAdjacentHTML("beforebegin", `<p class="resume"><a class="btn big" href="${resume.mod}.html#${resume.sec || ""}">Kaldığın yerden devam: ${resume.code} · ${(resume.name || "").replace(/</g, "&lt;")}</a></p>`);

// --- önkoşul haritası
for (const n of $$(".map a.n")) {
  const id = n.dataset.mod, k = progress.count(id);
  $$(".pips rect", n).forEach((r, i) => r.classList.toggle("on", i < k));
  n.classList.toggle("complete", k === 5);
  if (!window.PTL.modules.includes(id)) n.classList.add("soon");
  const hl = (on) => {
    for (const e of $$(`.map path.e[data-to="${id}"], .map path.e[data-from="${id}"]`)) e.classList.toggle(e.dataset.to === id ? "in" : "out", on);
    for (const e of $$(`.map path.e[data-to="${id}"]`)) $(`.map a.n[data-mod="${e.dataset.from}"]`)?.classList.toggle("pre", on);
    for (const e of $$(`.map path.e[data-from="${id}"]`)) $(`.map a.n[data-mod="${e.dataset.to}"]`)?.classList.toggle("post", on);
  };
  n.addEventListener("mouseenter", () => hl(true)); n.addEventListener("mouseleave", () => hl(false));
  n.addEventListener("focus", () => hl(true)); n.addEventListener("blur", () => hl(false));
}
for (const li of $$(".seq li[data-mod]")) { const k = progress.count(li.dataset.mod); if (k) li.insertAdjacentHTML("beforeend", ` <span class="status${k === 5 ? " complete" : ""}">${k}/5</span>`); }

// --- notlar sayfası
const noteList = $("#note-list");
if (noteList) {
  const titles = JSON.parse($("#mod-titles").textContent);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const paint = () => {
    const r = kv.get("resume");
    if (r?.mod) $("#resume-box").innerHTML = `<a href="${r.mod}.html#${r.sec || ""}">${esc(titles[r.mod] || r.mod)} — ${esc(r.name || "")}</a> <span class="muted small">(${new Date(r.t).toLocaleString("tr")})</span>`;
    const bms = kv.get("bookmarks", []) || [];
    if (bms.length) $("#bm-list").innerHTML = bms.map((b, i) => `<li><a href="${b.mod}.html">${esc(titles[b.mod] || b.mod)}</a> <button class="btn ghost small" type="button" data-rm="${i}">Kaldır</button></li>`).join("");
    const notes = kv.keys("note.").map((k) => [k.slice(5), kv.get(k)]).filter(([, v]) => v);
    if (notes.length) noteList.innerHTML = notes.map(([m, v]) => `<article class="note-item"><h3><a href="${m}.html">${esc(titles[m] || m)}</a></h3><pre class="note-text">${esc(v)}</pre></article>`).join("");
  };
  paint();
  $("#bm-list").addEventListener("click", (e) => { const i = e.target.dataset.rm; if (i == null) return; const b = kv.get("bookmarks", []); b.splice(Number(i), 1); kv.set("bookmarks", b); $("#bm-list").innerHTML = '<li class="muted">Yer imi yok.</li>'; paint(); });
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-act=notes-export]")) {
      const dump = { app: "pytorch-lab-notes", v: 1, bookmarks: kv.get("bookmarks", []), resume: kv.get("resume"), notes: Object.fromEntries(kv.keys("note.").map((k) => [k, kv.get(k)])), self: Object.fromEntries(kv.keys("self.").map((k) => [k, kv.get(k)])), practice: Object.fromEntries(kv.keys("pr.").map((k) => [k, kv.get(k)])) };
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([JSON.stringify(dump, null, 2)], { type: "application/json" })), download: "pytorch-lab-notlar.json" });
      a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }
  });
  document.addEventListener("change", async (e) => {
    if (e.target.dataset?.act !== "notes-import") return;
    try {
      const j = JSON.parse(await e.target.files[0].text());
      if (j.app !== "pytorch-lab-notes") throw new Error();
      kv.set("bookmarks", j.bookmarks || []); if (j.resume) kv.set("resume", j.resume);
      for (const o of [j.notes, j.self, j.practice]) for (const [k, v] of Object.entries(o || {})) kv.set(k, v);
      paint();
    } catch { alert("Bu dosya bir pytorch-lab not dosyası değil."); }
    e.target.value = "";
  });
}

// --- karışık tekrar
if ($("#rv-card")) import("./review.js").then((m) => m.initReview());

// --- modül sayfası
if ($("#page-data")) import("./module.js").then((m) => m.initModule());
else checkLocal();
