// Statik siteyi dist/ içine üretir: içerik (YAML + Markdown) → HTML, site/ varlıkları,
// KaTeX CSS+font'ları, labcheck.py ve yerel runner.
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";
import { makeMarkdown, esc } from "./md.mjs";

const root = join(import.meta.dirname, "..");
const C = (...p) => join(root, "content", ...p);
const out = join(root, "dist");
const read = (p) => readFileSync(p, "utf8");
const yaml = (p) => YAML.parse(read(p));

const site = yaml(C("site.yaml"));
const glossary = Object.fromEntries(Object.entries(yaml(C("glossary.yaml")).terms).map(([k, v]) => [k.toLowerCase(), v]));
const md = makeMarkdown(glossary);
const curriculum = yaml(C("curriculum.yaml"));
const sources = yaml(C("sources.yaml"));
const srcByKey = Object.fromEntries(sources.groups.flatMap((g) => g.items).map((s) => [s.key, s]));

export const STAGES = [
  { k: "a", name: "Elle", long: "Kâğıt-kalem, 2×2 boyutunda" },
  { k: "b", name: "Döngü", long: "Saf Python: for / if" },
  { k: "c", name: "NumPy", long: "NumPy / pandas, döngüsüz" },
  { k: "d", name: "PyTorch", long: "Tensor + autograd" },
  { k: "e", name: "Vektörize", long: "Batch, einsum, vmap + hız ölçümü" },
];

// --------------------------------------------------------------------- modüller
const modDir = C("modules");
const modules = {};
for (const id of readdirSync(modDir)) {
  const d = join(modDir, id);
  if (!existsSync(join(d, "module.yaml"))) continue;
  const m = yaml(join(d, "module.yaml"));
  const f = (n) => (existsSync(join(d, n)) ? read(join(d, n)) : null);
  m.md = { intro: f("intro.md"), outro: f("outro.md") };
  for (const s of "abcde") m.md[s] = f(`${s}.md`);
  m.files = {};
  for (const s of "bcde") m.files[s] = { start: f(`${s}.py`), sol: f(`${s}.sol.py`), check: f(`${s}.check.py`) };
  m.extras = m.extras || [];
  for (const x of m.extras) {
    if (!/^x\d+$/.test(x.id)) throw new Error(`${id}: ek alıştırma kimliği x<sayı> olmalı: ${x.id}`);
    m.md[x.id] = f(`${x.id}.md`);
    m.files[x.id] = { start: f(`${x.id}.py`), sol: f(`${x.id}.sol.py`), check: f(`${x.id}.check.py`) };
    for (const [k, v] of Object.entries(m.files[x.id])) if (!v) throw new Error(`${id}/${x.id}: ${k} dosyası yok`);
    m.stages[x.id] = { engine: x.engine || "browser" };
  }
  m.homework = m.homework || [];
  modules[id] = m;
}

const order = curriculum.phases.flatMap((p) => p.modules.map((x) => ({ ...x, phase: p.n })));
for (const id of Object.keys(modules)) if (!order.find((o) => o.id === id)) throw new Error(`müfredatta olmayan modül: ${id}`);
const ready = order.filter((o) => modules[o.id]);

// Ham HTML içindeki [[terim]] / [[terim|metin]] işaretlerini sözlük tooltip'ine çevirir.
const termify = (html) => html.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, k, t) => {
  const def = glossary[k.trim().toLowerCase()];
  if (!def) throw new Error(`sözlükte olmayan terim: [[${k}]]`);
  return `<dfn class="term" tabindex="0" data-tip="${esc(def)}">${esc((t || k).trim())}</dfn>`;
});

// --------------------------------------------------------------------- şablon
const nav = (active) => {
  const phases = curriculum.phases.map((p) => {
    const items = p.modules.map((x) => {
      const ok = !!modules[x.id];
      const cls = ["nav-mod", ok ? "ready" : "soon", x.id === active ? "active" : ""].join(" ").trim();
      const inner = `<span class="nav-code">${esc(x.code)}</span><span class="nav-title">${esc(x.title)}</span>`;
      return ok
        ? `<li><a class="${cls}" href="${x.id}.html" data-mod="${x.id}"${x.id === active ? ' aria-current="page"' : ""}>${inner}<span class="nav-prog" aria-hidden="true"></span></a></li>`
        : `<li><span class="${cls}" title="Yakında">${inner}</span></li>`;
    }).join("");
    const open = p.modules.some((x) => x.id === active) || (!active && p.n <= 1);
    return `<details class="nav-phase"${open ? " open" : ""}><summary><span class="ph-n">Faz ${p.n}</span> ${esc(p.title)}</summary><ul>${items}</ul></details>`;
  }).join("");
  return `<nav class="side" aria-label="Müfredat">${phases}</nav>`;
};

const page = ({ title, body, active = null, extraHead = "", bodyClass = "", data = null }) => `<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)} · ${esc(site.title)}</title>
<meta name="description" content="${esc(site.tagline)}: klasik ML'den ileri PyTorch'a, tarayıcıda çalışan alıştırmalarla.">
<script>try{const t=localStorage.getItem("ptl.theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400..800;1,7..72,400..600&family=JetBrains+Mono:wght@400;600&family=Caveat:wght@500;700&display=swap">
<link rel="stylesheet" href="vendor/katex/katex.min.css">
<link rel="stylesheet" href="css/style.css">
<link rel="icon" href="img/icon.svg" type="image/svg+xml">
${extraHead}</head>
<body class="${bodyClass}">
<a class="skip" href="#main">İçeriğe geç</a>
<header class="top">
  <a class="brand" href="index.html" aria-label="${esc(site.title)} ana sayfa"><img src="img/icon.svg" alt="" width="26" height="26"><span>${esc(site.title)}</span></a>
  <button class="nav-toggle" aria-expanded="false" aria-controls="side">Müfredat</button>
  <nav class="top-links" aria-label="Site">
    <a href="mufredat.html">Müfredat</a><a href="odevler.html">Ödevler</a><a href="kurulum.html">Motorlar</a><a href="kaynaklar.html">Kaynaklar</a><a href="sozluk.html">Sözlük</a>
  </nav>
  <button class="engine-pill" id="engine-pill" type="button" title="Kod çalıştırma motorlarının durumu">Motorlar</button>
  <button class="theme-toggle" type="button" aria-label="Tema değiştir"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9z" fill="currentColor"/></svg></button>
</header>
<div class="shell">
<div id="side">${nav(active)}</div>
<main id="main">
${body}
</main>
</div>
<footer class="foot">
  <p>Kod MIT, metin CC BY-SA 4.0 · <a href="https://github.com/${esc(site.repo)}" rel="noopener">GitHub</a> · Uyarlanan kaynaklar her modülün sonunda lisanslarıyla listelenir.</p>
</footer>
${data ? `<script type="application/json" id="page-data">${JSON.stringify(data).replace(/</g, "\\u003c")}</script>` : ""}
<script>window.PTL=${JSON.stringify({ pyodide: site.pyodide, port: site.runner_port, repo: site.repo, branch: site.branch, site: site.site_url, modules: ready.map((r) => r.id) })}</script>
<script type="module" src="js/app.js"></script>
</body>
</html>
`;

// --------------------------------------------------------------------- modül sayfası
function sourceList(keysOrItems) {
  const rows = keysOrItems.map((s) => {
    const it = typeof s === "string" ? srcByKey[s] : s;
    if (!it) throw new Error(`bilinmeyen kaynak anahtarı: ${s}`);
    const badge = it.use === "adapt" ? `<span class="lic adapt" title="Lisans uyarlamaya izin veriyor">uyarlanabilir</span>` : `<span class="lic link" title="Yalnızca bağlantı">yalnız link</span>`;
    return `<li><a href="${esc(it.url)}" rel="noopener" target="_blank">${md.inline(it.title)}</a> <span class="lic-name">${esc(it.license)}</span> ${badge}</li>`;
  });
  return `<ul class="sources">${rows.join("")}</ul>`;
}

function handHTML(m) {
  const groups = (m.hand?.groups || []).map((g, gi) => {
    const field = (f) => `<label class="hf" data-id="${esc(f.id)}" data-answer="${f.answer}" data-tol="${f.tol ?? 0}">
      <span class="hf-label">${md.inline(f.label)}</span>
      <input type="text" inputmode="decimal" autocomplete="off" spellcheck="false" aria-label="${esc(f.label.replace(/\$|\\/g, ""))}">
      <span class="hf-mark" aria-hidden="true"></span>
      <span class="hf-hint" hidden>${md.inline(f.hint || "")}</span></label>`;
    const fields = g.layout === "matrix"
      ? `<div class="hmatrix" style="--cols:${g.cols}">${g.fields.map(field).join("")}</div>`
      : `<div class="hlist">${g.fields.map(field).join("")}</div>`;
    return `<fieldset class="hgroup"><legend>${md.inline(g.title)}</legend>${fields}</fieldset>`;
  }).join("");
  return `<div class="hand" data-stage="a">${groups}
  <div class="hand-actions"><button class="btn primary" data-act="hand-check" type="button">Kontrol et</button><button class="btn" data-act="hand-hint" type="button">Yanlışlar için ipucu</button><button class="btn ghost" data-act="hand-reset" type="button">Temizle</button><span class="hand-status" role="status"></span></div></div>`;
}

function codeboxHTML(m, s) {
  const st = m.stages[s];
  const lines = m.files[s].start.split("\n").length;
  const label = /^x/.test(s) ? `Ek alıştırma ${s.slice(1)}` : `Aşama ${s}`;
  return `<div class="codebox" data-stage="${s}" data-engine="${st.engine}">
  <div class="cb-bar">
    <span class="cb-engine" data-role="engine"></span>
    <span class="cb-spacer"></span>
    <button class="btn primary" data-act="check" type="button" title="Çalıştır ve kontrol et (Ctrl+Enter)">Kontrol et</button>
    <button class="btn" data-act="run" type="button" title="Yalnızca çalıştır (Shift+Enter)">Çalıştır</button>
    <button class="btn ghost" data-act="reset" type="button">Sıfırla</button>
    <button class="btn ghost" data-act="solution" type="button">Çözüm</button>
  </div>
  <textarea class="cb-code" spellcheck="false" autocapitalize="off" autocomplete="off" rows="${Math.min(Math.max(lines + 1, 8), 34)}" aria-label="${label} kodu">${esc(m.files[s].start)}</textarea>
  <div class="cb-out" aria-live="polite"></div>
</div>`;
}

function extrasHTML(m) {
  if (!m.extras.length) return "";
  const items = m.extras.map((x, i) => `<section class="stage extra" id="${x.id}" data-stage="${x.id}">
<h3><span class="stage-n x">E${i + 1}</span> ${md.inline(x.title)}<span class="stage-engine" data-engine-badge="${m.stages[x.id].engine}"></span></h3>
<div class="prose">${md.block(m.md[x.id] || "")}</div>
${codeboxHTML(m, x.id)}
</section>`).join("\n");
  return `<section class="extras" id="ek"><h2>Ek alıştırmalar</h2><p class="muted">Beş aşamanın dışındaki pratikler: konunun ders notlarında geçen ama ana akışa sığmayan her parçası burada, kendi kontrolüyle. Sıra önemli değil.</p>${items}</section>`;
}

function homeworkItemHTML(h, withModule = null) {
  const steps = (h.steps || []).map((x) => `<li>${md.inline(x)}</li>`).join("");
  const acc = (h.accept || []).map((x) => `<li>${md.inline(x)}</li>`).join("");
  return `<article class="hw">
<h3>${md.inline(h.title)}${withModule ? ` <small><a href="${withModule.id}.html">${esc(withModule.code)}</a></small>` : ""}</h3>
<p class="hw-paper"><a href="${esc(h.url)}" rel="noopener" target="_blank">${md.inline(h.paper)}</a>${h.level ? ` · <span class="hw-level">${esc(h.level)}</span>` : ""}${h.hours ? ` · ≈${h.hours} saat` : ""}</p>
<div class="prose">${md.block(h.goal || "")}</div>
${steps ? `<h4>Adımlar</h4><ol class="hw-steps">${steps}</ol>` : ""}
${acc ? `<h4>Kabul ölçütleri</h4><ul class="hw-accept">${acc}</ul>` : ""}
</article>`;
}

function homeworkHTML(m) {
  if (!m.homework.length) return "";
  return `<section class="homework" id="odev"><h2>Makale ödevleri</h2><p class="muted">İsteğe bağlı, kapsamlı ödevler: bir makaleyi baştan uygulamak. Otomatik kontrol yok; kabul ölçütleri sayısal ve kendin doğrulayabileceğin şekilde yazıldı.</p>${m.homework.map((h) => homeworkItemHTML(h)).join("")}</section>`;
}

function quizHTML(m) {
  if (!m.quiz?.length) return "";
  const qs = m.quiz.map((q, qi) => `<li class="q" data-q="${qi}"><div class="q-text">${md.inline(q.q)}</div><ul class="q-opts">${q.options.map((o, oi) => `<li><button type="button" class="q-opt" data-ok="${o.ok ? 1 : 0}">${md.inline(o.t)}</button><div class="q-why" hidden>${md.inline(o.why || "")}</div></li>`).join("")}</ul></li>`).join("");
  return `<section class="quiz-sec" id="sinama"><h2>Sınama</h2><p class="muted">Her seçeneğin neden doğru ya da yanlış olduğu açıklamasıyla açılır.</p><ol class="quiz">${qs}</ol></section>`;
}

function modulePage(id) {
  const m = modules[id];
  const o = order.find((x) => x.id === id);
  const idx = ready.findIndex((x) => x.id === id);
  const prev = ready[idx - 1], next = ready[idx + 1];
  const prereq = (m.prereq || []).map((p) => {
    const x = order.find((y) => y.id === p);
    if (!x) throw new Error(`${id}: bilinmeyen önkoşul ${p}`);
    return modules[p] ? `<a href="${p}.html">${esc(x.code)} ${esc(x.title)}</a>` : `<span>${esc(x.code)} ${esc(x.title)}</span>`;
  });
  const railExtra = (m.extras.length ? `<li><a href="#ek" data-rail="ek"><span class="rail-n">+${m.extras.length}</span><span class="rail-name">Ek</span></a></li>` : "")
    + (m.homework.length ? `<li><a href="#odev" data-rail="odev"><span class="rail-n">✎</span><span class="rail-name">Ödev</span></a></li>` : "");
  const rail = `<ol class="rail" aria-label="Aşamalar">${STAGES.map((s, i) => `<li><a href="#${s.k}" data-rail="${s.k}"><span class="rail-n">${i + 1}</span><span class="rail-name">${s.name}</span></a></li>`).join("")}${railExtra}</ol>`;
  const stageSec = STAGES.map((s, i) => {
    const eng = s.k === "a" ? "" : `<span class="stage-engine" data-engine-badge="${m.stages[s.k].engine}"></span>`;
    return `<section class="stage" id="${s.k}" data-stage="${s.k}">
<h2><span class="stage-n">${i + 1}</span> ${s.name} <span class="stage-long">${s.long}</span>${eng}</h2>
<div class="prose">${md.block(m.md[s.k])}</div>
${s.k === "a" ? handHTML(m) : codeboxHTML(m, s.k)}
</section>`;
  }).join("\n");
  const colab = `https://colab.research.google.com/github/${site.repo}/blob/${site.branch}/notebooks/${id}.ipynb`;
  const body = `<article class="module" data-module="${id}">
<header class="mod-head">
  <p class="mod-code">${esc(m.code)} · Faz ${o.phase}</p>
  <h1>${esc(m.title)}</h1>
  <p class="lede">${md.inline(m.summary.trim())}</p>
  <dl class="mod-meta">
    <div><dt>Süre</dt><dd>≈ ${Math.round(m.minutes / 30) / 2} saat</dd></div>
    <div><dt>Önkoşul</dt><dd>${prereq.join(", ") || "yok"}</dd></div>
    <div><dt>Notebook</dt><dd><a href="${colab}" rel="noopener" target="_blank">Colab'da aç</a> · <a href="https://github.com/${site.repo}/blob/${site.branch}/notebooks/${id}.ipynb" rel="noopener" target="_blank">.ipynb</a></dd></div>
  </dl>
  <details class="goals" open><summary>Bu modülün sonunda</summary><ul>${m.goals.map((g) => `<li>${md.inline(g)}</li>`).join("")}</ul></details>
</header>
<div class="rail-wrap">${rail}</div>
<section class="intro prose">${md.block(m.md.intro)}</section>
${stageSec}
${extrasHTML(m)}
${m.md.outro ? `<section class="outro prose">${md.block(m.md.outro)}</section>` : ""}
${quizHTML(m)}
${homeworkHTML(m)}
<section class="mod-sources"><h2>Kaynaklar</h2>${sourceList(m.sources)}</section>
<nav class="pager">${prev ? `<a class="prev" href="${prev.id}.html"><small>Önceki</small>${esc(prev.code)} ${esc(prev.title)}</a>` : "<span></span>"}${next ? `<a class="next" href="${next.id}.html"><small>Sonraki</small>${esc(next.code)} ${esc(next.title)}</a>` : "<span></span>"}</nav>
</article>`;
  const keys = ["b", "c", "d", "e", ...m.extras.map((x) => x.id)];
  const data = { id, stages: Object.fromEntries(keys.map((s) => [s, { engine: m.stages[s].engine, start: m.files[s].start, sol: m.files[s].sol, check: m.files[s].check }])) };
  return page({ title: `${m.code} ${m.title}`, body, active: id, data, bodyClass: "page-module" });
}

// --------------------------------------------------------------------- diğer sayfalar
function indexPage() {
  const phases = curriculum.phases.map((p) => {
    const n = p.modules.length, r = p.modules.filter((x) => modules[x.id]).length;
    return `<li class="ph-card${p.gpu ? " gpu" : ""}"><a href="mufredat.html#faz-${p.n}"><span class="ph-num">${p.n}</span><span class="ph-body"><strong>${esc(p.title)}</strong><span>${esc(p.blurb)}</span><span class="ph-meta">${n} modül · ≈${p.weeks} hafta${r ? ` · ${r} hazır` : ""}${p.gpu ? " · GPU" : ""}</span></span></a></li>`;
  }).join("");
  const first = ready[0];
  const body = `<section class="hero">
  <h1>Aynı işlemi beş kez yaz.<br>Beşincisinde ne yaptığını bileceksin.</h1>
  <p class="lede">Klasik ML'den ileri PyTorch'a ${order.length} modül. Her konu önce 2×2'lik bir örnekle kâğıtta, sonra döngüyle, NumPy'da, PyTorch'ta ve en sonda vektörize hâliyle çözülür; her aşama bir öncekinin sonucuyla birebir eşleşmek zorunda.</p>
  <div class="ladder" aria-label="Aynı hesabın beş hâli">
    <div class="rung r1"><span class="rn">1</span><span class="rl">Elle</span><span class="rv hand-font">1·5 + 2·7 = <b>19</b></span></div>
    <div class="rung r2"><span class="rn">2</span><span class="rl">Döngü</span><code class="rv">for t in range(k): s += A[i][t] * B[t][j]</code></div>
    <div class="rung r3"><span class="rn">3</span><span class="rl">NumPy</span><code class="rv">C = A @ B</code></div>
    <div class="rung r4"><span class="rn">4</span><span class="rl">PyTorch</span><code class="rv">(A @ B).sum().backward()</code></div>
    <div class="rung r5"><span class="rn">5</span><span class="rl">Vektörize</span><code class="rv">torch.einsum("bik,bkj->bij", As, Bs)</code></div>
  </div>
  <p class="cta"><a class="btn primary big" href="${first.id}.html">${esc(first.code)} ile başla</a> <a class="btn big" href="kurulum.html">Kod nerede çalışıyor?</a></p>
</section>
<section class="how">
  <h2>Kod nerede çalışır</h2>
  <div class="how-grid">
    <div><h3>Tarayıcıda</h3><p>Elle, döngü ve NumPy aşamaları [[pyodide|Pyodide]] ile sayfanın içinde çalışır. İlk açılışta Python tarayıcına bir kez iner (~6 MB sıkıştırılmış), sonra önbellekte kalır; kodun cihazından çıkmaz.</p></div>
    <div><h3>Yerel PyTorch</h3><p>PyTorch aşamaları tek komutla açılan bir yerel motora gider: <code>uv run …/torchlab.py</code>. Gerçek PyTorch, gerçek hız ölçümü; varsa GPU'n da.</p></div>
    <div><h3>Colab</h3><p>Her modülün notebook'u aynı kontrolleri içerir. Kurulum yapmak istemezsen ya da GPU gerekiyorsa oradan aç.</p></div>
  </div>
</section>
<section class="phases"><h2>Yol</h2><ol class="ph-list">${phases}</ol></section>`;
  return page({ title: "Ana sayfa", body: termify(body), bodyClass: "page-home" });
}

function curriculumPage() {
  const total = order.length, weeks = curriculum.phases.reduce((a, p) => a + p.weeks, 0);
  const sec = curriculum.phases.map((p) => `<section class="cur-phase" id="faz-${p.n}"><h2><span class="ph-num">${p.n}</span> ${esc(p.title)} <small>≈${p.weeks} hafta${p.gpu ? " · GPU gerekir" : ""}</small></h2><p class="muted">${esc(p.blurb)}</p>
<table class="cur"><thead><tr><th>Modül</th><th>Beş aşamanın çekirdeği</th><th>Kaynaklar</th><th>Durum</th></tr></thead><tbody>${p.modules.map((x) => `<tr data-mod="${x.id}"><td><strong>${esc(x.code)}</strong> ${modules[x.id] ? `<a href="${x.id}.html">${esc(x.title)}</a>` : esc(x.title)}</td><td>${esc(x.core)}</td><td>${(x.src || []).map((k) => { const s = srcByKey[k]; if (!s) throw new Error(`kaynak yok: ${k}`); return `<a href="${esc(s.url)}" rel="noopener" target="_blank" title="${esc(s.title)}">${esc(s.title.split(" — ")[0].replace(/^Stanford |^Karpathy$/, ""))}</a>`; }).join(", ")}</td><td>${modules[x.id] ? `<span class="status ready" data-prog="${x.id}">hazır</span>` : `<span class="status soon">yakında</span>`}</td></tr>`).join("")}</tbody></table></section>`).join("");
  const body = `<header class="page-head"><h1>Müfredat</h1><p class="lede">${curriculum.phases.length} faz, ${total} modül, ≈${weeks} hafta (haftada 8–10 saat varsayımıyla tahmin). Faz 0 → 1 → 3 omurgadır; Faz 2 ve 4 paralel gidebilir.</p>
<div class="progress-tools"><span id="prog-summary" class="muted"></span><button class="btn" type="button" data-act="prog-export">İlerlemeyi dışa aktar</button><label class="btn">İçe aktar<input type="file" accept="application/json" data-act="prog-import" hidden></label><button class="btn ghost" type="button" data-act="prog-clear">Sıfırla</button></div></header>${sec}`;
  return page({ title: "Müfredat", body });
}

function sourcesPage() {
  const sec = sources.groups.map((g) => `<section><h2>${esc(g.title)}</h2><table class="src-table"><thead><tr><th>Kaynak</th><th>Lisans</th><th>Kullanım</th><th>Modüller</th></tr></thead><tbody>${g.items.map((s) => {
    const mods = order.filter((o) => (o.src || []).includes(s.key)).map((o) => modules[o.id] ? `<a href="${o.id}.html">${o.code}</a>` : o.code);
    return `<tr><td><a href="${esc(s.url)}" rel="noopener" target="_blank">${esc(s.title)}</a></td><td>${esc(s.license)}${s.verified === false ? ' <span class="warn" title="Lisans bu projede doğrulanmadı">?</span>' : ""}</td><td>${s.use === "adapt" ? '<span class="lic adapt">uyarlanabilir</span>' : '<span class="lic link">yalnız link</span>'}</td><td>${mods.join(", ") || "—"}</td></tr>`;
  }).join("")}</tbody></table></section>`).join("");
  const body = `<header class="page-head"><h1>Kaynaklar ve lisanslar</h1><p class="lede">“Uyarlanabilir” kaynaklardan (MIT, BSD, Apache, CC BY, CC BY-SA) alıştırma ve anlatım uyarlanır, atıf her modülün sonunda verilir. ND (türetme yasak) ya da lisansı doğrulanmamış kaynaklara yalnızca bağlantı verilir; ND lisansı çeviriyi de yasaklar.</p></header>${sec}`;
  return page({ title: "Kaynaklar", body });
}

function homeworkPage() {
  let n = 0;
  const sec = curriculum.phases.map((p) => {
    const items = p.modules.filter((x) => modules[x.id]?.homework.length).flatMap((x) => modules[x.id].homework.map((h) => { n++; return homeworkItemHTML(h, x); }));
    return items.length ? `<section class="hw-phase"><h2><span class="ph-num">${p.n}</span> ${esc(p.title)}</h2>${items.join("")}</section>` : "";
  }).join("");
  const body = `<header class="page-head"><h1>Makale ödevleri</h1><p class="lede">${n} ödev. Her biri bir makaleyi (ya da klasik bir algoritmayı) sıfırdan uygulatır; ilgili modülü bitirdikten sonra yapılması için yazıldı. Kabul ölçütleri sayısal: sonucu kendin ölçüp karşılaştırabilirsin.</p></header>${sec}`;
  return page({ title: "Makale ödevleri", body: termify(body) });
}

function glossaryPage() {
  const rows = Object.entries(yaml(C("glossary.yaml")).terms).sort((a, b) => a[0].localeCompare(b[0], "tr")).map(([k, v]) => `<div class="gl-row" id="t-${esc(k.replace(/\s+/g, "-"))}"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  return page({ title: "Sözlük", body: `<header class="page-head"><h1>Sözlük</h1><p class="lede">Metinde altı noktalı çizili terimlerin üzerine gelince (ya da dokununca) bu tanımlar açılır. Teknik terimler İngilizce bırakıldı; ilk kullanımda burada açıklanır.</p></header><dl class="glossary">${rows}</dl>` });
}

function markdownPage(file, title) {
  return page({ title, body: `<article class="prose page-prose">${md.block(read(C("pages", file)))}</article>` });
}

// --------------------------------------------------------------------- yaz
rmSync(out, { recursive: true, force: true });
cpSync(join(root, "site"), out, { recursive: true });
mkdirSync(join(out, "vendor/katex"), { recursive: true });
cpSync(join(root, "node_modules/katex/dist/katex.min.css"), join(out, "vendor/katex/katex.min.css"));
cpSync(join(root, "node_modules/katex/dist/fonts"), join(out, "vendor/katex/fonts"), { recursive: true });
mkdirSync(join(out, "runner"), { recursive: true });
cpSync(join(root, "runner/torchlab.py"), join(out, "runner/torchlab.py"));

const pages = {
  "index.html": indexPage(),
  "mufredat.html": curriculumPage(),
  "kaynaklar.html": sourcesPage(),
  "sozluk.html": glossaryPage(),
  "odevler.html": homeworkPage(),
  "kurulum.html": markdownPage("kurulum.md", "Motorlar ve kurulum"),
};
for (const r of ready) pages[`${r.id}.html`] = modulePage(r.id);
for (const [f, html] of Object.entries(pages)) writeFileSync(join(out, f), html);

let files = 0, bytes = 0;
(function walk(d) { for (const f of readdirSync(d)) { const p = join(d, f), s = statSync(p); if (s.isDirectory()) walk(p); else { files++; bytes += s.size; } } })(out);
console.log(`dist/: ${Object.keys(pages).length} sayfa (${ready.length} modül), ${files} dosya, ${(bytes / 1024 / 1024).toFixed(1)} MB`);
