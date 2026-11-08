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

export const PTYPES = {
  "çözülmüş": "Çözülmüş örnek", "hesap": "Hesaplama", "açıkla": "Kendi cümlenle açıkla", "yorumla": "Grafik / sonuç yorumla",
  "tamamla": "Eksik adımı tamamla", "hata-bul": "Hatayı bul ve düzelt", "karşı-örnek": "Karşı örnek üret", "kanıt": "Kanıt",
  "kod": "Kod yaz / incele / hata ayıkla", "yöntem-seç": "Yöntem seç ve gerekçelendir", "uyarla": "Gerçek probleme uyarla",
  "birleştir": "Konuları birleştiren problem", "proje": "Açık uçlu deney / küçük proje",
};
const LEVEL_NAME = ["Çözülmüş örnek", "Temel", "Uygulama", "Zorlayıcı"];

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
  m.practice = m.practice || [];
  m.labs = m.labs || [];
  m.pitfalls = m.pitfalls || [];
  m.whatif = m.whatif || [];
  // a.md içinde "<!-- çözüm -->" satırından sonrası çözülmüş adımlardır: başta gizli.
  if (m.md.a && m.md.a.includes("<!-- çözüm -->")) { const [q, sol] = m.md.a.split("<!-- çözüm -->"); m.md.a = q; m.md.aSol = sol; }
  const LEVELS = [0, 1, 2, 3];
  for (const [i, x] of m.practice.entries()) {
    x.id = x.id || `p${i + 1}`;
    if (!PTYPES[x.type]) throw new Error(`${id}/${x.id}: bilinmeyen pratik türü ${x.type}`);
    if (!LEVELS.includes(x.level)) throw new Error(`${id}/${x.id}: level 0–3 olmalı`);
    if (!x.solution) throw new Error(`${id}/${x.id}: açıklamalı çözüm yok`);
    if (x.level > 0 && !(x.hints || []).length) throw new Error(`${id}/${x.id}: ipucu yok`);
    if (x.choices && !x.choices.some((c) => c.ok)) throw new Error(`${id}/${x.id}: doğru seçenek yok`);
  }
  for (const L of m.labs) if (!existsSync(join(root, "site/js/labs", `${L.lab}.js`))) throw new Error(`${id}: deney dosyası yok: ${L.lab}`);
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
  const links = [["mufredat.html", "Müfredat"], ["harita.html", "Önkoşul haritası"], ["tekrar.html", "Karışık tekrar"], ["odevler.html", "Makale ödevleri"], ["notlar.html", "Notlarım"], ["sozluk.html", "Sözlük"], ["kapsam.html", "Kapsam tablosu"]];
  return `<nav class="side" aria-label="Müfredat"><ul class="side-links">${links.map(([h, t]) => `<li><a href="${h}">${t}</a></li>`).join("")}</ul>${phases}</nav>`;
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
    <a href="mufredat.html">Müfredat</a><a href="harita.html">Harita</a><a href="tekrar.html">Tekrar</a><a href="odevler.html">Ödevler</a><a href="notlar.html">Notlarım</a><a href="sozluk.html">Sözlük</a>
  </nav>
  <div class="search" role="search"><label class="sr-only" for="q">Sitede ara</label><input id="q" type="search" placeholder="Ara ( / )" autocomplete="off" spellcheck="false" aria-controls="q-res" aria-expanded="false"><div id="q-res" class="search-res" role="listbox" hidden></div></div>
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
  <p><a href="kaynaklar.html">Kaynaklar ve lisanslar</a> · <a href="kapsam.html">Kapsam tablosu</a> · <a href="degisiklikler.html">Değişiklik günlüğü</a> · <a href="kurulum.html">Motorlar</a></p>
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
    const field = (f) => {
      const hints = f.hints || (f.hint ? [f.hint] : []);
      const mist = (f.mistakes || []).map((x) => ({ v: x.answer, m: md.inline(x.msg) }));
      return `<label class="hf" data-id="${esc(f.id)}" data-answer="${f.answer}" data-tol="${f.tol ?? 0}"${mist.length ? ` data-mistakes="${esc(JSON.stringify(mist))}"` : ""}>
      <span class="hf-label">${md.inline(f.label)}</span>
      <input type="text" inputmode="decimal" autocomplete="off" spellcheck="false" aria-label="${esc(f.label.replace(/\$|\\/g, ""))}">
      <span class="hf-mark" aria-hidden="true"></span>
      <span class="hf-why" role="status" hidden></span>
      ${hints.map((x, i) => `<span class="hf-hint" data-n="${i}" hidden><b>İpucu ${i + 1}/${hints.length}:</b> ${md.inline(x)}</span>`).join("")}</label>`;
    };
    const fields = g.layout === "matrix"
      ? `<div class="hmatrix" style="--cols:${g.cols}">${g.fields.map(field).join("")}</div>`
      : `<div class="hlist">${g.fields.map(field).join("")}</div>`;
    return `<fieldset class="hgroup"><legend>${md.inline(g.title)}</legend>${fields}</fieldset>`;
  }).join("");
  return `<div class="hand" data-stage="a">${groups}
  <div class="hand-actions"><button class="btn primary" data-act="hand-check" type="button">Kontrol et</button><button class="btn" data-act="hand-hint" type="button" title="Yanlış ya da boş alanlar için bir sonraki ipucunu açar">Sıradaki ipucu</button><button class="btn ghost" data-act="hand-reset" type="button">Temizle</button><span class="hand-status" role="status"></span></div>
  ${m.md.aSol ? `<details class="box box-sol hand-sol"><summary><span class="box-kind">Çözüm</span> Çözüm yolu ve yorum (önce kendin dene)</summary><div class="box-body">${md.block(m.md.aSol)}</div></details>` : ""}</div>`;
}

function hintsHTML(list) {
  if (!list?.length) return "";
  return `<div class="cb-hints" data-n="${list.length}"><button class="btn small" type="button" data-act="next-hint">İpucu al (0/${list.length})</button><ol class="hint-list">${list.map((x, i) => `<li data-n="${i}" hidden>${md.inline(x)}</li>`).join("")}</ol></div>`;
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
  ${hintsHTML((m.hints || {})[s])}
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


// --------------------------------------------------------------------- pratik, deney, yanılgı, öz değerlendirme
function practiceItemHTML(m, x) {
  const hints = (x.hints || []).map((h, i) => `<li data-n="${i}" hidden>${md.block(h)}</li>`).join("");
  let answer = "";
  if (x.answer != null) {
    const mist = (x.mistakes || []).map((y) => ({ v: y.answer, m: md.inline(y.msg) }));
    answer = `<div class="pr-answer" data-answer="${x.answer}" data-tol="${x.tol ?? 0}"${mist.length ? ` data-mistakes="${esc(JSON.stringify(mist))}"` : ""}><label>Cevabın <input type="text" inputmode="decimal" autocomplete="off" spellcheck="false">${x.unit ? ` <span class="unit">${md.inline(x.unit)}</span>` : ""}</label><button class="btn primary" type="button" data-act="pr-check">Kontrol et</button><span class="pr-msg" role="status"></span></div>`;
  } else if (x.choices) {
    answer = `<ul class="q-opts pr-choices">${x.choices.map((c) => `<li><button type="button" class="q-opt" data-ok="${c.ok ? 1 : 0}">${md.inline(c.t)}</button><div class="q-why" hidden>${md.inline(c.why || "")}</div></li>`).join("")}</ul>`;
  } else if (x.rubric) {
    answer = `<div class="pr-open"><label class="sr-only" for="pr-${m.id}-${x.id}">Cevabın</label><textarea id="pr-${m.id}-${x.id}" rows="4" placeholder="Cevabını buraya yaz (bu cihazda saklanır). Sonra çözümü açıp ölçütlerle kendini değerlendir."></textarea></div>`;
  }
  const rubric = x.rubric ? `<fieldset class="pr-rubric"><legend>Değerlendirme ölçütleri (çözümü okuduktan sonra işaretle)</legend>${x.rubric.map((r, i) => `<label><input type="checkbox" data-r="${i}"> ${md.inline(r)}</label>`).join("")}</fieldset>` : "";
  const worked = x.level === 0;
  return `<article class="pr" id="pr-${esc(x.id)}" data-pid="${esc(x.id)}" data-level="${x.level}" data-type="${esc(x.type)}">
<header class="pr-head"><span class="pr-level l${x.level}">${LEVEL_NAME[x.level]}</span><span class="pr-type">${PTYPES[x.type]}</span>${x.title ? `<h3>${md.inline(x.title)}</h3>` : ""}</header>
<dl class="pr-meta"><div><dt>Hedef</dt><dd>${md.inline(x.goal || "")}</dd></div>${x.needs ? `<div><dt>Gerekenler</dt><dd>${md.inline(x.needs)}</dd></div>` : ""}</dl>
<div class="prose pr-q">${md.block(x.q)}</div>
${answer}
${hints ? `<div class="pr-hints"><button class="btn small" type="button" data-act="pr-hint">İpucu al (0/${x.hints.length})</button><ol class="hint-list">${hints}</ol></div>` : ""}
<details class="box box-sol pr-sol"${worked ? " open" : ""}><summary><span class="box-kind">${worked ? "Adım adım çözüm" : "Açıklamalı çözüm"}</span>${worked ? "" : " (önce kendin dene)"}</summary><div class="box-body">${md.block(x.solution)}${x.alt ? `<h4>Başka bir yol</h4>${md.block(x.alt)}` : ""}</div></details>
${rubric}
</article>`;
}

function practiceHTML(m) {
  if (!m.practice.length) return "";
  const sorted = [...m.practice].sort((a, b) => a.level - b.level);
  const counts = [0, 1, 2, 3].map((l) => m.practice.filter((x) => x.level === l).length);
  const filt = `<div class="pr-filter" role="group" aria-label="Zorluğa göre süz"><button class="btn small on" type="button" data-lvl="all">Hepsi (${m.practice.length})</button>${counts.map((n, l) => n ? `<button class="btn small" type="button" data-lvl="${l}">${LEVEL_NAME[l]} (${n})</button>` : "").join("")}</div>`;
  return `<section class="practice" id="pratik"><h2>Pratik</h2><p class="muted">Kolaydan zora: çözülmüş örnekler → temel → uygulama → zorlayıcı. İpuçları kademeli açılır; çözüm başta kapalı. Sayısal sorularda sık yapılan hatalar tanınır ve nedeni söylenir; açık uçlu sorularda ölçütlerle kendini değerlendirirsin.</p>${filt}<div class="pr-list">${sorted.map((x) => practiceItemHTML(m, x)).join("")}</div></section>`;
}

function labHTML(m, L) {
  const presets = (L.presets || []).map((p, i) => `<button class="btn small" type="button" data-preset="${i}" title="${esc((p.note || "").replace(/[$\\]/g, ""))}">${md.inline(p.name)}</button>`).join("");
  const predict = L.predict ? `<div class="lab-predict"><p><b>Önce tahmin et:</b> ${md.inline(L.predict.q)}</p><ul class="q-opts">${L.predict.options.map((o) => `<li><button type="button" class="q-opt" data-ok="${o.ok ? 1 : 0}">${md.inline(o.t)}</button><div class="q-why" hidden>${md.inline(o.why || "")}</div></li>`).join("")}</ul><p class="muted small">Seçimini yaptıktan sonra deneyde doğrula.</p></div>` : "";
  const qs = (L.questions || []).map((q) => `<details class="lab-q"><summary>${md.inline(q.q)}</summary><div class="prose">${md.block(q.a)}</div></details>`).join("");
  return `<section class="lab" id="lab-${esc(L.id)}" data-lab-id="${esc(L.id)}" data-lab="${esc(L.lab)}">
<h3><span class="lab-tag">Deney</span> ${md.inline(L.title)}</h3>
<div class="prose">${md.block(L.goal || "")}</div>
${predict}
<div class="lab-ui" aria-live="polite"><noscript>Bu deney JavaScript gerektirir.</noscript></div>
${presets ? `<div class="lab-presets"><span class="muted">Hazır senaryolar:</span> ${presets}</div><p class="lab-preset-note muted" role="status"></p>` : ""}
${qs ? `<div class="lab-qs"><h4>Deneyle cevapla</h4>${qs}</div>` : ""}
${L.limits ? `<p class="lab-limits"><b>Modelin sınırları:</b> ${md.inline(L.limits)}</p>` : ""}
</section>`;
}

function pitfallsHTML(m) {
  if (!m.pitfalls.length && !m.whatif.length) return "";
  const pf = m.pitfalls.map((x) => `<li class="pf"><p class="pf-wrong"><span class="pf-tag">Yanılgı</span> ${md.inline(x.wrong)}</p><p class="pf-right"><span class="pf-tag ok">Doğrusu</span> ${md.inline(x.right)}</p>${x.why ? `<div class="prose small">${md.block(x.why)}</div>` : ""}</li>`).join("");
  const wi = m.whatif.map((x) => `<details class="box box-whatif"><summary><span class="box-kind">Koşul değişirse</span> ${md.inline(x.q)}</summary><div class="box-body">${md.block(x.a)}</div></details>`).join("");
  return `<section class="pitfalls" id="yanilgilar"><h2>Sık hatalar ve “ya şöyle olsaydı?”</h2>${pf ? `<ul class="pf-list">${pf}</ul>` : ""}${wi ? `<p class="muted">Cevabı açmadan önce tahminini yaz.</p>${wi}` : ""}</section>`;
}

function readinessData(m) {
  // önkoşul modüllerinin sınama sorularından ilk ikisi: başlangıç yoklaması
  const out = [];
  for (const p of m.prereq || []) {
    const q = modules[p]?.quiz || [];
    for (const x of q.slice(0, 2)) out.push({ mod: p, code: modules[p].code, q: md.inline(x.q), o: x.options.map((o) => ({ t: md.inline(o.t), ok: !!o.ok, why: md.inline(o.why || "") })) });
  }
  return out.slice(0, 6);
}

function placeLabs(m, html, placed) {
  return html.replace(/<p>::lab ([\w-]+)<\/p>/g, (_, lid) => {
    const L = m.labs.find((x) => x.id === lid);
    if (!L) throw new Error(`${m.id}: yerleştirilen deney yok: ${lid}`);
    placed.add(lid);
    return labHTML(m, L);
  });
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
  const placed = new Set();
  const R = (txt) => placeLabs(m, md.block(txt), placed);
  const railExtra = (m.labs.length ? `<li><a href="#${m.labs.length === 1 ? "lab-" + m.labs[0].id : "deney"}" data-rail="deney"><span class="rail-n">≈</span><span class="rail-name">Deney</span></a></li>` : "")
    + (m.practice.length ? `<li><a href="#pratik" data-rail="pratik"><span class="rail-n">✓</span><span class="rail-name">Pratik</span></a></li>` : "")
    + (m.extras.length ? `<li><a href="#ek" data-rail="ek"><span class="rail-n">+${m.extras.length}</span><span class="rail-name">Ek</span></a></li>` : "")
    + (m.homework.length ? `<li><a href="#odev" data-rail="odev"><span class="rail-n">✎</span><span class="rail-name">Ödev</span></a></li>` : "");
  const rail = `<ol class="rail" aria-label="Aşamalar">${STAGES.map((s, i) => `<li><a href="#${s.k}" data-rail="${s.k}"><span class="rail-n">${i + 1}</span><span class="rail-name">${s.name}</span></a></li>`).join("")}${railExtra}</ol>`;
  const stageSec = STAGES.map((s, i) => {
    const eng = s.k === "a" ? "" : `<span class="stage-engine" data-engine-badge="${m.stages[s.k].engine}"></span>`;
    return `<section class="stage" id="${s.k}" data-stage="${s.k}">
<h2><span class="stage-n">${i + 1}</span> ${s.name} <span class="stage-long">${s.long}</span>${eng}</h2>
<div class="prose">${R(m.md[s.k])}</div>
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
  <details class="goals" open><summary>Bu modülün sonunda</summary><ul class="goal-list">${m.goals.map((g, i) => `<li><label><input type="checkbox" data-goal="${i}" aria-label="Öz değerlendirme: bunu yapabiliyorum"> <span>${md.inline(g)}</span></label></li>`).join("")}</ul><p class="muted small">Kutuları, sonundaki ölçütle kendini sınayınca işaretle: “Bunu birine anlatıp kâğıtta ve kodda yapabiliyorum.” Sayfayı açmak ilerleme sayılmaz; aşama işaretleri yalnız kontrol geçince konur.</p></details>
  <div class="mod-tools"><button class="btn small" type="button" data-act="bookmark" aria-pressed="false">Yer imi ekle</button><details class="notes"><summary>Notlarım</summary><label class="sr-only" for="note-${id}">Bu modül için notların</label><textarea id="note-${id}" rows="5" placeholder="Bu modüle notların (yalnız bu cihazda saklanır; Notlarım sayfasında hepsi bir arada)."></textarea></details></div>
  ${readinessData(m).length ? `<details class="ready-check" id="hazir"><summary>Hazır mısın? Önkoşul yoklaması (${readinessData(m).length} soru)</summary><div class="rc-body"></div></details>` : ""}
</header>
<div class="rail-wrap">${rail}</div>
<section class="intro prose">${R(m.md.intro)}</section>
${m.labs.filter((L) => !placed.has(L.id)).length ? `<section class="labs" id="deney"><h2>Deney alanı</h2>${m.labs.filter((L) => !placed.has(L.id)).map((L) => labHTML(m, L)).join("")}</section>` : ""}
${stageSec}
${extrasHTML(m)}
${m.md.outro ? `<section class="outro prose">${R(m.md.outro)}</section>` : ""}
${practiceHTML(m)}
${pitfallsHTML(m)}
${quizHTML(m)}
<section class="review-sec" id="geri-don" hidden><h2>Geri dönmen önerilen yerler</h2><p class="muted">Sınamada ilk denemede yanlış yaptığın sorular, işaretlemediğin kazanımlar ve geçmediğin aşamalar.</p><ul class="review-list"></ul></section>
${homeworkHTML(m)}
<section class="mod-sources"><h2>Kaynaklar</h2>${sourceList(m.sources)}</section>
<nav class="pager">${prev ? `<a class="prev" href="${prev.id}.html"><small>Önceki</small>${esc(prev.code)} ${esc(prev.title)}</a>` : "<span></span>"}${next ? `<a class="next" href="${next.id}.html"><small>Sonraki</small>${esc(next.code)} ${esc(next.title)}</a>` : "<span></span>"}</nav>
</article>`;
  const keys = ["b", "c", "d", "e", ...m.extras.map((x) => x.id)];
  const labsData = Object.fromEntries(m.labs.map((L) => [L.id, { lab: L.lab, presets: (L.presets || []).map((p) => ({ params: p.params, note: md.inline(p.note || "") })), init: L.params || {} }]));
  const data = { id, code: m.code, ready: readinessData(m), labs: labsData, goals: m.goals.length, quizN: (m.quiz || []).length, stages: Object.fromEntries(keys.map((s) => [s, { engine: m.stages[s].engine, start: m.files[s].start, sol: m.files[s].sol, check: m.files[s].check }])) };
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


// --------------------------------------------------------------------- harita, tekrar, notlar, kapsam, arama
function mapPage() {
  const W = 190, H = 46, GX = 70, GY = 16, PADT = 40;
  const pos = {};
  curriculum.phases.forEach((p, ci) => p.modules.forEach((x, ri) => { pos[x.id] = { x: 20 + ci * (W + GX), y: PADT + ri * (H + GY), ph: p.n, ...x }; }));
  const width = 20 + curriculum.phases.length * (W + GX), height = PADT + Math.max(...curriculum.phases.map((p) => p.modules.length)) * (H + GY) + 20;
  const edges = [];
  for (const id of Object.keys(pos)) for (const q of modules[id]?.prereq || []) {
    const a = pos[q], b = pos[id];
    if (!a) continue;
    const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
    const d = a.ph === b.ph ? `M${a.x + W - 12},${a.y + H} C${a.x + W + 30},${(a.y + b.y) / 2 + H / 2} ${b.x + W + 30},${(a.y + b.y) / 2 + H / 2} ${b.x + W - 12},${b.y}` : `M${x1},${y1} C${x1 + GX * 0.6},${y1} ${x2 - GX * 0.6},${y2} ${x2},${y2}`;
    edges.push(`<path class="e" data-from="${q}" data-to="${id}" d="${d}"/>`);
  }
  const heads = curriculum.phases.map((p, ci) => `<text class="ph" x="${20 + ci * (W + GX)}" y="22">Faz ${p.n} · ${esc(p.title.length > 26 ? p.title.slice(0, 25) + "…" : p.title)}</text>`).join("");
  const nodes = Object.values(pos).map((n) => `<a href="${n.id}.html" class="n" data-mod="${n.id}"><title>${esc(n.code + " " + n.title)}${(modules[n.id]?.prereq || []).length ? " — önkoşul: " + modules[n.id].prereq.map((q) => pos[q]?.code).join(", ") : ""}</title><rect x="${n.x}" y="${n.y}" width="${W}" height="${H}" rx="7"/><text x="${n.x + 10}" y="${n.y + 18}" class="c">${esc(n.code)}</text><text x="${n.x + 10}" y="${n.y + 35}" class="t">${esc(n.title.length > 27 ? n.title.slice(0, 26) + "…" : n.title)}</text><g class="pips">${[0, 1, 2, 3, 4].map((i) => `<rect x="${n.x + W - 60 + i * 10}" y="${n.y + 10}" width="7" height="7" rx="1.5" data-i="${i}"/>`).join("")}</g></a>`).join("");
  // önerilen sıra: önkoşullara saygılı topolojik sıra (müfredat sırası eşitlik bozucu)
  const idx = Object.fromEntries(order.map((o, i) => [o.id, i]));
  const done = new Set(), seq = [];
  while (seq.length < order.length) {
    const nx = order.find((o) => !done.has(o.id) && (modules[o.id]?.prereq || []).every((q) => done.has(q) || idx[q] === undefined));
    if (!nx) throw new Error("önkoşul grafiğinde döngü");
    done.add(nx.id); seq.push(nx);
  }
  const list = seq.map((o) => `<li data-mod="${o.id}"><a href="${o.id}.html">${esc(o.code)} ${esc(o.title)}</a>${(modules[o.id]?.prereq || []).length ? ` <span class="muted small">← ${modules[o.id].prereq.map((q) => esc(pos[q]?.code || q)).join(", ")}</span>` : ""}</li>`).join("");
  const body = `<header class="page-head"><h1>Önkoşul haritası</h1><p class="lede">Her ok “önce bunu bitir” demek. Bir modülün üzerine gelince (ya da klavyeyle odaklanınca) önkoşulları ve onu gerektiren modüller vurgulanır; kutucuklar o modülde geçtiğin aşamaları gösterir. Faz 0 → 1 → 3 omurgadır; diğer fazlar ihtiyaca göre paralel gider.</p></header>
<div class="map-wrap" tabindex="0" aria-label="Önkoşul haritası (yatay kaydırılabilir)"><svg class="map" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-labelledby="map-t"><title id="map-t">Modüller ve önkoşul okları, fazlara göre sütunlarda</title><defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z"/></marker></defs><g class="edges">${edges.join("")}</g>${heads}${nodes}</svg></div>
<section><h2>Önerilen öğrenme sırası</h2><p class="muted">Her modül, önkoşulları bittikten sonra gelecek şekilde sıralandı. Metin sürümü, haritayı göremeyen ya da ekran okuyucu kullananlar için de eksiksizdir.</p><ol class="seq">${list}</ol></section>`;
  return page({ title: "Önkoşul haritası", body, bodyClass: "page-map" });
}

function reviewPage() {
  const body = `<header class="page-head"><h1>Karışık tekrar</h1><p class="lede">Çalıştığın modüllerin sınama sorularından aralıklı tekrar (Leitner kutuları: doğru bildiğin soru daha seyrek, yanlış bildiğin hemen yeniden gelir). Sorular modüller arasında karıştırılır; bu, konuları birbirinden ayırt etmeyi de çalıştırır.</p>
<div class="rv-tools"><label>Kapsam <select id="rv-scope"><option value="touched">Çalıştığım modüller</option><option value="all">Tüm modüller</option></select></label> <span id="rv-stat" class="muted" role="status"></span> <button class="btn ghost small" type="button" id="rv-reset">Tekrar geçmişini sıfırla</button></div></header>
<section id="rv-card" class="rv-card" aria-live="polite"><p class="muted">Yükleniyor…</p></section>`;
  return page({ title: "Karışık tekrar", body, bodyClass: "page-review" });
}

function notesPage() {
  const body = `<header class="page-head"><h1>Notlarım ve yer imlerim</h1><p class="lede">Modül sayfalarında yazdığın notlar, eklediğin yer imleri ve kaldığın yer. Hepsi yalnız bu tarayıcıda saklanır; dışa aktarıp başka cihaza taşıyabilirsin.</p>
<div class="progress-tools"><button class="btn" type="button" data-act="notes-export">Notları ve yer imlerini dışa aktar</button><label class="btn">İçe aktar<input type="file" accept="application/json" data-act="notes-import" hidden></label></div></header>
<section><h2>Kaldığın yer</h2><div id="resume-box" class="muted">Henüz bir modülde çalışmadın.</div></section>
<section><h2>Yer imleri</h2><ul id="bm-list" class="bm-list"><li class="muted">Yer imi yok. Modül sayfasının başındaki “Yer imi ekle” düğmesini kullan.</li></ul></section>
<section><h2>Notlar</h2><div id="note-list"><p class="muted">Not yok.</p></div></section>
<script type="application/json" id="mod-titles">${JSON.stringify(Object.fromEntries(ready.map((r) => [r.id, `${r.code} ${r.title}`]))).replace(/</g, "\\u003c")}</script>`;
  return page({ title: "Notlarım", body });
}

function coveragePage() {
  const cov = yaml(C("coverage.yaml"));
  const ST = { tam: "kapsandı", genis: "kapsandı + genişletildi", kismi: "kısmen", eksik: "eksik", bilincli: "bilinçli olarak farklı" };
  const link = (id) => modules[id] ? `<a href="${id}.html">${esc(modules[id].code)}</a>` : esc(id);
  const groups = cov.groups.map((g) => `<section class="cov-g"><h2>${esc(g.title)}</h2>${g.note ? `<p class="muted">${md.inline(g.note)}</p>` : ""}<div class="tbl-wrap"><table class="cov"><thead><tr><th>Kaynak bölümü</th><th>Sayfa(lar)</th><th>Durum</th><th>Kaynaktaki eksik / fark</th><th>Yapılan geliştirme</th></tr></thead><tbody>${g.rows.map((r) => {
    for (const id of r.pages || []) if (!modules[id]) throw new Error(`kapsam: bilinmeyen modül ${id}`);
    if (!ST[r.status]) throw new Error(`kapsam: bilinmeyen durum ${r.status}`);
    return `<tr><td><strong>${esc(r.src)}</strong>${r.topics ? `<br><span class="small muted">${md.inline(r.topics)}</span>` : ""}</td><td>${(r.pages || []).map(link).join(", ")}</td><td><span class="cov-st ${r.status}">${ST[r.status]}</span></td><td>${md.inline(r.gap || "—")}</td><td>${md.inline(r.done || "—")}</td></tr>`;
  }).join("")}</tbody></table></div></section>`).join("");
  const body = `<header class="page-head"><h1>Kapsam tablosu</h1><div class="lede prose">${md.block(cov.intro)}</div></header>${groups}`;
  return page({ title: "Kapsam tablosu", body });
}

function searchIndex() {
  const plain = (h) => String(h || "").replace(/<[^>]+>/g, " ").replace(/\$[^$]*\$/g, " ").replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, a, b) => b || a).replace(/[*_`#>]/g, "").replace(/\s+/g, " ").trim();
  const out = [];
  for (const r of ready) {
    const m = modules[r.id];
    out.push({ t: `${m.code} ${m.title}`, u: `${r.id}.html`, k: "modül", s: plain(m.summary).slice(0, 220) });
    for (const g of m.goals) out.push({ t: plain(g), u: `${r.id}.html`, k: `${m.code} kazanım`, s: "" });
    for (const txt of [m.md.intro, m.md.outro, ...STAGES.map((s) => m.md[s.k])]) for (const [, h] of String(txt || "").matchAll(/^#{2,4}\s+(.+?)(?:\s*\{#[\w-]+\})?$/gm)) out.push({ t: plain(h), u: `${r.id}.html`, k: `${m.code} başlık`, s: "" });
    for (const [, kind, t] of [m.md.intro, m.md.outro, ...STAGES.map((s) => m.md[s.k])].join("\n").matchAll(/^:::([^\s:]+)[ \t]+(.+)$/gm)) out.push({ t: plain(t), u: `${r.id}.html`, k: `${m.code} ${kind}`, s: "" });
    m.extras.forEach((x) => out.push({ t: plain(x.title), u: `${r.id}.html#${x.id}`, k: `${m.code} ek alıştırma`, s: "" }));
    m.labs.forEach((L) => out.push({ t: plain(L.title), u: `${r.id}.html#lab-${L.id}`, k: `${m.code} deney`, s: "" }));
    m.practice.forEach((x) => out.push({ t: plain(x.title || x.goal), u: `${r.id}.html#pr-${x.id}`, k: `${m.code} pratik`, s: "" }));
    m.homework.forEach((h) => out.push({ t: plain(h.title), u: `${r.id}.html#odev`, k: `${m.code} ödev`, s: plain(h.paper).slice(0, 120) }));
  }
  for (const [k, v] of Object.entries(yaml(C("glossary.yaml")).terms)) out.push({ t: k, u: `sozluk.html#t-${k.replace(/\s+/g, "-")}`, k: "sözlük", s: v.slice(0, 160) });
  return JSON.stringify(out);
}

function reviewData() {
  return JSON.stringify(ready.flatMap((r) => (modules[r.id].quiz || []).map((q, i) => ({ id: `${r.id}#${i}`, mod: r.id, code: modules[r.id].code, q: md.inline(q.q), o: q.options.map((o) => ({ t: md.inline(o.t), ok: !!o.ok, why: md.inline(o.why || "") })) }))));
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
  "harita.html": mapPage(),
  "tekrar.html": reviewPage(),
  "notlar.html": notesPage(),
  "kapsam.html": coveragePage(),
  "degisiklikler.html": markdownPage("degisiklikler.md", "Değişiklik günlüğü"),
};
writeFileSync(join(out, "search.json"), searchIndex());
writeFileSync(join(out, "review.json"), reviewData());
for (const r of ready) pages[`${r.id}.html`] = modulePage(r.id);
for (const [f, html] of Object.entries(pages)) writeFileSync(join(out, f), html);

let files = 0, bytes = 0;
(function walk(d) { for (const f of readdirSync(d)) { const p = join(d, f), s = statSync(p); if (s.isDirectory()) walk(p); else { files++; bytes += s.size; } } })(out);
console.log(`dist/: ${Object.keys(pages).length} sayfa (${ready.length} modül), ${files} dosya, ${(bytes / 1024 / 1024).toFixed(1)} MB`);
