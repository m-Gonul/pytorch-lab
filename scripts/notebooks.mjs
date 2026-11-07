// Her modül için Colab/Jupyter notebook'u üretir: notebooks/<id>.ipynb
// Notebook, sitedeki kontrollerin birebir aynısını (labcheck.py + X.check.py) taşır.
//   node scripts/notebooks.mjs          → yaz
//   node scripts/notebooks.mjs --check  → güncel değilse hata ver (CI)
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";

const root = join(import.meta.dirname, "..");
const read = (p) => readFileSync(p, "utf8");
const site = YAML.parse(read(join(root, "content/site.yaml")));
const labcheck = read(join(root, "site/py/labcheck.py"));
const STAGE_NAMES = { a: "Elle", b: "Döngü (saf Python)", c: "NumPy / pandas", d: "PyTorch", e: "Vektörize + hız" };

// Notebook markdown'ı: [[terim|metin]] → metin, {#id} başlık ekleri kalkar.
const nbmd = (s) => String(s ?? "").replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, k, t) => (t || k).trim()).replace(/\s*\{#[\w-]+\}/g, "");
const lines = (s) => { const a = s.split("\n"); return a.map((l, i) => (i < a.length - 1 ? l + "\n" : l)).filter((l, i) => !(i === a.length - 1 && l === "")); };
const md = (s) => ({ cell_type: "markdown", metadata: {}, source: lines(nbmd(s).trim()) });
const code = (s, meta = {}) => ({ cell_type: "code", execution_count: null, metadata: meta, outputs: [], source: lines(s.replace(/\s+$/, "")) });

export function buildNotebook(id) {
  const d = join(root, "content/modules", id);
  const m = YAML.parse(read(join(d, "module.yaml")));
  const f = (n) => (existsSync(join(d, n)) ? read(join(d, n)) : "");
  const page = `${site.site_url}${id}.html`;
  const extras = m.extras || [];
  const checks = Object.fromEntries(["b", "c", "d", "e", ...extras.map((x) => x.id)].map((s) => [s, f(`${s}.check.py`)]));
  const cells = [];
  cells.push(md(`# ${m.code} · ${m.title}\n\n${m.summary.trim()}\n\nEtkileşimli sürüm: [${page}](${page})\n\n**Nasıl çalışılır:** her aşamada önce \`%%writefile\` hücresindeki kodu düzenleyip çalıştır (dosyaya yazar), sonra altındaki **Kontrol** hücresini çalıştır. Kontroller sitedekiyle birebir aynı. GPU gerekmez; Faz 7 modülleri hariç.`));
  cells.push(code(`#@title Kurulum: kontrol kütüphanesi (çalıştır, düzenleme)\n%%writefile labcheck.py\n${labcheck}`, { cellView: "form" }));
  cells.push(code(`#@title Kurulum: bu modülün kontrolleri (çalıştır, düzenleme)\n%%writefile checks_${id.replace(/-/g, "_")}.py\nCHECKS = ${JSON.stringify(checks, null, 1)}`, { cellView: "form" }));
  cells.push(code(`import importlib, labcheck, checks_${id.replace(/-/g, "_")} as _c\nimportlib.reload(labcheck); importlib.reload(_c)\nCHECKS = _c.CHECKS\nprint("hazır")`));
  cells.push(md(`## Giriş\n\n${f("intro.md")}`));
  // (a) el hesabı: cevapları gir, kontrol et
  const fields = (m.hand?.groups || []).flatMap((g) => g.fields.map((x) => ({ ...x, group: g.title })));
  cells.push(md(`## 1 · ${STAGE_NAMES.a}\n\n${f("a.md")}\n\nSitede bu aşama etkileşimli bir formdur. Burada değerleri aşağıdaki sözlüğe yazıp kontrol hücresini çalıştır.\n\n${(m.hand?.groups || []).map((g) => `**${g.title}**\n\n${g.fields.map((x) => `- \`${x.id}\`: ${x.label}`).join("\n")}`).join("\n\n")}`));
  cells.push(code(`cevaplar = dict(\n${fields.map((x) => `    ${x.id}=None,`).join("\n")}\n)`));
  const ans = Object.fromEntries(fields.map((x) => [x.id, [x.answer, x.tol ?? 0]]));
  cells.push(code(`#@title Kontrol (1)\n_A = ${JSON.stringify(ans)}\n_ok = 0\nfor k, (v, tol) in _A.items():\n    g = cevaplar.get(k)\n    good = g is not None and abs(float(g) - v) <= max(tol, 1e-9)\n    _ok += good\n    print(("✓ " if good else ("· " if g is None else "✗ ")) + k)\nprint(f"\\n{_ok}/{len(_A)} doğru")`, { cellView: "form" }));
  for (const [i, s] of ["b", "c", "d", "e"].entries()) {
    cells.push(md(`## ${i + 2} · ${STAGE_NAMES[s]}\n\n${f(`${s}.md`)}`));
    cells.push(code(`%%writefile stage_${s}.py\n${f(`${s}.py`)}`));
    cells.push(code(`# Kontrol (${i + 2}): kodunu çalıştırır ve sitedeki testleri uygular\nlabcheck.run_file("stage_${s}.py", CHECKS["${s}"])`));
  }
  if (extras.length) {
    cells.push(md(`## Ek alıştırmalar\n\nBeş aşamanın dışındaki pratikler; her biri kendi kontrolüyle.`));
    for (const [i, x] of extras.entries()) {
      cells.push(md(`### E${i + 1} · ${x.title}${x.engine === "torch" ? " (PyTorch)" : ""}\n\n${f(`${x.id}.md`)}`));
      cells.push(code(`%%writefile extra_${x.id}.py\n${f(`${x.id}.py`)}`));
      cells.push(code(`# Kontrol (E${i + 1})\nlabcheck.run_file("extra_${x.id}.py", CHECKS["${x.id}"])`));
    }
  }
  if (m.md?.outro || existsSync(join(d, "outro.md"))) cells.push(md(f("outro.md")));
  for (const h of m.homework || []) {
    cells.push(md(`## Makale ödevi: ${h.title}\n\n[${h.paper}](${h.url})\n\n${h.goal}\n\n**Adımlar**\n\n${h.steps.map((x, i) => `${i + 1}. ${x}`).join("\n")}\n\n**Kabul ölçütleri**\n\n${h.accept.map((x) => `- ${x}`).join("\n")}`));
  }
  cells.push(md(`## Kaynaklar\n\n${m.sources.map((s) => `- [${s.title}](${s.url}) — ${s.license}`).join("\n")}`));
  return {
    cells,
    metadata: {
      colab: { provenance: [], toc_visible: true },
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python" },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

const ids = readdirSync(join(root, "content/modules")).filter((x) => existsSync(join(root, "content/modules", x, "module.yaml"))).sort();
const outDir = join(root, "notebooks");
mkdirSync(outDir, { recursive: true });
const checkOnly = process.argv.includes("--check");
let stale = [];
for (const id of ids) {
  const nb = buildNotebook(id);
  nb.cells.forEach((c, i) => (c.id = `${id}-${i}`));
  const text = JSON.stringify(nb, null, 1) + "\n";
  const p = join(outDir, `${id}.ipynb`);
  if (checkOnly) { if (!existsSync(p) || read(p) !== text) stale.push(id); }
  else writeFileSync(p, text);
}
if (checkOnly && stale.length) { console.error(`güncel olmayan notebook'lar: ${stale.join(", ")} — npm run notebooks`); process.exit(1); }
console.log(checkOnly ? `${ids.length} notebook güncel` : `${ids.length} notebook yazıldı → notebooks/`);
