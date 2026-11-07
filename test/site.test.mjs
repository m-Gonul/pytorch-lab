// Site yapısı: build çalışır, her iç bağlantı bir dosyaya gider, her modül sayfası
// 5 aşamayı ve 4 kod kutusunu taşır, notebook'lar güncel.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..");
const dist = join(root, "dist");
execFileSync("node", [join(root, "scripts/build.mjs")], { stdio: "pipe" });
const pages = readdirSync(dist).filter((f) => f.endsWith(".html"));
const modules = readdirSync(join(root, "content/modules")).filter((d) => existsSync(join(root, "content/modules", d, "module.yaml")));

test("her modülün bir sayfası var", () => {
  for (const m of modules) assert.ok(pages.includes(`${m}.html`), `${m}.html yok`);
});

test("iç bağlantılar ve varlıklar mevcut", () => {
  const bad = [];
  for (const p of pages) {
    const html = readFileSync(join(dist, p), "utf8");
    for (const [, href] of html.matchAll(/(?:href|src)="([^"#?]+)(?:[#?][^"]*)?"/g)) {
      if (/^(https?:|mailto:|data:)/.test(href) || href === "") continue;
      if (!existsSync(join(dist, href))) bad.push(`${p} → ${href}`);
    }
  }
  assert.deepEqual(bad, []);
});

test("modül sayfaları 5 aşama, el hesabı ve 4 kod kutusu taşıyor", () => {
  for (const m of modules) {
    const html = readFileSync(join(dist, `${m}.html`), "utf8");
    for (const s of "abcde") assert.match(html, new RegExp(`<section class="stage" id="${s}"`), `${m}: aşama ${s}`);
    const nx = (html.match(/<section class="stage extra"/g) || []).length;
    assert.equal((html.match(/class="codebox"/g) || []).length, 4 + nx, `${m}: kod kutusu sayısı`);
    assert.ok((html.match(/class="hf"/g) || []).length >= 4, `${m}: el hesabı alanları`);
    const data = JSON.parse(html.match(/<script type="application\/json" id="page-data">(.*?)<\/script>/s)[1]);
    assert.equal(data.id, m);
    for (const s of "bcde") for (const k of ["start", "sol", "check"]) assert.ok(data.stages[s][k]?.length > 20, `${m}/${s}.${k} boş`);
  }
});

test("ödevler sayfası var", () => {
  assert.ok(pages.includes("odevler.html"));
});

test("KaTeX hatası yok", () => {
  for (const p of pages) assert.doesNotMatch(readFileSync(join(dist, p), "utf8"), /katex-error/, p);
});

test("notebook'lar içerikle güncel", () => {
  execFileSync("node", [join(root, "scripts/notebooks.mjs"), "--check"], { stdio: "pipe" });
});

test("runner ve labcheck yayında", () => {
  assert.ok(existsSync(join(dist, "runner/torchlab.py")));
  assert.ok(existsSync(join(dist, "py/labcheck.py")));
  assert.match(readFileSync(join(dist, "runner/torchlab.py"), "utf8"), /# \/\/\/ script/);
});
