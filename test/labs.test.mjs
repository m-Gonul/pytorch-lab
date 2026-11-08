// Deney alanları: her deney varsayılanlarla ve hazır senaryolarla hatasız çalışır, her kontrol
// sonucu gerçekten değiştirir (süs kontrol yok), geçersiz değerler açıklamalı reddedilir.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import YAML from "yaml";

const root = join(import.meta.dirname, "..");
const labDir = join(root, "site/js/labs");
const { rng, plot } = await import(join(labDir, "plot.js"));
const { evaluate, defaults } = await import(join(labDir, "runtime.js"));
// görüntü çizimi için en küçük DOM taklidi
globalThis.document = { createElement: () => ({ getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }), toDataURL: () => "data:image/png;base64," }) };

const specs = {};
for (const f of readdirSync(labDir)) if (!["plot.js", "runtime.js"].includes(f)) specs[f.replace(/\.js$/, "")] = (await import(join(labDir, f))).default;

const finiteDeep = (o, path = "") => {
  if (typeof o === "number") { assert.ok(!Number.isNaN(o), `NaN: ${path}`); return; }
  if (Array.isArray(o)) o.forEach((v, i) => finiteDeep(v, `${path}[${i}]`));
  else if (o && typeof o === "object") for (const [k, v] of Object.entries(o)) finiteDeep(v, `${path}.${k}`);
};
const sig = (out) => JSON.stringify(out, (k, v) => (typeof v === "number" ? Number(v.toPrecision(6)) : v));

for (const [name, spec] of Object.entries(specs)) {
  if (spec.custom) {
    test(`etkileşimli şekil ${name}: kendi hesap testleri`, () => {
      assert.equal(typeof spec.mount, "function");
      assert.equal(typeof spec.selfTest, "function", "custom şekiller hesaplarını selfTest ile doğrulamalı");
      spec.selfTest(assert);
    });
    continue;
  }
  test(`deney ${name}: varsayılanla çalışır, çizilir`, () => {
    const r = evaluate(spec, defaults(spec), 1);
    assert.ok(r.out, `${name}: ${JSON.stringify(r.issues)}`);
    assert.ok(r.out.plots.length >= 1 && r.out.stats.length >= 1, "grafik ve ölçüm var");
    for (const P of r.out.plots) { finiteDeep(P.series?.map((s) => s.y.filter((v) => v != null && Number.isFinite(v)))); assert.match(plot(P), /<svg/); }
    assert.ok(typeof r.out.note === "string" && r.out.note.length > 40, "yorum metni var");
    // aynı tohum → aynı sonuç (tekrarlanabilirlik)
    assert.equal(sig(evaluate(spec, defaults(spec), 7).out), sig(evaluate(spec, defaults(spec), 7).out));
  });

  test(`deney ${name}: her kontrol sonucu değiştirir`, () => {
    const base0 = defaults(spec);
    for (const q of spec.params) {
      const base = { ...base0, ...Object.fromEntries(Object.entries(q.on || {}).map(([k, v]) => [k, v.includes(base0[k]) ? base0[k] : v[0]])) }, b0 = sig(evaluate(spec, base, 1).out);
      const alts = q.options ? q.options.map((_, i) => i).filter((i) => i !== base[q.id]) : [q.max, q.min, (q.min + q.max) / 2].map((v) => (q.int ? Math.round(v) : v)).filter((v) => v !== base[q.id]);
      const changed = alts.some((v) => { const r = evaluate(spec, { ...base, [q.id]: v }, 1); return !r.out || sig(r.out) !== b0; });
      assert.ok(changed, `${name}.${q.id} sonucu etkilemiyor`);
    }
    if (spec.seed) { const sb = { ...base0, ...(spec.seedOn || {}) }; assert.notEqual(sig(evaluate(spec, sb, 2).out), sig(evaluate(spec, sb, 1).out), `${name}: tohum etkisiz`); }
  });

  test(`deney ${name}: aralık dışı değer açıklamayla reddedilir`, () => {
    const q = spec.params.find((x) => !x.options);
    const r = evaluate(spec, { ...defaults(spec), [q.id]: (q.hardMax ?? q.max) + 1e6 }, 1);
    assert.ok(!r.out && r.issues.some((x) => x.level === "error" && x.msg.includes(q.label)), `${name}.${q.id}`);
    const r2 = evaluate(spec, { ...defaults(spec), [q.id]: NaN }, 1);
    assert.ok(!r2.out);
  });
}

test("conv1d: sayılan çıktı boyu her ayarda formüle eşit", () => {
  const s = specs.conv1d, R = rng(3);
  for (let i = 0; i < 300; i++) {
    const p = { ker: Math.floor(R() * 5), k: 1 + 2 * Math.floor(R() * 10), s: 1 + Math.floor(R() * 8), pad: Math.floor(R() * 3), dil: 1 + Math.floor(R() * 6), noise: 0.1 };
    const r = evaluate(s, p, 1);
    if (!r.out) continue;
    const st = Object.fromEntries(r.out.stats.map((x) => [x.label, x.value]));
    assert.equal(st["Çıktı uzunluğu (sayılan)"], st["⌊(n + 2P − ((k−1)d+1))/s⌋ + 1"], JSON.stringify(p));
  }
});

test("gd1d: kararlılık sınırı 2/a", () => {
  const s = specs.gd1d;
  for (const a of [0.5, 2, 7]) {
    const end = (lr) => Math.abs(evaluate(s, { a, lr, x0: 1, T: 60 }, 1).out.plots[0].series[0].y.at(-1));
    assert.ok(end(1.95 / a) < 1 && end(2.05 / a) > 1, `a=${a}`);
  }
});

test("modüllerdeki deney tanımları geçerli (dosya, senaryo parametreleri)", () => {
  const mods = join(root, "content/modules");
  let n = 0;
  for (const id of readdirSync(mods)) {
    const f = join(mods, id, "module.yaml");
    if (!existsSync(f)) continue;
    for (const L of YAML.parse(readFileSync(f, "utf8")).labs || []) {
      n++;
      const spec = specs[L.lab];
      assert.ok(spec, `${id}: bilinmeyen deney ${L.lab}`);
      if (spec.custom) { for (const pr of L.presets || []) for (const k of Object.keys(pr.params || {})) assert.ok((spec.keys || []).includes(k), `${id}/${L.id}: bilinmeyen ayar ${k}`); continue; }
      const ids = new Set(spec.params.map((q) => q.id));
      for (const pr of L.presets || []) {
        const { B, seed, ...A } = pr.params || {};
        for (const k of [...Object.keys(A), ...Object.keys(B || {})]) assert.ok(ids.has(k), `${id}/${L.id} senaryo "${pr.name}": bilinmeyen parametre ${k}`);
        const r = evaluate(spec, { ...defaults(spec), ...A }, seed || 1);
        assert.ok(r.out, `${id}/${L.id} senaryo "${pr.name}" çalışmıyor: ${JSON.stringify(r.issues)}`);
        if (B) assert.ok(evaluate(spec, { ...defaults(spec), ...A, ...B }, 1).out, `${id}/${L.id} B`);
      }
      if (L.predict) assert.equal(L.predict.options.filter((o) => o.ok).length, 1, `${id}/${L.id}: tahmin sorusunda tek doğru`);
    }
  }
  assert.ok(n >= 18, `en az 18 deney (${n})`);
});
