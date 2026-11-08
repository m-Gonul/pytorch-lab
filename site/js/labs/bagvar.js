// M2.5 — Bagging: B ağacın ortalaması varyansı ρσ² + (1 − ρ)σ²/B'ye indirir. Ağaçlar arası korelasyon ρ, ortalamanın indiremeyeceği taban.
import { fmt, linspace } from "./plot.js";
import { regTree, evalT } from "./boost1d.js";

const truth = (x) => Math.sin(2 * x) + 0.4 * x;
function dataset(n, noise, r) { const x = [], y = []; for (let i = 0; i < n; i++) { const xi = 6 * r() - 3; x.push(xi); y.push(truth(xi) + noise * r.normal()); } return { x, y }; }
function draw(n, frac, boot, r) { // örnek indisleri: bootstrap (yerine koyarak, n) ya da alt örnekleme (yerine koymadan, frac·n)
  if (boot) return Array.from({ length: n }, () => Math.floor(r() * n));
  const m = Math.max(4, Math.round(frac * n)), idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx.slice(0, m);
}

export default {
  params: [
    { id: "depth", label: "Ağaç derinliği", min: 1, max: 8, value: 6, int: true, help: "Derin ağaç: düşük yanlılık, yüksek varyans — bagging'in en çok işe yaradığı yer." },
    { id: "B", label: "Gösterilen topluluk boyu B", min: 1, max: 50, value: 20, int: true },
    { id: "samp", label: "Örnekleme", options: ["bootstrap (yerine koyarak, n)", "alt örnekleme (yerine koymadan)"], value: 0 },
    { id: "frac", label: "Alt örnekleme oranı", min: 0.1, max: 0.9, step: 0.05, value: 0.5, on: { samp: [1] }, help: "Küçük oran: ağaçlar daha farklı veri görür, ρ düşer — ama her biri daha az veriyle daha gürültülü (σ² artar)." },
    { id: "noise", label: "Gürültü σ", min: 0.05, max: 1, step: 0.05, value: 0.4 },
  ],
  seed: true,
  run(q, r) {
    const n = 100, D = Math.round(q.depth), boot = q.samp === 0, gx = linspace(-2.8, 2.8, 29), Bs = [1, 2, 3, 5, 10, 20, 50];
    // 1) gösterim: tek veri kümesi
    const S = dataset(n, q.noise, r), trees = Array.from({ length: Math.round(q.B) }, () => { const id = draw(n, q.frac, boot, r); return regTree(id.map((i) => S.x[i]), id.map((i) => S.y[i]), id.map((_, k) => k), D, 2); });
    const fx = linspace(-3, 3, 241), avg = fx.map((v) => trees.reduce((s, T) => s + evalT(T, v), 0) / trees.length);
    // 2) varyans: R veri kümesi × 50 ağaç; her x'te tek ağaç varyansı σ², aynı veri kümesindeki iki ağacın korelasyonu ρ
    const R = 24, P = []; // P[k][b][g]
    for (let k = 0; k < R; k++) { const Dk = dataset(n, q.noise, r); P.push(Array.from({ length: 50 }, () => { const id = draw(n, q.frac, boot, r); const T = regTree(id.map((i) => Dk.x[i]), id.map((i) => Dk.y[i]), id.map((_, j) => j), D, 2); return gx.map((v) => evalT(T, v)); })); }
    const varAvg = (B) => { let s = 0; for (let g = 0; g < gx.length; g++) { const a = P.map((tr) => tr.slice(0, B).reduce((t, p) => t + p[g], 0) / B), m = a.reduce((u, v) => u + v, 0) / R; s += a.reduce((u, v) => u + (v - m) ** 2, 0) / (R - 1); } return s / gx.length; };
    let sig2 = 0, cov = 0;
    for (let g = 0; g < gx.length; g++) {
      const all = P.flatMap((tr) => tr.map((p) => p[g])), m = all.reduce((u, v) => u + v, 0) / all.length;
      sig2 += all.reduce((u, v) => u + (v - m) ** 2, 0) / (all.length - 1);
      let c = 0, cnt = 0; for (const tr of P) for (let b = 0; b < 49; b += 2) { c += (tr[b][g] - m) * (tr[b + 1][g] - m); cnt++; } cov += c / cnt;
    }
    sig2 /= gx.length; cov /= gx.length; const rho = Math.max(0, Math.min(1, cov / sig2));
    const emp = Bs.map(varAvg), th = Bs.map((B) => rho * sig2 + ((1 - rho) * sig2) / B);
    return {
      plots: [
        { title: `Tek veri kümesi: ${trees.length} ağaç (ince) ve ortalamaları (kalın)`, xlabel: "x", ylabel: "y", series: [
          { name: "veri", x: S.x, y: S.y, kind: "scatter", c: 0, r: 2.2 },
          ...trees.slice(0, 8).map((T, k) => ({ name: k ? "" : "tek tek ağaçlar (ilk 8)", x: fx, y: fx.map((v) => evalT(T, v)), c: 4, kind: "step" })),
          { name: "gerçek f(x)", x: fx, y: fx.map(truth), c: 1, dash: true },
          { name: `bagging ortalaması (B = ${trees.length})`, x: fx, y: avg, c: 2 },
        ] },
        { title: "Topluluk tahmininin varyansı (veri kümeleri arasında), B'ye göre", xlabel: "B (log₂)", ylabel: "varyans", series: [
          { name: "deneysel (24 veri kümesi)", x: Bs.map(Math.log2), y: emp, c: 2, dots: true },
          { name: "formül ρσ² + (1 − ρ)σ²/B", x: Bs.map(Math.log2), y: th, c: 0, dash: true },
        ], marks: [{ y: rho * sig2, label: "taban ρσ²" }] },
      ],
      stats: [
        { label: "Tek ağacın varyansı σ² (x'ler üzerinde ort.)", value: sig2 },
        { label: "Aynı veriden iki ağacın korelasyonu ρ", value: rho },
        { label: "B = 50 ile varyans (deneysel / formül)", value: `${fmt(emp.at(-1), 3)} / ${fmt(th.at(-1), 3)}` },
        { label: "Varyans azalması (B = 1 → 50)", value: `${fmt(emp[0] / emp.at(-1), 3)} kat` },
      ],
      note: `Ortalama alma, ağaçların <b>birbirinden bağımsız</b> kısmını 1/B ile söndürür; ortak kısım (ρσ²) kalır — B ne kadar büyük olursa olsun. Aynı veri kümesinden çekilen bootstrap örneklemleri büyük ölçüde örtüştüğü için ağaçlar ilişkilidir (ρ ≈ ${fmt(rho, 2)}). Rastgele orman, her bölünmede özniteliklerin yalnız rastgele bir alt kümesine bakarak ρ'yu daha da düşürür (burada tek öznitelik olduğu için alt örnekleme oranıyla benzer bir etki gösterilir). Bagging yanlılığı değiştirmez: tek tek ağaçlar sığsa ortalama da sığdır.`,
    };
  },
};
