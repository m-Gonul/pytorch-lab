// M1.3 — Ridge ve lasso düzenlileştirme yolları: katsayılar λ ile nasıl küçülür, hangisi tam sıfıra gider; doğrulama hatası ve λ seçimi.
import { fmt, linspace } from "./plot.js";

export function makeData(q, r, nTot) {
  const d = Math.round(q.d), k = Math.min(Math.round(q.k), d), X = [], y = [];
  const wTrue = Array.from({ length: d }, (_, j) => (j < k ? (j % 2 ? -1 : 1) * (1.5 - j * 0.15) : 0));
  for (let i = 0; i < nTot; i++) {
    const z = r.normal(), row = [];
    for (let j = 0; j < d; j++) row.push(Math.sqrt(q.rho) * z + Math.sqrt(1 - q.rho) * r.normal()); // ortak faktörle korelasyon ρ
    X.push(row); y.push(row.reduce((s, v, j) => s + v * wTrue[j], 0) + q.noise * r.normal());
  }
  return { X, y, wTrue };
}
function center(X, y) { const n = X.length, d = X[0].length, mx = Array(d).fill(0); let my = 0; for (let i = 0; i < n; i++) { my += y[i] / n; for (let j = 0; j < d; j++) mx[j] += X[i][j] / n; } return { Xc: X.map((r) => r.map((v, j) => v - mx[j])), yc: y.map((v) => v - my), mx, my }; }
function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j]; } } return M.map((r, i) => r[n] / r[i]); }
// amaç: (1/2n)‖y − Xw‖² + λ·ceza;  ridge ceza ½‖w‖², lasso ‖w‖₁
export function ridge(Xc, yc, lam) { const n = Xc.length, d = Xc[0].length; const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => Xc.reduce((s, r) => s + r[i] * r[j], 0) / n + (i === j ? lam : 0))); const b = Array.from({ length: d }, (_, i) => Xc.reduce((s, r, k) => s + r[i] * yc[k], 0) / n); return solve(A, b); }
export const soft = (z, t) => Math.sign(z) * Math.max(Math.abs(z) - t, 0);
export function lasso(Xc, yc, lam, w0) {
  const n = Xc.length, d = Xc[0].length, w = w0 ? [...w0] : Array(d).fill(0), res = yc.map((v, i) => v - Xc[i].reduce((s, x, j) => s + x * w[j], 0)), nr = Array.from({ length: d }, (_, j) => Xc.reduce((s, r) => s + r[j] * r[j], 0) / n);
  for (let it = 0; it < 200; it++) {
    let mx = 0;
    for (let j = 0; j < d; j++) {
      let rho = 0; for (let i = 0; i < n; i++) rho += Xc[i][j] * (res[i] + Xc[i][j] * w[j]); rho /= n;
      const nw = soft(rho, lam) / nr[j], dw = nw - w[j];
      if (dw) { for (let i = 0; i < n; i++) res[i] -= Xc[i][j] * dw; w[j] = nw; mx = Math.max(mx, Math.abs(dw)); }
    }
    if (mx < 1e-7) break;
  }
  return w;
}
const mseOn = (X, y, w, b) => X.reduce((s, r, i) => s + (r.reduce((a, v, j) => a + v * w[j], b) - y[i]) ** 2, 0) / X.length;

export default {
  params: [
    { id: "method", label: "Yöntem", options: ["ridge (L2)", "lasso (L1)"], value: 1 },
    { id: "n", label: "Eğitim örneği n", min: 10, max: 200, value: 40, int: true },
    { id: "d", label: "Öznitelik sayısı d", min: 2, max: 20, value: 10, int: true },
    { id: "k", label: "Gerçekten etkili öznitelik sayısı", min: 1, max: 20, value: 3, int: true },
    { id: "rho", label: "Öznitelikler arası korelasyon ρ", min: 0, max: 0.95, step: 0.05, value: 0.3 },
    { id: "noise", label: "Gürültü σ", min: 0.1, max: 3, step: 0.1, value: 1 },
    { id: "lam", label: "Seçili λ (dikey çizgi)", min: 0.0001, max: 10, value: 0.1, log: true, hardMin: 0 },
  ],
  seed: true,
  validate(q) { return q.k > q.d ? [{ level: "warn", msg: `Etkili öznitelik sayısı (${q.k}) d'den (${q.d}) büyük: k = d alındı.` }] : []; },
  run(q, r) {
    const n = Math.round(q.n), { X, y, wTrue } = makeData(q, r, n + 400), Xtr = X.slice(0, n), ytr = y.slice(0, n), Xva = X.slice(n), yva = y.slice(n);
    const { Xc, yc, mx, my } = center(Xtr, ytr), d = wTrue.length;
    const lams = linspace(Math.log10(1e-4), Math.log10(10), 41).map((e) => 10 ** e).reverse();
    const path = []; let w = null;
    for (const l of lams) { w = q.method === 1 ? lasso(Xc, yc, l, w) : ridge(Xc, yc, l); path.push(w); }
    const bOf = (w) => my - w.reduce((s, v, j) => s + v * mx[j], 0);
    const tr = path.map((w) => mseOn(Xtr, ytr, w, bOf(w))), va = path.map((w) => mseOn(Xva, yva, w, bOf(w)));
    const best = va.indexOf(Math.min(...va)), ws = q.method === 1 ? lasso(Xc, yc, q.lam, null) : ridge(Xc, yc, q.lam);
    const lx = lams.map(Math.log10), nz = ws.filter((v) => Math.abs(v) > 1e-9).length;
    const ols = n > d ? ridge(Xc, yc, 0) : null;
    return {
      plots: [
        { title: `${q.method === 1 ? "Lasso" : "Ridge"} yolu: katsayılar λ'ya göre (kalın = gerçekten etkili)`, xlabel: "log₁₀ λ", ylabel: "katsayı wⱼ", series: Array.from({ length: d }, (_, j) => ({ name: j < 6 ? `w${j + 1}${wTrue[j] ? ` (gerçek ${fmt(wTrue[j], 2)})` : " (gerçek 0)"}` : "", x: lx, y: path.map((p) => p[j]), c: j < Math.min(q.k, d) ? j % 4 : 4, dash: !wTrue[j] })), marks: [{ x: Math.log10(q.lam), label: "seçili λ" }, { y: 0, label: "" }] },
        { title: "Eğitim ve doğrulama hatası (400 ayrı örnek)", xlabel: "log₁₀ λ", ylabel: "MSE", series: [
          { name: "eğitim", x: lx, y: tr, c: 0 }, { name: "doğrulama", x: lx, y: va, c: 2 },
        ], marks: [{ x: lx[best], label: "en iyi λ" }, { y: q.noise ** 2, label: "σ² (taban)" }] },
      ],
      stats: [
        { label: "Seçili λ'da sıfırdan farklı katsayı", value: `${nz} / ${d}` },
        { label: "Doğrulamada en iyi λ", value: lams[best] },
        { label: "En iyi doğrulama MSE", value: va[best] },
        { label: "Cezasız (en küçük kareler) doğrulama MSE", value: ols ? mseOn(Xva, yva, ols, bOf(ols)) : "n ≤ d: tek çözüm yok" },
        { label: "Gürültü tabanı σ²", value: q.noise ** 2 },
      ],
      note: q.method === 1
        ? "Lasso yolunda katsayılar λ büyüdükçe <b>teker teker tam sıfıra</b> iner (soft-threshold); yalnız en güçlü öznitelikler kalır — otomatik öznitelik seçimi. Öznitelikler çok ilişkiliyse (ρ büyük) lasso grubu içinden keyfi birini seçip diğerlerini sıfırlayabilir; tohumu değiştirince seçim değişir."
        : "Ridge yolunda katsayılar λ büyüdükçe <b>düzgünce</b> sıfıra yaklaşır ama tam sıfır olmaz; ilişkili öznitelikler ağırlığı paylaşır. Ridge en çok, verinin az değiştiği (küçük tekil değerli) yönleri bastırır: koşul sayısını iyileştirir ve varyansı düşürür.",
    };
  },
};
