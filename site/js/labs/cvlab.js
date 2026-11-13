// M1.3 — k-fold çapraz doğrulama ile λ seçimi: her fold içinde merkezleme (sızıntı yok),
// Gram matrisinin Jacobi özdeğer ayrışımıyla tüm λ'lar için ridge yolu tek seferde.
// Sağdaki bant: fold'lar arası değişkenlik; 1-SE kuralı en iyinin bir standart hatası içindeki en basit λ'yı seçer.
import { fmt, linspace } from "./plot.js";

// Simetrik matrisin Jacobi özdeğer ayrışımı: A = V diag(mu) Vᵀ
export function jacobi(A0) {
  const d = A0.length, A = A0.map((r) => [...r]), V = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += A[p][q] ** 2;
    if (off < 1e-16) break;
    for (let p = 0; p < d - 1; p++) for (let q = p + 1; q < d; q++) {
      if (Math.abs(A[p][q]) < 1e-15) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.hypot(1, th)), c = 1 / Math.hypot(1, t), s = t * c;
      for (let i = 0; i < d; i++) { const aip = A[i][p], aiq = A[i][q]; A[i][p] = c * aip - s * aiq; A[i][q] = s * aip + c * aiq; }
      for (let i = 0; i < d; i++) { const api = A[p][i], aqi = A[q][i]; A[p][i] = c * api - s * aqi; A[q][i] = s * api + c * aqi; }
      for (let i = 0; i < d; i++) { const vip = V[i][p], viq = V[i][q]; V[i][p] = c * vip - s * viq; V[i][q] = s * vip + c * viq; }
    }
  }
  return { V, mu: A.map((r, i) => r[i]) };
}
// ridge yolu Gram hilesiyle: w(λ) = (A+λI)⁻¹c = V diag(1/(μⱼ+λ)) Vᵀc — tüm λ'lar tek özdeğer ayrışımından
export function pathFrom(V, mu, c, lams) {
  const d = mu.length;
  const Vtc = Array.from({ length: d }, (_, j) => V.reduce((a, row, i) => a + row[j] * c[i], 0));
  return lams.map((l) => {
    const coef = Vtc.map((v, j) => v / (mu[j] + l));
    return V.map((row) => row.reduce((a, v, j) => a + v * coef[j], 0));
  });
}
export function makeData(q, r, n) {
  const d = Math.round(q.d), k = Math.min(Math.round(q.ktrue), d), X = [], y = [];
  const wTrue = Array.from({ length: d }, (_, j) => (j < k ? (j % 2 ? -1 : 1) * (1.5 - j * 0.2) : 0));
  for (let i = 0; i < n; i++) {
    const z = r.normal(), row = [];
    for (let j = 0; j < d; j++) row.push(Math.sqrt(q.rho) * z + Math.sqrt(1 - q.rho) * r.normal());
    X.push(row); y.push(row.reduce((s, v, j) => s + v * wTrue[j], 0) + q.noise * r.normal());
  }
  return { X, y, wTrue, k };
}

export default {
  params: [
    { id: "k", label: "Fold sayısı k", min: 2, max: 10, step: 1, value: 5, int: true },
    { id: "n", label: "Örnek sayısı n", min: 20, max: 150, step: 5, value: 60, int: true },
    { id: "d", label: "Öznitelik sayısı d", min: 2, max: 10, step: 1, value: 6, int: true },
    { id: "ktrue", label: "Gerçekten etkili öznitelik", min: 1, max: 10, step: 1, value: 2, int: true },
    { id: "rho", label: "Öznitelik korelasyonu ρ", min: 0, max: 0.9, step: 0.05, value: 0.3 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 2, step: 0.1, value: 1 },
  ],
  seed: true,
  validate(q) { return q.ktrue > q.d ? [{ level: "warn", msg: `Etkili öznitelik sayısı (${q.ktrue}) d'den (${q.d}) büyük: k = d alındı.` }] : []; },
  run(q, r) {
    const n = Math.round(q.n), kk = Math.round(q.k), d = Math.round(q.d);
    const { X, y, wTrue } = makeData(q, r, n);
    const lams = linspace(-3, 1, 33).map((e) => 10 ** e).reverse();
    const mse = Array.from({ length: lams.length }, () => Array(kk).fill(0));
    for (let f = 0; f < kk; f++) {
      const tr = [], va = [];
      for (let i = 0; i < n; i++) (i % kk === f ? va : tr).push(i); // i. fold: idx[i::k], ön karıştırma yok
      // merkezleme yalnız eğitim kısmından — ön işleme sızıntısı yok (uyarı kutusu)
      const dcol = X[0].length, mx = Array(dcol).fill(0); let my = 0;
      for (const i of tr) { my += y[i] / tr.length; for (let j = 0; j < dcol; j++) mx[j] += X[i][j] / tr.length; }
      const Xc = tr.map((i) => X[i].map((v, j) => v - mx[j])), yc = tr.map((i) => y[i] - my);
      const A = Array.from({ length: dcol }, (_, a) => Array.from({ length: dcol }, (_, b) => Xc.reduce((s, row) => s + row[a] * row[b], 0) / tr.length));
      const c = Array.from({ length: dcol }, (_, a) => Xc.reduce((s, row, i) => s + row[a] * yc[i], 0) / tr.length);
      const { V, mu } = jacobi(A);
      const Ws = pathFrom(V, mu, c, lams);
      for (let li = 0; li < lams.length; li++) {
        const w = Ws[li], b = my - w.reduce((s, v, j) => s + v * mx[j], 0);
        mse[li][f] = va.reduce((s, i) => s + (X[i].reduce((a, v, j) => a + v * w[j], b) - y[i]) ** 2, 0) / va.length;
      }
    }
    const mean = mse.map((m) => m.reduce((a, v) => a + v, 0) / kk);
    const sd = mse.map((m, i) => Math.sqrt(m.reduce((a, v) => a + (v - mean[i]) ** 2, 0) / (kk - 1) || 0));
    const best = mean.indexOf(Math.min(...mean));
    const se = sd[best] / Math.sqrt(kk), thr = mean[best] + se;
    let seIdx = best;
    for (let i = 0; i < lams.length; i++) if (mean[i] <= thr) { seIdx = i; break; } // λ büyükten küçüğe: ilk uygun = en basit
    const lx = lams.map(Math.log10);
    return {
      plots: [
        {
          title: `k = ${kk} fold CV: ortalama hata ± 1 standart sapma`,
          xlabel: "log₁₀ λ", ylabel: "doğrulama MSE",
          series: [
            { name: "ortalama", x: lx, y: mean, c: 0 },
            { name: "+1 std", x: lx, y: mean.map((m, i) => m + sd[i]), c: 4, dash: true },
            { name: "−1 std", x: lx, y: mean.map((m, i) => m - sd[i]), c: 4, dash: true },
          ],
          marks: [
            { x: lx[best], label: "min kuralı" },
            { x: lx[seIdx], label: "1-SE kuralı" },
            { y: q.noise ** 2, label: "σ² taban" },
          ],
        },
        {
          title: "Fold'lar ayrı ayrı (kalabalık = değişkenlik)",
          xlabel: "log₁₀ λ", ylabel: "fold MSE",
          series: Array.from({ length: kk }, (_, f) => ({ name: `fold ${f + 1}`, x: lx, y: mse.map((m) => m[f]), c: f % 5, dash: f % 2 === 1 })),
        },
      ],
      stats: [
        { label: "En iyi λ (min kuralı)", value: lams[best] },
        { label: "1-SE kuralının λ'sı (daha basit)", value: lams[seIdx] },
        { label: "En iyi CV MSE", value: mean[best] },
        { label: "1-SE eşiği (min + 1 std hatası)", value: thr },
        { label: "Gürültü tabanı σ²", value: q.noise ** 2 },
      ],
      note: `Her fold kendi merkezlemesini eğitim kısmından öğrendi — ön işleme sızıntısı yok. En küçük ortalama hata λ = ${fmt(lams[best], 3)}'ta; ama 1-SE kuralı, en iyinin bir standart hatası içinde kalan <b>en büyük</b> λ'yı (en basit modeli) seçer: λ = ${fmt(lams[seIdx], 3)}. İki seçim benzer hata verir; 1-SE daha az katsayıyla daha kararlı bir model demektir. Fold çizgilerinin saçılması, λ seçiminin kaçınılmaz belirsizliğidir — bu yüzden tek bir doğrulama kümesine güvenilmez.`,
    };
  },
};
