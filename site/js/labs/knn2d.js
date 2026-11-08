// M2.1 — k-NN, 2B: karar sınırları k ile nasıl yumuşar; mesafe ölçüsü, ağırlıklı oy, öznitelik ölçeği; eğitim/doğrulama hatası k'ya göre.
import { fmt, range } from "./plot.js";

const CEN = [[-1.2, -0.8], [1.3, -0.6], [0, 1.3]];
export function data(n, spread, r, stretch = 1) { const X = [], Y = []; for (let i = 0; i < n; i++) { const k = i % 3; X.push([(CEN[k][0] + spread * r.normal()) * stretch, CEN[k][1] + spread * r.normal()]); Y.push(k); } return { X, Y }; }
export const DIST = [(a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]), (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]), (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]))];
// eşitlik kuralı: eşit mesafede küçük indeks önce; eşit oyda küçük sınıf
export function predict(X, Y, q, k, metric, weighted, skip = -1) {
  const d = X.map((x, j) => [j === skip ? Infinity : DIST[metric](q, x), j]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const v = [0, 0, 0];
  for (let i = 0; i < k; i++) { const [dd, j] = d[i]; v[Y[j]] += weighted ? 1 / Math.max(dd, 1e-9) : 1; }
  let best = 0; for (let c = 1; c < 3; c++) if (v[c] > v[best]) best = c;
  return best;
}

export default {
  params: [
    { id: "k", label: "Komşu sayısı k", min: 1, max: 60, value: 5, int: true },
    { id: "metric", label: "Mesafe", options: ["Öklid (L2)", "Manhattan (L1)", "Chebyshev (L∞)"], value: 0 },
    { id: "w", label: "Oy", options: ["eşit oy", "1/mesafe ağırlıklı"], value: 0 },
    { id: "spread", label: "Sınıf içi yayılım", min: 0.2, max: 1.5, step: 0.05, value: 0.7 },
    { id: "n", label: "Eğitim örneği", min: 15, max: 240, value: 90, int: true },
    { id: "stretch", label: "x₁'in ölçek çarpanı (birim değişimi)", min: 0.1, max: 10, value: 1, log: true, help: "x₁'i örneğin metreden santimetreye çevirmek gibi. Ölçeklenmemiş öznitelikte mesafe o özniteliğe hükmeder." },
  ],
  seed: true,
  validate(q) { return q.k > q.n ? [{ level: "error", msg: `Komşu sayısı k = ${q.k}, eğitim örneği sayısından (${Math.round(q.n)}) büyük olamaz.` }] : []; },
  run(q, r) {
    const n = Math.round(q.n), { X, Y } = data(n, q.spread, r, q.stretch), V = data(300, q.spread, r, q.stretch), k = Math.round(q.k);
    const sx = Math.max(1, q.stretch), Rx = 3.6 * sx, Ry = 3.6, M = 90;
    const am = (x0, x1) => predict(X, Y, [x0, x1], k, q.metric, q.w);
    const G = []; for (let j = 0; j < M; j++) { const row = []; for (let i = 0; i < M; i++) row.push(am(-Rx + (i * 2 * Rx) / (M - 1), -Ry + (j * 2 * Ry) / (M - 1))); G.push(row); }
    const bx = [], by = [];
    for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) { if ((i < M - 1 && G[j][i] !== G[j][i + 1]) || (j < M - 1 && G[j][i] !== G[j + 1][i])) { bx.push(-Rx + (i * 2 * Rx) / (M - 1)); by.push(-Ry + (j * 2 * Ry) / (M - 1)); } }
    const ks = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60].filter((v) => v <= n - 1);
    const err = (kk, train) => { let e = 0; const S = train ? X : V.X, T = train ? Y : V.Y; for (let i = 0; i < S.length; i++) if (predict(X, Y, S[i], kk, q.metric, q.w) !== T[i]) e++; return e / S.length; };
    const loo = (kk) => { let e = 0; for (let i = 0; i < n; i++) if (predict(X, Y, X[i], kk, q.metric, q.w, i) !== Y[i]) e++; return e / n; };
    const tr = ks.map((kk) => err(kk, true)), va = ks.map((kk) => err(kk, false)), lo = ks.map(loo);
    return {
      plots: [
        { title: `k = ${k}: karar sınırları (mor) ve eğitim noktaları`, xlabel: q.stretch !== 1 ? `x₁ (× ${fmt(q.stretch, 3)})` : "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-Rx, Rx], ylim: [-Ry, Ry],
          series: [
            { name: "karar sınırı", x: bx, y: by, kind: "scatter", c: 3, r: 1.1 },
            ...[0, 1, 2].map((c) => ({ name: `sınıf ${c + 1}`, x: X.filter((_, i) => Y[i] === c).map((v) => v[0]), y: X.filter((_, i) => Y[i] === c).map((v) => v[1]), kind: "scatter", c: [0, 2, 1][c], r: 3 })),
          ] },
        { title: "Hata oranı k'ya göre", xlabel: "k", ylabel: "hata", yzero: true, series: [
          { name: "eğitim (kendisi dahil)", x: ks, y: tr, c: 0, dots: true },
          { name: "birini-dışarıda-bırak (LOO)", x: ks, y: lo, c: 1, dots: true, dash: true },
          { name: "ayrı doğrulama (300)", x: ks, y: va, c: 2, dots: true },
        ], marks: [{ x: k, label: "seçili k" }] },
      ],
      stats: [
        { label: "Eğitim hatası (seçili k)", value: err(k, true) },
        { label: "Doğrulama hatası (seçili k)", value: err(k, false) },
        { label: "Doğrulamada en iyi k (denenenler)", value: ks[va.indexOf(Math.min(...va))] },
        { label: "Tahmin başına mesafe hesabı", value: n },
      ],
      note: `k = 1'de her eğitim noktası kendi komşusudur: eğitim hatası 0, sınırlar pürüzlü ve her gürültülü noktanın çevresinde "ada" oluşur (yüksek varyans). k büyüdükçe sınırlar yumuşar; k → n'de her yer çoğunluk sınıfına gider (yüksek yanlılık). Eğitim hatası bu yüzden k seçmek için kullanılamaz — LOO ya da ayrı doğrulama gerekir. ${q.stretch !== 1 ? `<b>x₁ ${fmt(q.stretch, 3)} katına ölçeklendi:</b> ${q.stretch > 1 ? "mesafeler neredeyse yalnız x₁ farkıyla belirleniyor, x₂'deki bilgi kayboluyor; sınırlar dikeyleşir" : "x₁ farkları küçüldü, sınıflandırma yalnız x₂'ye dayanıyor; sınırlar yataylaşır"}. k-NN'den önce öznitelikleri standartlaştır.` : ""}`,
    };
  },
};
