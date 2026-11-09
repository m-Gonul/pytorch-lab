// M4.9 — Çapa kutularını k-means ile bulmak: eğitim kutularının (genişlik, yükseklik) çiftleri üzerinde Öklid uzaklığı ile
// d = 1 − IoU uzaklığı (YOLOv2) karşılaştırması. Ölçüt: her kutunun en iyi çapasıyla ortalama IoU'su ("Avg IoU").
import { fmt, range } from "./plot.js";

export const iouWH = (a, b) => { const i = Math.min(a[0], b[0]) * Math.min(a[1], b[1]); return i / (a[0] * a[1] + b[0] * b[1] - i); };
export function kmeans(X, k, dist, r, iters = 60) {
  // k-means++ başlatma, seçili uzaklıkla
  const C = [X[Math.floor(r() * X.length)]];
  while (C.length < k) { const d = X.map((x) => Math.min(...C.map((c) => dist(x, c))) ** 2), s = d.reduce((a, b) => a + b, 0); let u = r() * s, i = 0; while (u > d[i] && i < X.length - 1) { u -= d[i]; i++; } C.push(X[i]); }
  let A = [];
  for (let it = 0; it < iters; it++) {
    A = X.map((x) => { let b = 0; C.forEach((c, j) => { if (dist(x, c) < dist(x, C[b])) b = j; }); return b; });
    for (let j = 0; j < k; j++) { const m = X.filter((_, i) => A[i] === j); if (m.length) C[j] = [m.reduce((s, x) => s + x[0], 0) / m.length, m.reduce((s, x) => s + x[1], 0) / m.length]; }
  }
  return { C, A };
}
// sentetik veri: küçük nesneler (çok), yayalar (uzun), arabalar (geniş), büyük nesneler (az)
export function boxes(n, mix, r) {
  const G = [[20, 20, 0.35], [40, 100, 0.25], [140, 70, 0.25], [300, 260, 0.15]];
  const w = G.map((g, i) => (i === 3 ? g[2] * mix : g[2])), tot = w.reduce((a, b) => a + b, 0);
  return range(n).map(() => { let u = r() * tot, i = 0; while (i < 3 && u >= w[i]) { u -= w[i]; i++; } const g = G[i]; return [g[0] * Math.exp(0.3 * r.normal()), g[1] * Math.exp(0.3 * r.normal())]; });
}
const cols = [0, 2, 1, 3, 5, 4];

export default {
  params: [
    { id: "k", label: "Çapa sayısı k", min: 1, max: 9, value: 5, int: true },
    { id: "d", label: "k-means uzaklığı", options: ["Öklid (w, h)", "1 − IoU (YOLOv2)"], value: 1 },
    { id: "mix", label: "Büyük nesnelerin göreli sıklığı", min: 0, max: 3, step: 0.1, value: 1 },
    { id: "n", label: "Eğitim kutusu sayısı", min: 50, max: 2000, value: 600, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const X = boxes(Math.round(q.n), q.mix, r), dist = q.d === 0 ? (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) : (a, b) => 1 - iouWH(a, b);
    const { C, A } = kmeans(X, Math.round(q.k), dist, r), best = X.map((x) => Math.max(...C.map((c) => iouWH(x, c))));
    const avg = best.reduce((a, b) => a + b, 0) / X.length, small = X.map((x, i) => [x, best[i]]).filter(([x]) => x[0] * x[1] < 32 * 32), smallAvg = small.length ? small.reduce((a, [, v]) => a + v, 0) / small.length : NaN;
    const other = q.d === 0 ? (a, b) => 1 - iouWH(a, b) : (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]), C2 = kmeans(X, Math.round(q.k), other, r).C, avg2 = X.map((x) => Math.max(...C2.map((c) => iouWH(x, c)))).reduce((a, b) => a + b, 0) / X.length;
    const ks = range(9).map((i) => i + 1);
    return {
      plots: [
        { title: "Eğitim kutuları (w, h) ve çapalar (log–log)", xlabel: "log₁₀ genişlik (piksel)", ylabel: "log₁₀ yükseklik", size: { w: 420, h: 380 }, xlim: [0.8, 3], ylim: [0.8, 3],
          series: [...C.map((c, j) => ({ name: j < 6 ? `küme ${j + 1}` : "", x: X.filter((_, i) => A[i] === j).map((x) => Math.log10(x[0])), y: X.filter((_, i) => A[i] === j).map((x) => Math.log10(x[1])), kind: "scatter", c: cols[j % 6], r: 1.8 })), { name: "çapa", x: C.map((c) => Math.log10(c[0])), y: C.map((c) => Math.log10(c[1])), kind: "scatter", c: 5, r: 7 }] },
        { title: "Kutuların en iyi çapayla IoU'su (dağılım)", xlabel: "IoU", ylabel: "kutu oranı", yzero: true, series: [{ name: "histogram", x: range(20).map((i) => 0.025 + i * 0.05), y: range(20).map((i) => best.filter((v) => v >= i * 0.05 && v < (i + 1) * 0.05 + (i === 19 ? 1e-9 : 0)).length / X.length), kind: "bar", c: 0, bw: 0.9 }], marks: [{ x: avg, label: `ortalama ${fmt(avg, 3)}` }] },
      ],
      stats: [
        { label: "Ortalama IoU (en iyi çapa)", value: avg },
        { label: `Aynı k, ${q.d === 0 ? "1 − IoU" : "Öklid"} uzaklığıyla`, value: avg2 },
        { label: "Küçük kutular (alan < 32²) için ortalama IoU", value: smallAvg },
        { label: "Çapalar (w × h)", value: C.map((c) => `${Math.round(c[0])}×${Math.round(c[1])}`).join(", ") },
      ],
      note: `Ağ her çapa için ölçek faktörlerinin logaritmasını tahmin ettiğinden (b_w = p_w·e^(t_w)), iyi çapalar küçük düzeltmeler demektir. ${q.d === 0 ? "<b>Öklid uzaklığı</b> büyük kutularda büyük hatalar üretir: kümeleri büyük kutulara doğru çeker, çok sayıdaki küçük kutu kötü temsil edilir (küçük kutuların ortalama IoU'suna bak)." : "<b>1 − IoU uzaklığı</b> ölçekten bağımsızdır: 10 piksellik fark küçük kutuda büyük, büyük kutuda küçük sayılır. Redmon & Farhadi (YOLOv2) bu yüzden onu seçti; k = 5 ile ortalama IoU'nun elle seçilmiş 9 çapaya yakın olduğunu bildirdiler."} k büyüdükçe ortalama IoU artar ama model karmaşıklaşır (her hücrede k kutu).`,
    };
  },
};
