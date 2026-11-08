// M2.5 — Gradyan artırma (1B regresyon, stump ya da sığ ağaç): her tur sözde artıklara bir ağaç oturtur; öğrenme oranı (shrinkage), kayıp seçimi, aşırı uyum.
import { fmt, linspace, range } from "./plot.js";

const truth = (x) => Math.sin(2 * x) + 0.4 * x;
export function data(n, noise, out, r) { const x = [], y = []; for (let i = 0; i < n; i++) { const xi = 6 * r() - 3; x.push(xi); y.push(truth(xi) + noise * r.normal() + (r() < out ? 4 + 2 * r() : 0)); } return { x, y }; }
// 1B regresyon ağacı (SSE ile bölünme), yapraklar: ortalama
export function regTree(x, r, idx, depth, minLeaf = 3) {
  const n = idx.length, mean = idx.reduce((s, i) => s + r[i], 0) / n;
  if (depth === 0 || n < 2 * minLeaf) return { v: mean };
  const s = [...idx].sort((a, b) => x[a] - x[b]), tot = n * mean; let SL = 0, best = null;
  for (let k = 0; k < n - 1; k++) { SL += r[s[k]]; const nl = k + 1, nr = n - nl; if (nl < minLeaf || nr < minLeaf || x[s[k]] === x[s[k + 1]]) continue; const g = (SL * SL) / nl + ((tot - SL) ** 2) / nr; if (!best || g > best.g) best = { g, t: (x[s[k]] + x[s[k + 1]]) / 2 }; }
  if (!best) return { v: mean };
  return { t: best.t, L: regTree(x, r, idx.filter((i) => x[i] <= best.t), depth - 1, minLeaf), R: regTree(x, r, idx.filter((i) => x[i] > best.t), depth - 1, minLeaf) };
}
export const evalT = (T, x) => (T.v !== undefined ? T.v : evalT(x <= T.t ? T.L : T.R, x));
const LOSS = [
  { name: "MSE", L: (y, f) => (y - f) ** 2 / 2, g: (y, f) => y - f, init: (y) => y.reduce((a, b) => a + b, 0) / y.length },
  { name: "mutlak (MAE)", L: (y, f) => Math.abs(y - f), g: (y, f) => Math.sign(y - f), init: (y) => { const s = [...y].sort((a, b) => a - b); return s[s.length >> 1]; } },
  { name: "Huber (δ = 1)", L: (y, f) => (Math.abs(y - f) <= 1 ? (y - f) ** 2 / 2 : Math.abs(y - f) - 0.5), g: (y, f) => Math.max(-1, Math.min(1, y - f)), init: (y) => { const s = [...y].sort((a, b) => a - b); return s[s.length >> 1]; } },
];

export default {
  params: [
    { id: "M", label: "Tur sayısı M", min: 1, max: 500, value: 100, int: true, log: true },
    { id: "eta", label: "Öğrenme oranı (shrinkage) η", min: 0.01, max: 1, value: 0.2, log: true, hardMin: 1e-6 },
    { id: "depth", label: "Her ağacın derinliği", min: 1, max: 5, value: 1, int: true, help: "1 = stump (tek bölünme). Derinlik, ağaçların kaçlı etkileşim kurabileceğini belirler." },
    { id: "loss", label: "Kayıp", options: LOSS.map((l) => l.name), value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 1, step: 0.02, value: 0.3 },
    { id: "out", label: "Aykırı değer oranı", min: 0, max: 0.2, step: 0.01, value: 0 },
  ],
  seed: true,
  run(q, r) {
    const { x, y } = data(80, q.noise, q.out, r), V = data(500, q.noise, 0, r), Lo = LOSS[q.loss], M = Math.round(q.M), D = Math.round(q.depth);
    const F0 = Lo.init(y); let F = x.map(() => F0); const trees = [], tr = [], te = []; let lastRes = null;
    const predict = (xx, m) => trees.slice(0, m).reduce((s, T) => s + q.eta * evalT(T, xx), F0);
    const mse = (S, m) => S.x.reduce((s, xx, i) => s + (predict(xx, m) - S.y[i]) ** 2, 0) / S.x.length;
    for (let m = 0; m < M; m++) {
      const res = y.map((yi, i) => Lo.g(yi, F[i])); // sözde artık = −∂L/∂F
      const T = regTree(x, res, range(x.length), D); trees.push(T); lastRes = res;
      F = F.map((f, i) => f + q.eta * evalT(T, x[i]));
    }
    const ms = [...new Set([0, 1, 2, 5, 10, 20, 50, 100, 200, 500].filter((m) => m <= M).concat([M]))].sort((a, b) => a - b);
    for (const m of ms) { tr.push(mse({ x, y }, m)); te.push(mse(V, m)); }
    const gx = linspace(-3, 3, 241);
    const show = [...new Set([1, Math.min(5, M), M])];
    return {
      plots: [
        { title: `Veri, gerçek fonksiyon ve F_m (m = ${show.join(", ")})`, xlabel: "x", ylabel: "y", series: [
          { name: "veri", x, y, kind: "scatter", c: 0, r: 2.6 },
          { name: "gerçek f(x)", x: gx, y: gx.map(truth), c: 4, dash: true },
          ...show.map((m, k) => ({ name: `F_${m}`, x: gx, y: gx.map((v) => predict(v, m)), c: k === show.length - 1 ? 2 : k === 0 ? 5 : 3, dash: k !== show.length - 1 })),
        ] },
        { title: `Son turun sözde artıkları ve ona oturtulan ağaç h_${M}`, xlabel: "x", ylabel: "−∂L/∂F", series: [
          { name: "sözde artık rᵢ", x, y: lastRes, kind: "scatter", c: 1, r: 2.6 },
          { name: `h_${M}(x)`, x: gx, y: gx.map((v) => evalT(trees[M - 1], v)), c: 2, kind: "step" },
        ], marks: [{ y: 0, label: "" }] },
        { title: "Kare hata, tur sayısına göre", xlabel: "tur m", ylabel: "MSE", ylog: true, series: [
          { name: "eğitim", x: ms, y: tr, c: 0, dots: true }, { name: "test (aykırısız)", x: ms, y: te, c: 2, dots: true },
        ], marks: [{ y: Math.max(q.noise ** 2, 1e-4), label: "σ²" }] },
      ],
      stats: [
        { label: "F₀ (başlangıç sabiti)", value: F0 },
        { label: `Eğitim / test MSE (m = ${M})`, value: `${fmt(tr.at(-1), 3)} / ${fmt(te.at(-1), 3)}` },
        { label: "Testte en iyi tur (denenenler)", value: ms[te.indexOf(Math.min(...te))] },
      ],
      note: `Her tur, mevcut tahminin kayba göre negatif gradyanını (sözde artık) yeni bir küçük ağaçla yaklaştırır ve η kadar ekler: <b>fonksiyon uzayında gradyan inişi</b>. MSE'de sözde artık sıradan artıktır (y − F); MAE'de yalnız işaretidir (±1) — aykırı değerin büyüklüğü artık önemli değildir; Huber'de ±1'e kırpılmış artıktır. ${q.eta >= 0.5 ? "η büyük: her ağaç hatanın büyük kısmını hemen düzeltir, hızlı ama gürültüye de hızla uyar. " : "Küçük η (shrinkage) her ağacın katkısını kısar: aynı doğruluk için daha çok tur gerekir ama genelleme genelde daha iyidir (Friedman 2001). "}Tur sayısı da bir düzenlileştiricidir: test hatası bir noktadan sonra yükselir, erken durdurma ile seçilir.`,
    };
  },
};
