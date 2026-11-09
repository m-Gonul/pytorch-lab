// M3.7 — Az veride aşırı uyum ve üç çare: inverted dropout, ayrık weight decay (AdamW), early stopping; ayrıca MC dropout belirsizlik bandı.
// 1 → H → 1 tanh ağı, tam-batch Adam. Doğrulama kaybı her birkaç adımda ölçülür; en iyi adımdaki ağ ayrıca saklanır (early stopping).
import { fmt, linspace, range } from "./plot.js";

const truth = (x) => Math.sin(3 * x) + 0.3 * x;
export function train(q, r) {
  const H = Math.round(q.H), n = Math.round(q.n), T = Math.round(q.T), p = q.p, lam = q.wd, lr = 0.01;
  const X = range(n).map(() => 2 * r() - 1), Y = X.map((x) => truth(x) + q.noise * r.normal());
  const Xv = linspace(-1, 1, 120), Yv = Xv.map((x) => truth(x) + q.noise * r.normal());
  // parametreler: a = w1 (H), b = b1 (H), c = w2 (H), d = b2
  const P = { a: range(H).map(() => 4 * r.normal()), b: range(H).map(() => 3 * r.normal()), c: range(H).map(() => r.normal() / Math.sqrt(H)), d: [0] };
  const M = {}, V = {}; for (const k in P) { M[k] = P[k].map(() => 0); V[k] = P[k].map(() => 0); }
  const fwd = (Q, x, mask) => { let y = Q.d[0]; for (let j = 0; j < H; j++) { if (mask && !mask[j]) continue; y += Q.c[j] * Math.tanh(Q.a[j] * x + Q.b[j]) * (mask ? 1 / (1 - p) : 1); } return y; };
  const mse = (Q, xs, ys) => xs.reduce((s, x, i) => s + (fwd(Q, x) - ys[i]) ** 2, 0) / xs.length;
  const copy = (Q) => ({ a: [...Q.a], b: [...Q.b], c: [...Q.c], d: [...Q.d] });
  const every = Math.max(1, Math.round(T / 150)), steps = [], tr = [], va = [];
  let best = { v: Infinity, t: 0, P: copy(P) };
  for (let t = 1; t <= T; t++) {
    const G = { a: P.a.map(() => 0), b: P.b.map(() => 0), c: P.c.map(() => 0), d: [0] };
    for (let i = 0; i < n; i++) {
      const mask = p > 0 ? range(H).map(() => r() >= p) : null, s = mask ? 1 / (1 - p) : 1;
      const h = P.a.map((a, j) => Math.tanh(a * X[i] + P.b[j]));
      let y = P.d[0]; for (let j = 0; j < H; j++) if (!mask || mask[j]) y += P.c[j] * h[j] * s;
      const e = (2 * (y - Y[i])) / n; G.d[0] += e;
      for (let j = 0; j < H; j++) { if (mask && !mask[j]) continue; G.c[j] += e * h[j] * s; const dz = e * P.c[j] * s * (1 - h[j] * h[j]); G.a[j] += dz * X[i]; G.b[j] += dz; }
    }
    for (const k in P) for (let j = 0; j < P[k].length; j++) {
      M[k][j] = 0.9 * M[k][j] + 0.1 * G[k][j]; V[k][j] = 0.999 * V[k][j] + 0.001 * G[k][j] ** 2;
      const mh = M[k][j] / (1 - 0.9 ** t), vh = V[k][j] / (1 - 0.999 ** t);
      P[k][j] -= lr * (mh / (Math.sqrt(vh) + 1e-8) + (k === "a" || k === "c" ? lam * P[k][j] : 0));
    }
    if (t % every === 0 || t === T) { const v = mse(P, Xv, Yv); steps.push(t); tr.push(mse(P, X, Y)); va.push(v); if (v < best.v) best = { v, t, P: copy(P) }; }
  }
  return { X, Y, Xv, Yv, P, best, steps, tr, va, fwd, H };
}

export default {
  params: [
    { id: "p", label: "Dropout oranı p (gizli katman)", min: 0, max: 0.8, step: 0.05, value: 0, hardMax: 0.95, why: "p → 1'de neredeyse tüm nöronlar düşer; 1/(1 − p) ölçeği patlar." },
    { id: "wd", label: "Weight decay λ (ayrık, AdamW biçimi)", min: 0, max: 0.3, step: 0.005, value: 0, help: "Her adımda w ← w − η·λ·w (yalnız ağırlıklar, bias'lar hariç)." },
    { id: "T", label: "Eğitim adımı", min: 200, max: 4000, value: 2500, int: true, log: true },
    { id: "n", label: "Eğitim örneği sayısı", min: 6, max: 80, value: 12, int: true, log: true },
    { id: "noise", label: "Etiket gürültüsü σ", min: 0, max: 0.6, step: 0.02, value: 0.25 },
    { id: "H", label: "Gizli nöron sayısı H", min: 8, max: 128, value: 64, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const R = train(q, r), xs = linspace(-1.05, 1.05, 160), p = q.p;
    // MC dropout: 40 rastgele maskeyle tahmin, ortalama ± 2 std
    let band = null, mcStd = 0;
    if (p > 0) {
      const S = 40, mk = () => range(R.H).map(() => r() >= p), preds = range(S).map(() => { const m = mk(); return xs.map((x) => R.fwd(R.P, x, m)); });
      const mu = xs.map((_, i) => preds.reduce((s, v) => s + v[i], 0) / S), sd = xs.map((_, i) => Math.sqrt(preds.reduce((s, v) => s + (v[i] - mu[i]) ** 2, 0) / (S - 1)));
      band = { mu, sd }; mcStd = sd.reduce((a, b) => a + b, 0) / sd.length;
    }
    const finalV = R.va[R.va.length - 1], wn = Math.sqrt(R.P.a.reduce((s, v) => s + v * v, 0) + R.P.c.reduce((s, v) => s + v * v, 0));
    const series = [
      { name: "gerçek f(x)", x: xs, y: xs.map(truth), c: 5, dash: true },
      { name: "son ağ (eval)", x: xs, y: xs.map((x) => R.fwd(R.P, x)), c: 0 },
      { name: `early stopping (adım ${R.best.t})`, x: xs, y: xs.map((x) => R.fwd(R.best.P, x)), c: 2 },
      { name: "eğitim örnekleri", x: R.X, y: R.Y, kind: "scatter", c: 1, r: 4 },
    ];
    if (band) series.push({ name: "MC dropout ort. ± 2 std", x: xs, y: band.mu.map((m, i) => m + 2 * band.sd[i]), c: 3 }, { name: "", x: xs, y: band.mu.map((m, i) => m - 2 * band.sd[i]), c: 3 });
    return {
      plots: [
        { title: "Uyum: eğitim noktaları, gerçek fonksiyon ve öğrenilen ağlar", xlabel: "x", ylabel: "y", ylim: [-2.6, 2.6], series },
        { title: "Kayıp eğrileri (eval modunda ölçülür)", xlabel: "adım", ylabel: "MSE", ylog: true, series: [{ name: "eğitim", x: R.steps, y: R.tr.map((v) => Math.max(v, 1e-8)), c: 0 }, { name: "doğrulama", x: R.steps, y: R.va.map((v) => Math.max(v, 1e-8)), c: 2 }], marks: [{ x: R.best.t, label: "en iyi doğrulama" }, ...(q.noise > 0 ? [{ y: q.noise * q.noise, label: "gürültü tabanı σ²" }] : [])] },
      ],
      stats: [
        { label: "Son eğitim MSE", value: R.tr[R.tr.length - 1] },
        { label: "Son doğrulama MSE", value: finalV },
        { label: "Early stopping: en iyi doğrulama MSE (adım)", value: `${fmt(R.best.v, 3)} (${R.best.t})` },
        { label: "Ağırlık normu ‖w‖ (son)", value: wn },
        { label: "MC dropout ortalama std", value: p > 0 ? mcStd : "— (p = 0)" },
      ],
      note: `Doğrulama hatasının ulaşabileceği taban gürültü varyansıdır (σ² = ${fmt(q.noise * q.noise, 3)}): bunun altındaki eğitim hatası ezberdir. ${R.tr[R.tr.length - 1] < 0.5 * q.noise * q.noise && finalV > 1.3 * R.best.v ? "<b>Aşırı uyum:</b> eğitim hatası gürültü tabanının çok altına inmiş, doğrulama hatası en iyi noktasından yükselmiş — early stopping (turuncu) son ağdan (mavi) daha iyi." : finalV <= 1.05 * R.best.v ? "Doğrulama hatası eğitimin sonunda hâlâ en iyisine yakın: bu ayarlarda early stopping'e gerek kalmıyor." : "Doğrulama hatası en iyi noktasından biraz yükselmiş."} ${p > 0 ? `Dropout eğitimde her adımda nöronların %${fmt(p * 100, 3)}'ini düşürür ve kalanları 1/(1 − p) = ${fmt(1 / (1 - p), 3)} ile ölçekler; eval'de tüm ağ kullanılır. Mor çizgiler aynı eğitilmiş ağın 40 rastgele dropout maskesiyle tahminlerinin ortalama ± 2 std'sidir (MC dropout): veri olmayan bölgelerde bandın genişlemesi beklenir, ama bu bir model belirsizliği <b>yaklaşımıdır</b>, kalibre bir güven aralığı değil.` : ""} ${q.wd > 0 ? `Weight decay ‖w‖'yi küçültür: daha düzgün (daha az salınan) fonksiyonlar.` : ""}`,
    };
  },
};
