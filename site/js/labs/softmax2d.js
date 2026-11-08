// M1.5 — 3 sınıflı softmax regresyonu, 2B: karar bölgeleri, güven haritası ve gradyan P − Y ile eğitim.
import { fmt, range } from "./plot.js";

export function softmax(z) { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }
export function lse(z) { const m = Math.max(...z); return m + Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0)); }
const C = [[-1.6, -1], [1.6, -1], [0, 1.7]];
export function train(X, Y, q, K = 3) {
  let W = Array.from({ length: K }, () => [0, 0]), b = Array(K).fill(0); const L = [], n = X.length;
  for (let t = 0; t <= q.steps; t++) {
    const gW = W.map(() => [0, 0]), gb = Array(K).fill(0); let loss = 0;
    for (let i = 0; i < n; i++) {
      const z = W.map((w, k) => w[0] * X[i][0] + w[1] * X[i][1] + b[k]);
      loss += lse(z) - z[Y[i]];
      const p = softmax(z);
      for (let k = 0; k < K; k++) { const d = p[k] - (k === Y[i] ? 1 : 0); gW[k][0] += d * X[i][0]; gW[k][1] += d * X[i][1]; gb[k] += d; }
    }
    L.push(loss / n + (q.lam / 2) * W.reduce((s, w) => s + w[0] ** 2 + w[1] ** 2, 0));
    if (t === q.steps) break;
    W = W.map((w, k) => [w[0] - q.lr * (gW[k][0] / n + q.lam * w[0]), w[1] - q.lr * (gW[k][1] / n + q.lam * w[1])]); b = b.map((v, k) => v - (q.lr * gb[k]) / n);
  }
  return { W, b, L };
}

export default {
  params: [
    { id: "spread", label: "Sınıf içi yayılım σ", min: 0.2, max: 2, step: 0.05, value: 0.7 },
    { id: "n", label: "Sınıf başına örnek", min: 5, max: 100, value: 30, int: true },
    { id: "lam", label: "L2 cezası λ", min: 0, max: 0.5, step: 0.005, value: 0.01 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 5, value: 0.5, log: true, hardMin: 1e-6 },
    { id: "steps", label: "Adım sayısı", min: 1, max: 1000, value: 200, int: true, log: true },
    { id: "view", label: "Arka plan", options: ["en büyük olasılık (güven)", "P(sınıf 3 | x)"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const X = [], Y = [], nc = Math.round(q.n);
    for (let k = 0; k < 3; k++) for (let i = 0; i < nc; i++) { X.push([C[k][0] + q.spread * r.normal(), C[k][1] + q.spread * r.normal()]); Y.push(k); }
    const T = train(X, Y, { ...q, steps: Math.round(q.steps) }), { W, b } = T;
    const R = 4, N = 56, z = [], bx = [], by = [];
    const prob = (x0, x1) => softmax(W.map((w, k) => w[0] * x0 + w[1] * x1 + b[k]));
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const p = prob(-R + ((i + 0.5) * 2 * R) / N, -R + ((j + 0.5) * 2 * R) / N); z.push(q.view === 0 ? Math.max(...p) : p[2]); }
    // karar sınırları: ince ızgarada argmax değişen hücreler
    const M = 160, am = (x0, x1) => { const p = prob(x0, x1); return p.indexOf(Math.max(...p)); };
    for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) { const x0 = -R + (i * 2 * R) / (M - 1), x1 = -R + (j * 2 * R) / (M - 1); if (i < M - 1 && am(x0, x1) !== am(x0 + (2 * R) / (M - 1), x1)) { bx.push(x0); by.push(x1); } }
    const acc = X.reduce((s, x, i) => s + (am(...x) === Y[i] ? 1 : 0), 0) / X.length;
    const shift = W.map((w) => [w[0] + 5, w[1] - 3]); // tüm logitlere aynı fonksiyonu eklemek
    return {
      plots: [
        { title: q.view === 0 ? "Güven max P(k | x) (koyu = düşük güven) ve karar sınırları" : "P(sınıf 3 | x) (koyu = düşük)", xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-R, R], ylim: [-R, R], field: { x0: -R, x1: R, y0: -R, y1: R, nx: N, ny: N, z },
          series: [
            ...[0, 1, 2].map((k) => ({ name: `sınıf ${k + 1}`, x: X.filter((_, i) => Y[i] === k).map((v) => v[0]), y: X.filter((_, i) => Y[i] === k).map((v) => v[1]), kind: "scatter", c: [0, 2, 1][k], r: 3 })),
            { name: "karar sınırı (argmax değişir)", x: bx, y: by, kind: "scatter", c: 3, r: 0.9 },
          ] },
        { title: "Eğitim kaybı (ortalama çapraz entropi)", xlabel: "adım", ylabel: "kayıp", ylog: true, series: [{ name: "kayıp", x: range(T.L.length), y: T.L, c: 0 }], marks: [{ y: Math.log(3), label: "ln 3 (rastgele tahmin)" }] },
      ],
      stats: [
        { label: "Eğitim doğruluğu", value: `${fmt(acc * 100, 3)} %` },
        { label: "Son kayıp", value: T.L.at(-1) },
        { label: "Başlangıç kaybı (W = 0): ln K", value: Math.log(3) },
        { label: "Ağırlık satırları w₁, w₂, w₃", value: W.map((w) => `(${fmt(w[0], 2)}, ${fmt(w[1], 2)})`).join(" ") },
        { label: "Tüm wₖ'ya aynı vektörü eklesen olasılık değişir mi?", value: Math.abs(softmax(shift.map((w, k) => w[0] + w[1] + b[k]))[0] - softmax(W.map((w, k) => w[0] + w[1] + b[k]))[0]) < 1e-12 ? "hayır (kaydırma değişmezliği)" : "evet" },
      ],
      note: "Her sınıfın bir logiti zₖ = wₖ·x + bₖ vardır; iki sınıf arasındaki sınır zₖ = zⱼ, yani (wₖ − wⱼ)·x + (bₖ − bⱼ) = 0 — bir doğru. Bölgeler bu yüzden dışbükey çokgenlerdir. Softmax yalnız logit <b>farklarına</b> bakar: tüm logitlere aynı sayıyı (hatta aynı doğrusal fonksiyonu) eklemek olasılıkları değiştirmez; bu yüzden parametreler tek değil, L2 cezası (λ > 0) birini seçer. Kayıp ln 3'ten başlar (W = 0'da her sınıf 1/3). Gradyan ∂ℓ/∂z = p − onehot(y): her örnek kendi sınıfının logitini yukarı, ona olasılık çalan diğerlerini olasılıklarıyla orantılı aşağı iter.",
    };
  },
};
