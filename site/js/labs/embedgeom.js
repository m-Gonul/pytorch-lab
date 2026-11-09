// M4.5 — Kategorik bir özniteliğin 2-B embedding'i eğitimle nasıl düzenlenir? California konut verisindeki ocean_proximity benzeri
// 5 kategori; hedef iki sayı (fiyat, kıyıya yakınlık). Model ŷ = W·e_c + b; e (embedding) ve W, SGD ile birlikte öğrenilir.
import { fmt, range } from "./plot.js";

export const CATS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"];
const MU = [[2.4, 0.6], [1.2, 0.1], [2.5, 1.0], [2.6, 0.9], [3.8, 1.0]];
export function train(q, r) {
  const freq = [0.44, 0.32, 0.13, 0.11, q.island], tot = freq.reduce((a, b) => a + b, 0), N = 2000;
  const cum = []; freq.reduce((a, f, i) => (cum[i] = a + f / tot), 0);
  const C = range(N).map(() => { const u = r(); return cum.findIndex((c) => u <= c); }), Y = C.map((c) => MU[c].map((m) => m + q.noise * r.normal()));
  const E0 = CATS.map(() => [q.init * r.normal(), q.init * r.normal()]), E = E0.map((e) => [...e]);
  const W = [[r.normal() * 0.5, r.normal() * 0.5], [r.normal() * 0.5, r.normal() * 0.5]], b = [2, 0.5], loss = [], B = 32, steps = Math.round(q.epochs * (N / B));
  const path = CATS.map(() => []);
  for (let t = 0; t < steps; t++) {
    const gE = CATS.map(() => [0, 0]), gW = [[0, 0], [0, 0]], gb = [0, 0]; let L = 0;
    for (let k = 0; k < B; k++) {
      const i = Math.floor(r() * N), c = C[i], e = E[c], yh = [0, 1].map((o) => W[o][0] * e[0] + W[o][1] * e[1] + b[o]), d = yh.map((v, o) => v - Y[i][o]);
      L += (d[0] ** 2 + d[1] ** 2) / B;
      for (let o = 0; o < 2; o++) { gb[o] += d[o] / B; for (let j = 0; j < 2; j++) { gW[o][j] += (d[o] * e[j]) / B; gE[c][j] += (d[o] * W[o][j]) / B; } }
    }
    for (let o = 0; o < 2; o++) { b[o] -= q.lr * gb[o]; for (let j = 0; j < 2; j++) W[o][j] -= q.lr * gW[o][j]; }
    // embedding gradyanı yalnız batch'te görülen kategorilerin satırlarına düşer (scatter-add)
    CATS.forEach((_, c) => { E[c][0] -= q.lr * gE[c][0]; E[c][1] -= q.lr * gE[c][1]; });
    if (t % Math.max(1, Math.floor(steps / 120)) === 0) { loss.push([t / (N / B), L]); CATS.forEach((_, c) => path[c].push([...E[c]])); }
  }
  return { E0, E, W, b, loss, path, freq: freq.map((f) => f / tot) };
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export default {
  params: [
    { id: "epochs", label: "Epoch", min: 0.05, max: 30, value: 8, log: true },
    { id: "lr", label: "Öğrenme oranı", min: 0.005, max: 0.5, value: 0.05, log: true },
    { id: "init", label: "Başlangıç ölçeği (rastgele embedding std'si)", min: 0.05, max: 2, value: 0.5, log: true },
    { id: "noise", label: "Hedef gürültüsü σ", min: 0, max: 1, step: 0.05, value: 0.3 },
    { id: "island", label: "ISLAND kategorisinin sıklığı (göreli)", min: 0.0005, max: 0.2, value: 0.002, log: true },
  ],
  seed: true,
  run(q, r) {
    const R = train(q, r), cols = [0, 2, 1, 3, 5];
    const dNear = dist(R.E[2], R.E[3]), dInl = dist(R.E[3], R.E[1]), dNear0 = dist(R.E0[2], R.E0[3]), dInl0 = dist(R.E0[3], R.E0[1]);
    return {
      plots: [
        { title: "Embedding'ler: başlangıç (halka) → eğitim sonrası (nokta)", xlabel: "boyut 1", ylabel: "boyut 2", size: { w: 420, h: 400 },
          series: CATS.flatMap((n, c) => [{ name: n, x: R.path[c].map((p) => p[0]), y: R.path[c].map((p) => p[1]), c: cols[c] }, { name: "", x: [R.E[c][0]], y: [R.E[c][1]], kind: "scatter", c: cols[c], r: 7 }, { name: "", x: [R.E0[c][0]], y: [R.E0[c][1]], kind: "scatter", c: cols[c], r: 3 }]) },
        { title: "Eğitim kaybı (MSE, iki hedef toplamı)", xlabel: "epoch", ylabel: "kayıp", ylog: true, series: [{ name: "batch kaybı", x: R.loss.map((v) => v[0]), y: R.loss.map((v) => Math.max(v[1], 1e-6)), c: 0 }], marks: q.noise > 0 ? [{ y: 2 * q.noise * q.noise, label: "gürültü tabanı 2σ²" }] : [] },
      ],
      stats: [
        { label: "‖NEAR OCEAN − NEAR BAY‖: başlangıç → son", value: `${fmt(dNear0, 3)} → ${fmt(dNear, 3)}` },
        { label: "‖NEAR BAY − INLAND‖: başlangıç → son", value: `${fmt(dInl0, 3)} → ${fmt(dInl, 3)}` },
        { label: "One-hot'ta herhangi iki kategori arası", value: `√2 ≈ ${fmt(Math.SQRT2, 4)} (hep aynı)` },
        { label: "ISLAND'ın örneklerdeki payı", value: `%${fmt(R.freq[4] * 100, 3)}` },
      ],
      note: `Embedding'ler rastgele başlar; gradyan inişi benzer etkiye sahip kategorileri (NEAR BAY ve NEAR OCEAN: ikisi de pahalı ve kıyıda) <b>birbirine yaklaştırır</b>, farklı olanı (INLAND) uzaklaştırır — çünkü hedefi iyi tahmin etmenin yolu benzerleri benzer vektörlerle göstermektir. Bu temsil öğrenmesidir. One-hot kodlamada her çift arası uzaklık √2, iç çarpım 0'dır: "benzerlik" bilgisi yoktur. Embedding gradyanı yalnız batch'te görülen kategorilerin satırlarına düşer (scatter-add): ${R.freq[4] < 0.01 ? "<b>nadir ISLAND</b> çok az güncelleme alır ve başlangıç noktasına yakın kalabilir — nadir tokenların embedding'leri de böyledir." : "ISLAND yeterince sık, o da düzenleniyor."} Model W·e + b ile e'nin yalnız bir doğrusal dönüşümünü gördüğü için embedding'in kendisi döndürme/ölçekleme dışında belirlidir: eksenlerin anlamı yoktur, göreli konumlar vardır.`,
    };
  },
};
