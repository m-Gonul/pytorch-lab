// M6.2 — Straight-through estimator ile nicemleme farkında eğitim (QAT), en küçük örnek: y = a·x, model ŷ = q(w)·x,
// q(w) = Δ·round(clamp(w, −1, 1)/Δ), Δ = 2^(1−b) (b bit). Kayıp (q − a)²·E[x²], x ∈ [−1, 1] ızgarası. Gerçek türev ∂q/∂w = 0
// (neredeyse her yerde), STE: ∂q/∂w := 1; kırpmalı STE: ∂q/∂w := 1[|w| ≤ 1]. Tam batch gradyan inişi.
import { fmt, range } from "./plot.js";

export function quant(w, bits) { const D = 2 ** (1 - bits), c = Math.max(-1, Math.min(1, w)); return D * Math.round(c / D); }
export function train(q) {
  const ex2 = 1 / 3, ws = [q.w0], qs = [quant(q.w0, q.bits)], Ls = [];
  let w = q.w0;
  for (let t = 0; t < q.steps; t++) {
    const qw = quant(w, q.bits), L = (qw - q.a) ** 2 * ex2; Ls.push(L);
    const gq = 2 * (qw - q.a) * ex2, dq = q.est === 0 ? 1 : q.est === 1 ? 0 : Math.abs(w) <= 1 ? 1 : 0;
    w -= q.lr * gq * dq; ws.push(w); qs.push(quant(w, q.bits));
  }
  Ls.push((quant(w, q.bits) - q.a) ** 2 * ex2);
  return { ws, qs, Ls };
}
export default {
  params: [
    { id: "bits", label: "Bit sayısı b (adım Δ = 2^(1−b))", min: 1, max: 6, value: 2, int: true },
    { id: "a", label: "Hedef eğim a", min: -1.5, max: 1.5, step: 0.01, value: 0.6 },
    { id: "est", label: "Geri geçişte ∂q/∂w", options: ["STE: 1", "gerçek türev: 0", "kırpmalı STE: 1[|w| ≤ 1]"], value: 0 },
    { id: "lr", label: "Öğrenme oranı", min: 0.01, max: 10, value: 0.5, log: true },
    { id: "steps", label: "Adım", min: 10, max: 500, value: 100, int: true, log: true },
    { id: "w0", label: "Başlangıç w", min: -1, max: 1, step: 0.05, value: -0.5 },
  ],
  run(q) {
    const { ws, qs, Ls } = train(q), D = 2 ** (1 - q.bits), best = quant(q.a, q.bits), floor = (best - q.a) ** 2 / 3;
    const tail = qs.slice(-20), flips = tail.reduce((s, v, i) => s + (i && v !== tail[i - 1] ? 1 : 0), 0);
    const P = [`Izgara adımı Δ = ${fmt(D, 3)}; a = ${q.a}'ya en yakın nicemlenmiş değer ${fmt(best, 3)} ve ulaşılabilecek en küçük kayıp ${fmt(floor, 3)}. Son durum: w = ${fmt(ws.at(-1), 3)}, q(w) = ${fmt(qs.at(-1), 3)}, kayıp ${fmt(Ls.at(-1), 3)}.`];
    if (q.est === 1) P.push("Gerçek türev: round parçalı sabittir, türevi neredeyse her yerde 0; w hiç hareket etmez. Nicemlenmiş ağları doğrudan eğitmenin sorunu budur.");
    else P.push(`STE geri geçişte round'u özdeşlik sayar: gradyan "q yerine w kullanılsaydı" ne olacağını söyler. Gizli (tam hassasiyetli) w sürekli güncellenir, ileri geçiş ise hep nicemlenmiş q(w)'yu kullanır.`);
    if (flips > 2 && q.est !== 1) P.push(`Son 20 adımda q(w) ${flips} kez değişti: w iki ızgara noktası arasındaki eşiğin etrafında salınıyor. a ızgara noktalarının arasında kalınca gradyan hiç sıfırlanmaz; QAT'deki ağırlık salınımı (Nagel ve ark. 2022) bu mekanizmadır. Öğrenme oranını düşürmek salınımı daraltır ama yok etmez.`);
    if (Math.abs(q.a) > 1 && q.est === 2) P.push("|a| > 1: q en fazla ±1 olabilir. Kırpmalı STE, |w| > 1 olunca gradyanı keser ve w sınırda durur; düz STE'de w sınırsızca büyümeye devam eder (q yine 1'de kalır), bu da gizli ağırlığı anlamsız bölgeye kaçırır.");
    if (Math.abs(q.a) > 1 && q.est === 0) P.push(`|a| > 1 ve düz STE: q 1'de doyduğu hâlde gradyan sıfır olmadığı için w = ${fmt(ws.at(-1), 3)}'e kadar kaçtı. Kırpmalı STE bunu önler.`);
    const ts = range(ws.length);
    return {
      plots: [
        { title: "Gizli ağırlık w ve nicemlenmiş q(w)", xlabel: "adım", ylabel: "değer", series: [
          { name: "w (tam hassasiyet)", x: ts, y: ws, c: 0 }, { name: "q(w) (ileri geçişte)", x: ts, y: qs, c: 1 }, { name: "hedef a", x: [0, ws.length - 1], y: [q.a, q.a], c: 4, dash: true }] },
        { title: "Kayıp", xlabel: "adım", ylabel: "kayıp", ylog: true, series: [
          { name: "kayıp", x: range(Ls.length), y: Ls.map((v) => Math.max(v, 1e-8)), c: 0 }, { name: "nicemleme tabanı", x: [0, Ls.length - 1], y: [Math.max(floor, 1e-8), Math.max(floor, 1e-8)], c: 4, dash: true }] },
      ],
      stats: [
        { label: "Son w / q(w)", value: `${fmt(ws.at(-1), 3)} / ${fmt(qs.at(-1), 3)}` },
        { label: "Son kayıp / taban", value: `${fmt(Ls.at(-1), 3)} / ${fmt(floor, 3)}` },
        { label: "Son 20 adımda q değişimi", value: flips },
      ],
      note: P.join(" "),
    };
  },
};
