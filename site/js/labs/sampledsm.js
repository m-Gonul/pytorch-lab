// M4.13 — Örneklenmiş softmax (Jean ve ark. 2015): büyük çıktı sözlüğünde −log softmax(Wh + b)_t kaybının h'ye göre gradyanı
// ∇ = Σ_w p_w w_w − w_t'yi tüm V kelime yerine hedef + m örnek (öneri dağılımı q'dan, hedef çıkarılmış) üzerinden kestirmek.
// x3'teki gibi: örnek logit'lerinden log q_s çıkarılır (önem örneklemesi düzeltmesi), hedef logit'i olduğu gibi kalır.
// Modelin logit'leri Zipf benzeri bir önyargı (b_w = −log sıra) + rastgele W·h'den oluşur.
import { fmt, range } from "./plot.js";

const D = 16;
export function setup(V, r) {
  const W = range(V).map(() => range(D).map(() => 0.6 * r.normal())), h = range(D).map(() => r.normal() * 0.5);
  const z = W.map((w, i) => w.reduce((s, x, k) => s + x * h[k], 0) - Math.log(i + 1));
  return { W, h, z };
}
export function fullGrad(W, z, t) {
  const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), Z = e.reduce((a, b) => a + b, 0), g = range(D).map(() => 0);
  e.forEach((x, i) => { const p = x / Z; for (let k = 0; k < D; k++) g[k] += p * W[i][k]; });
  for (let k = 0; k < D; k++) g[k] -= W[t][k];
  return { g, pt: e[t] / Z };
}
export function sampledGrad(W, z, t, S, logq, corr) {
  const lm = corr === 2 ? Math.log(Math.max(1, S.length)) : 0, idx = [t, ...S], zz = idx.map((i, j) => z[i] - (j && corr ? logq[i] + lm : 0)), m = Math.max(...zz), e = zz.map((x) => Math.exp(x - m)), Z = e.reduce((a, b) => a + b, 0), g = range(D).map(() => 0);
  idx.forEach((i, j) => { const p = e[j] / Z; for (let k = 0; k < D; k++) g[k] += p * W[i][k]; });
  for (let k = 0; k < D; k++) g[k] -= W[t][k];
  return g;
}
const nrm = (a) => Math.hypot(...a), sub = (a, b) => a.map((x, i) => x - b[i]), cos = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (nrm(a) * nrm(b) + 1e-12);
function sampler(q, r) { const cdf = []; q.reduce((a, p, i) => (cdf[i] = a + p), 0); return () => { const u = r() * cdf.at(-1); let lo = 0, hi = cdf.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] < u) lo = m + 1; else hi = m; } return lo; }; }
export function estimate(W, z, t, q, m, corr, R, r) {
  const logq = q.map(Math.log), draw = sampler(q, r), acc = range(D).map(() => 0), gs = [];
  let hits = 0;
  for (let k = 0; k < R; k++) { const S = []; for (let j = 0; j < m; j++) { const s = draw(); if (s === t) hits++; else S.push(s); } const g = sampledGrad(W, z, t, S, logq, corr); gs.push(g); for (let i = 0; i < D; i++) acc[i] += g[i] / R; }
  return { mean: acc, gs, hits: hits / (R * m) };
}

export default {
  params: [
    { id: "V", label: "Çıktı sözlüğü V", min: 100, max: 20000, value: 5000, int: true, log: true },
    { id: "m", label: "Örnek sayısı m", min: 2, max: 2000, value: 50, int: true, log: true },
    { id: "prop", label: "Öneri dağılımı q", options: ["düzgün (1/V)", "Zipf (∝ 1/sıra)"], value: 1 },
    { id: "corr", label: "Örnek logit'lerinden çıkarılan", options: ["hiçbir şey (düzeltme yok)", "log q_s (ek x3)", "log(m·q_s): beklenen örnek sayısı"], value: 2 },
    { id: "t", label: "Hedef kelimenin sırası (1 = en sık)", min: 1, max: 5000, value: 30, int: true, log: true },
  ],
  seed: true,
  validate(q) { return q.t > q.V ? [{ level: "error", msg: "Hedef sırası V'den büyük olamaz." }] : q.m >= q.V ? [{ level: "warn", msg: "m ≥ V: örnekleme tam softmax'tan pahalı." }] : []; },
  run(q, r) {
    const { W, z } = setup(q.V, r), t = q.t - 1, F = fullGrad(W, z, t), raw = range(q.V).map((i) => (q.prop ? 1 / (i + 1) : 1)), Zq = raw.reduce((a, b) => a + b, 0), qq = raw.map((x) => x / Zq);
    const R = 120, cur = estimate(W, z, t, qq, q.m, q.corr, R, r), bias = nrm(sub(cur.mean, F.g)) / nrm(F.g);
    const single = cur.gs.reduce((s, g) => s + nrm(sub(g, F.g)) / nrm(F.g), 0) / R, cs = cur.gs.reduce((s, g) => s + cos(g, F.g), 0) / R;
    const ms = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000].filter((m) => m < q.V), curve = (c) => ms.map((m) => { const e = estimate(W, z, t, qq, m, c, 40, r); return nrm(sub(e.mean, F.g)) / nrm(F.g); });
    const c0 = curve(0), c1 = curve(1), c2 = curve(2);
    const P = [`Tam gradyan V = ${q.V} satırın hepsine dokunur (${q.V}·16 çarpma); örneklenmiş sürüm ${q.m + 1} satıra (${fmt(q.V / (q.m + 1), 3)} kat ucuz). Modelin hedefe verdiği olasılık p_t = ${fmt(F.pt, 3)}.`];
    P.push(`${R} örnek kümesinin ortalama gradyanının tam gradyana göreli uzaklığı (yanlılık) ${fmt(bias, 3)}; tek bir örnek kümesininki ortalama ${fmt(single, 3)} (yanlılık + varyans), kosinüs benzerliği ${fmt(cs, 3)}.`);
    if (q.prop === 1 && q.corr === 0) P.push("Zipf önerisi ve düzeltme yok: sık kelimeler örneklerde fazla temsil edildiği için gradyanın 'beklenen bağlam' kısmı sık kelimelere doğru kayar; m büyüse de bu yanlılık sıfıra inmez.");
    if (q.corr === 1) P.push(`log q düzeltmesiyle her örnek e^{z_s}/q_s ağırlığı alır; Σ_s e^{z_s}/q_s, hedef dışındaki toplamın <b>m katını</b> kestirir, hedef terimi ise 1 katında kalır. Hedefin payı m kat küçülür: p_t küçükse (burada ${fmt(F.pt, 3)}) bu fark önemsizdir, p_t büyükse yanlılık m arttıkça sıfıra inmez.`);
    if (q.corr === 2) P.push("log(m·q_s) düzeltmesiyle Σ_s e^{z_s}/(m·q_s), hedef dışı toplamın yansız kestiricisidir; hedef terimiyle birlikte paydanın tamamı kestirilir ve m büyüdükçe tahmin tam gradyana yakınsar (TensorFlow sampled_softmax_loss'un kullandığı biçim).");
    if (q.prop === 0 && q.corr === 1) P.push("Düzgün öneride log q = −log V sabittir; düzeltme yalnız örneklere ortak bir log V ekler (hedefe göre ağırlıklarını büyütür).");
    if (cur.hits > 0.02) P.push(`Örneklerin %${fmt(100 * cur.hits, 2)}'i hedefin kendisiydi ve atıldı ("accidental hit").`);
    return {
      plots: [
        { title: "Ortalama örneklenmiş gradyanın yanlılığı", xlabel: "örnek sayısı m (log₁₀ ekseni)", ylabel: "‖E[ĝ] − ∇‖ / ‖∇‖", ylog: true, xticks: ms.filter((m) => [2, 10, 100, 1000].includes(m)).map((m) => [Math.log10(m), String(m)]), series: [
          { name: "düzeltme yok", x: ms.map(Math.log10), y: c0, c: 5, dots: true },
          { name: "log q", x: ms.map(Math.log10), y: c1, c: 2, dots: true },
          { name: "log(m·q)", x: ms.map(Math.log10), y: c2, c: 0, dots: true }], marks: [{ x: Math.log10(q.m), label: `m = ${q.m}` }] },
        { title: "Gradyanın 16 bileşeni", xlabel: "bileşen", ylabel: "değer", series: [
          { name: "tam ∇", x: range(D), y: F.g, kind: "bar", c: 4, bw: 0.4, off: -0.2 },
          { name: `örneklenmiş, ${R} kümenin ortalaması`, x: range(D), y: cur.mean, kind: "bar", c: [5, 2, 0][q.corr], bw: 0.4, off: 0.2 }] },
      ],
      stats: [
        { label: "Yanlılık ‖E[ĝ] − ∇‖/‖∇‖", value: bias },
        { label: "Tek kümenin göreli hatası (ortalama)", value: single },
        { label: "cos(ĝ, ∇) ortalama", value: cs },
        { label: "Yanlılık, taranan en büyük m'de (log q / log(m·q))", value: `${fmt(c1.at(-1), 3)} / ${fmt(c2.at(-1), 3)}` },
        { label: "Maliyet oranı V / (m + 1)", value: q.V / (q.m + 1) },
        { label: "Modelin hedef olasılığı p_t", value: F.pt },
      ],
      note: P.join(" "),
    };
  },
};
