// M0.4 — Dağılımdan örnek çek: histogram teorik PMF/PDF'ye, örneklem ortalaması gerçek ortalamaya yaklaşır (büyük sayılar yasası).
import { fmt, linspace, range } from "./plot.js";

const lgamma = (z) => { // Lanczos
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
};
export const binomPmf = (k, m, p) => Math.exp(lgamma(m + 1) - lgamma(k + 1) - lgamma(m - k + 1) + (p > 0 ? k * Math.log(p) : k ? -Infinity : 0) + (p < 1 ? (m - k) * Math.log(1 - p) : m - k ? -Infinity : 0));

export const DISTS = [
  { name: "Bernoulli(p)", disc: true, mean: (q) => q.p, vr: (q) => q.p * (1 - q.p), draw: (q, r) => (r() < q.p ? 1 : 0), pmf: (q) => [[0, 1], [1 - q.p, q.p]] },
  { name: "Binom(m, p)", disc: true, mean: (q) => q.m * q.p, vr: (q) => q.m * q.p * (1 - q.p), draw: (q, r) => { let s = 0; for (let i = 0; i < q.m; i++) s += r() < q.p; return s; }, pmf: (q) => { const k = range(q.m + 1); return [k, k.map((i) => binomPmf(i, q.m, q.p))]; } },
  { name: "Gauss N(μ, σ²)", mean: (q) => q.mu, vr: (q) => q.sigma ** 2, draw: (q, r) => q.mu + q.sigma * r.normal(), pdf: (q, x) => Math.exp(-((x - q.mu) ** 2) / (2 * q.sigma ** 2)) / (q.sigma * Math.sqrt(2 * Math.PI)), lim: (q) => [q.mu - 4 * q.sigma, q.mu + 4 * q.sigma] },
  { name: "Üstel Exp(λ)", mean: (q) => 1 / q.lam, vr: (q) => 1 / q.lam ** 2, draw: (q, r) => -Math.log(1 - r()) / q.lam, pdf: (q, x) => (x < 0 ? 0 : q.lam * Math.exp(-q.lam * x)), lim: (q) => [0, 6 / q.lam] },
];

export default {
  params: [
    { id: "dist", label: "Dağılım", options: DISTS.map((d) => d.name), value: 2 },
    { id: "p", label: "Başarı olasılığı p", min: 0, max: 1, step: 0.01, value: 0.3, on: { dist: [0, 1] }, why: "Olasılık [0, 1] aralığında olmalı." },
    { id: "m", label: "Deneme sayısı m", min: 1, max: 40, step: 1, value: 10, int: true, on: { dist: [1] } },
    { id: "mu", label: "Ortalama μ", min: -5, max: 5, step: 0.1, value: 0, on: { dist: [2] } },
    { id: "sigma", label: "Standart sapma σ", min: 0.1, max: 4, step: 0.1, value: 1, hardMin: 0.01, on: { dist: [2] }, why: "σ ≤ 0 geçerli bir Gauss vermez." },
    { id: "lam", label: "Hız λ", min: 0.1, max: 5, step: 0.1, value: 1, hardMin: 0.01, on: { dist: [3] }, why: "λ > 0 olmalı." },
    { id: "N", label: "Örnek sayısı N", min: 5, max: 20000, value: 200, log: true, int: true, help: "N büyüdükçe histogram teorik eğriye, örneklem ortalaması gerçek ortalamaya yaklaşır." },
  ],
  seed: true,
  run(q, r) {
    const D = DISTS[q.dist], N = Math.round(q.N), xs = [];
    for (let i = 0; i < N; i++) xs.push(D.draw(q, r));
    const mean = xs.reduce((a, b) => a + b, 0) / N, v0 = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
    const M = D.mean(q), V = D.vr(q);
    let P;
    if (D.disc) {
      const [ks, pm] = D.pmf(q), cnt = ks.map(() => 0);
      for (const x of xs) cnt[x]++;
      P = { title: `${D.name}: göreli frekans ve PMF`, xlabel: "değer k", ylabel: "olasılık", yzero: true, series: [
        { name: `örneklem: sayı / N (N = ${N})`, x: ks, y: cnt.map((c) => c / N), kind: "bar", c: 0, bw: 0.6 },
        { name: "teorik P(X = k)", x: ks, y: pm, kind: "scatter", c: 2, r: 4.5 },
      ] };
    } else {
      const [a, b] = D.lim(q), nb = Math.max(10, Math.min(60, Math.round(Math.sqrt(N) * 1.5))), w = (b - a) / nb, cnt = new Array(nb).fill(0);
      for (const x of xs) { const i = Math.floor((x - a) / w); if (i >= 0 && i < nb) cnt[i]++; }
      const cx = range(nb).map((i) => a + (i + 0.5) * w), gx = linspace(a, b, 200);
      P = { title: `${D.name}: yoğunluk histogramı ve PDF`, xlabel: "x", ylabel: "yoğunluk", yzero: true, series: [
        { name: `histogram: sayı / (N · kutu genişliği)`, x: cx, y: cnt.map((c) => c / (N * w)), kind: "bar", c: 0, bw: 0.95 },
        { name: "teorik p(x)", x: gx, y: gx.map((x) => D.pdf(q, x)), c: 2 },
      ] };
    }
    const run = [], idx = [];
    let s = 0, step = Math.max(1, Math.floor(N / 300));
    for (let i = 0; i < N; i++) { s += xs[i]; if (i % step === 0 || i === N - 1) { idx.push(i + 1); run.push(s / (i + 1)); } }
    return {
      plots: [P, { title: "Koşan ortalama: ilk n örneğin ortalaması", xlabel: "n", ylabel: "ortalama", series: [
        { name: "x̄ₙ", x: idx, y: run, c: 0 },
        { name: "μ ± 2σ/√n bandı", x: idx, y: idx.map((n) => M + (2 * Math.sqrt(V)) / Math.sqrt(n)), c: 4, dash: true },
        { name: "", x: idx, y: idx.map((n) => M - (2 * Math.sqrt(V)) / Math.sqrt(n)), c: 4, dash: true },
      ], marks: [{ y: M, label: "gerçek E[X]" }] }],
      stats: [
        { label: "Teorik E[X]", value: M }, { label: "Örneklem ortalaması x̄", value: mean },
        { label: "Teorik Var[X]", value: V }, { label: "Örneklem varyansı (1/N, MLE)", value: v0 },
        { label: "Örneklem varyansı (1/(N−1), yansız)", value: N > 1 ? (v0 * N) / (N - 1) : NaN },
        { label: "Ortalamanın standart hatası σ/√N", value: Math.sqrt(V / N) },
      ],
      note: `${D.disc ? "Ayrık dağılımda her çubuk bir olasılıktır (toplamı 1). " : "Sürekli dağılımda histogram <b>yoğunluk</b> olarak ölçeklenir (alanı 1): tek bir noktanın olasılığı 0'dır, olasılık aralıkların alanıdır. Yoğunluk 1'i aşabilir (σ küçükken dene). "}Sağdaki eğri <b>büyük sayılar yasası</b>dır: x̄ₙ gerçek ortalamaya yaklaşır ve sapması kabaca σ/√n hızında küçülür — N'yi 100 kat artırmak hatayı yalnız 10 kat azaltır. Kesikli bant, yaklaşık %95 aralığıdır.`,
    };
  },
};
