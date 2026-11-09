// M3.5 — Aktivasyon fonksiyonları ve türevleri; N(μ, σ²) girdi altında çıktı istatistikleri (ortalama, ikinci moment, ölü oranı, türevin ortalaması).
import { fmt, linspace } from "./plot.js";

const SA = 1.6732632423543772, SL = 1.0507009873554805;
const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)), phi = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
export const ACTS = [
  { name: "sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } },
  { name: "tanh", f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
  { name: "ReLU", f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0) },
  { name: "Leaky ReLU (α)", f: (z, a) => (z > 0 ? z : a * z), d: (z, a) => (z > 0 ? 1 : a) },
  { name: "ELU (α)", f: (z, a) => (z > 0 ? z : a * (Math.exp(z) - 1)), d: (z, a) => (z > 0 ? 1 : a * Math.exp(z)) },
  { name: "SELU", f: (z) => SL * (z > 0 ? z : SA * (Math.exp(z) - 1)), d: (z) => SL * (z > 0 ? 1 : SA * Math.exp(z)) },
  { name: "GELU", f: (z) => z * Phi(z), d: (z) => Phi(z) + z * phi(z) },
  { name: "Swish / SiLU", f: (z) => z / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s + z * s * (1 - s); } },
];

export default {
  params: [
    { id: "k", label: "Aktivasyon", options: ACTS.map((a) => a.name), value: 2 },
    { id: "alpha", label: "α (Leaky ReLU eğimi / ELU doyma düzeyi)", min: 0.001, max: 1, value: 0.01, log: true, on: { k: [3, 4] } },
    { id: "mu", label: "Girdi ortalaması μ", min: -3, max: 3, step: 0.1, value: 0, help: "Ön-aktivasyonların ortalaması: büyük negatif μ ReLU'yu 'öldürür', büyük |μ| sigmoid/tanh'ı doyurur." },
    { id: "sd", label: "Girdi std σ", min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const A = ACTS[q.k], a = q.alpha, zs = linspace(-5, 5, 401), N = 20000, xs = Array.from({ length: N }, () => q.mu + q.sd * r.normal());
    const ys = xs.map((z) => A.f(z, a)), ds = xs.map((z) => A.d(z, a)), mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    const m = mean(ys), m2 = mean(ys.map((y) => y * y)), dead = mean(ds.map((d) => (Math.abs(d) < 1e-3 ? 1 : 0))), dm = mean(ds), d2 = mean(ds.map((d) => d * d));
    const nb = 50, lo = Math.min(...ys), hi = Math.max(...ys), w = (hi - lo) / nb || 1, cnt = new Array(nb).fill(0); for (const y of ys) cnt[Math.min(nb - 1, Math.floor((y - lo) / w))]++;
    return {
      plots: [
        { title: `${A.name}: f(z) ve f′(z)`, xlabel: "z", ylabel: "değer", ylim: [-2, 3], series: [{ name: "f(z)", x: zs, y: zs.map((z) => A.f(z, a)), c: 0 }, { name: "f′(z)", x: zs, y: zs.map((z) => A.d(z, a)), c: 2, dash: true }], marks: [{ x: 0, label: "" }, { y: 0, label: "" }] },
        { title: `z ~ N(${fmt(q.mu, 2)}, ${fmt(q.sd, 2)}²) için f(z)'nin dağılımı`, xlabel: "f(z)", ylabel: "yoğunluk", yzero: true, series: [{ name: "histogram", x: cnt.map((_, i) => lo + (i + 0.5) * w), y: cnt.map((c) => c / (N * w)), kind: "bar", c: 0, bw: 0.95 }] },
      ],
      stats: [
        { label: "E[f(z)] (sıfır merkezli mi?)", value: m },
        { label: "E[f(z)²] / E[z²] (ikinci moment oranı)", value: m2 / (q.mu * q.mu + q.sd * q.sd) },
        { label: "E[f′(z)] (ortalama geçirgenlik)", value: dm },
        { label: "E[f′(z)²] (geri geçişte varyans çarpanı)", value: d2 },
        { label: "Türevi ≈ 0 olan oran (ölü/doymuş)", value: `%${fmt(dead * 100, 3)}` },
      ],
      note: `İleri geçişte E[f(z)²]/E[z²], bir katmandan sonra ikinci momentin hangi oranda korunduğudur; geri geçişte E[f′(z)²], gradyan varyansının çarpanıdır — başlatma bu iki çarpanı telafi edecek biçimde seçilir (He'nin 2'si ReLU'nun 1/2'sini telafi eder). ${q.k === 0 ? "Sigmoid: ortalama 0.5 (sıfır merkezli değil), türev en fazla 1/4 — her katman gradyanı en az 4 kat küçültür. " : q.k === 1 ? "tanh: sıfır merkezli, türev en fazla 1; ama |z| büyükken doyar. " : q.k === 2 ? "ReLU: yarısı tam sıfır (türev 0) — μ negatif olunca 'ölü' oran hızla artar; o nöronlar hiç gradyan almaz. " : q.k === 5 ? "SELU: N(0, 1) girdide çıktı ortalaması ≈ 0 ve ikinci moment ≈ 1 — öz-normalizasyonun sabit noktası. " : ""}`,
    };
  },
};
