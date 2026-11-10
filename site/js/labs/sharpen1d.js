// M9.4 — Keskinleştirme 1-B: bulanık bir basamak (σ_b) + gürültü. Yöntemler: Laplasyen g = f − c·f'' (c ağırlığı), keskinliği azaltılmış
// maskeleme / highboost g = f + k(f − G_σ ∗ f). Ölçütler: %10–%90 kenar genişliği, aşma (overshoot), düz bölgedeki gürültü std'si.
import { fmt, range } from "./plot.js";

const N = 120, gsm = (f, s) => { const R = Math.ceil(3 * s), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / s) ** 2)), S = w.reduce((a, b) => a + b, 0); return f.map((_, i) => w.reduce((a, v, j) => a + v * f[Math.max(0, Math.min(N - 1, i - R + j))], 0) / S); };
export default {
  params: [
    { id: "m", label: "Yöntem", options: ["Laplasyen: g = f − c·f″", "unsharp/highboost: g = f + k(f − bulanık)"], value: 1 },
    { id: "k", label: "Miktar (c ya da k)", min: 0, max: 4, step: 0.1, value: 1 },
    { id: "s", label: "Maske bulanıklaştırması σ (unsharp)", min: 0.5, max: 6, step: 0.1, value: 2, on: { m: [1] } },
    { id: "blur", label: "Görüntünün kendi bulanıklığı σ_b", min: 0.5, max: 5, step: 0.1, value: 2 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.01 },
  ],
  seed: true,
  run(q, r) {
    const step = range(N).map((i) => (i < 60 ? 0.2 : 0.8)), fc = gsm(step, q.blur), f = fc.map((v) => v + q.noise * r.normal());
    const sharpenOf = (a, k) => q.m === 0 ? a.map((v, i) => v - k * ((a[Math.min(N - 1, i + 1)] + a[Math.max(0, i - 1)]) - 2 * v)) : (() => { const b = gsm(a, q.s); return a.map((v, i) => v + k * (v - b[i])); })(), sharpen = (k) => sharpenOf(f, k), gc = sharpenOf(fc, q.k);
    const g = sharpen(q.k);
    const width = (a) => { const seg = a.slice(40, 81), i1 = seg.findIndex((v) => v >= 0.26), i2 = seg.findIndex((v) => v >= 0.74); return i1 < 0 || i2 < 0 ? NaN : i2 - i1; };
    const over = (a) => Math.max(Math.max(...a.slice(55, 90)) - 0.8, 0.2 - Math.min(...a.slice(30, 65)), 0);
    const nstd = (a) => { const z = a.slice(5, 35), m = z.reduce((s, v) => s + v, 0) / z.length; return Math.sqrt(z.reduce((s, v) => s + (v - m) ** 2, 0) / z.length); };
    const ks = range(21).map((i) => i * 0.2), sw = ks.map((k) => { const hc = sharpenOf(fc, k), h = sharpen(k); return [width(hc), over(hc), nstd(h)]; });
    const P = [`Gürültüsüz kenarda genişlik ${width(fc)} → ${width(gc)} örnek, aşma ${fmt(over(gc), 3)}; düz bölgede gürültü std'si ${fmt(nstd(f), 3)} → ${fmt(nstd(g), 3)} (×${fmt(nstd(g) / Math.max(nstd(f), 1e-9), 3)}).`];
    P.push(q.m === 0 ? "Laplasyen keskinleştirme kenarın iki yanına ters işaretli tepeler ekler: kenar dikleşir ama aşma ve gürültü büyür (f″'nün gürültü kazancı √6). 3 noktalı Laplasyen çok dar bir süzgeçtir: geniş bulanıklığı az düzeltir." : "Maske = f − bulanık, görüntünün yüksek frekans bileşenidir. k = 1 klasik keskinliği azaltılmış maskeleme, k > 1 highboost. Maskenin σ'sı görüntünün bulanıklığına yakın seçilince kenar en çok dikleşir; büyük k aşmayı (halo) ve gürültüyü büyütür.");
    return {
      plots: [
        { title: "Bulanık girdi ve keskinleştirilmiş çıkış", xlabel: "x", ylabel: "değer", series: [{ name: "temiz basamak", x: range(N), y: step, c: 4, dash: true }, { name: "girdi", x: range(N), y: f, c: 2 }, { name: "çıkış", x: range(N), y: g, c: 0 }], xlim: [30, 90] },
        { title: "Miktarın etkisi", xlabel: "c ya da k", ylabel: "değer", series: [{ name: "aşma", x: ks, y: sw.map((s) => s[1]), c: 1 }, { name: "gürültü std ×10", x: ks, y: sw.map((s) => 10 * s[2]), c: 0 }, { name: "kenar genişliği /10", x: ks, y: sw.map((s) => s[0] / 10), c: 2 }], marks: [{ x: q.k, label: "seçili" }] },
      ],
      stats: [{ label: "Kenar genişliği, gürültüsüz (girdi → çıkış)", value: `${width(fc)} → ${width(gc)}` }, { label: "Aşma (gürültüsüz)", value: over(gc) }, { label: "Gürültü std (girdi → çıkış)", value: `${fmt(nstd(f), 3)} → ${fmt(nstd(g), 3)}` }],
      note: P.join(" "),
    };
  },
};
