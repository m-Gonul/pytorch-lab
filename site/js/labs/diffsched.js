// M5.3 — Gürültü takvimi: ᾱ_t, SNR ve gürültülenmiş bir sinyal
import { fmt, range, linspace } from "./plot.js";
export default {
  params: [
    { id: "kind", label: "Takvim", options: ["doğrusal β (DDPM)", "kosinüs ᾱ (Nichol & Dhariwal)"], value: 0 },
    { id: "T", label: "Adım sayısı T", min: 10, max: 1000, step: 10, value: 1000, int: true },
    { id: "b1", label: "β_1 (doğrusal)", min: 1e-5, max: 0.01, log: true, value: 1e-4, on: { kind: [0] } },
    { id: "bT", label: "β_T (doğrusal)", min: 0.001, max: 0.1, log: true, value: 0.02, on: { kind: [0] } },
    { id: "t", label: "Gösterilen adım t", min: 0, max: 1000, step: 1, value: 300, int: true },
  ],
  seed: true,
  validate: (p) => [...(p.t > p.T ? [{ level: "error", msg: `t = ${p.t} > T = ${p.T}: adım 0…T aralığında olmalı.` }] : []), ...(p.kind === 0 && p.b1 > p.bT ? [{ level: "warn", msg: "β_1 > β_T: gürültü azalan takvim — alışılmış değil, sonlarda veri yeterince bozulmayabilir." }] : [])],
  run(p, rnd) {
    let ab;
    if (p.kind === 0) { const b = linspace(p.b1, p.bT, p.T); ab = []; let a = 1; for (const x of b) { a *= 1 - x; ab.push(a); } }
    else { const s = 0.008, f = (t) => Math.cos(((t / p.T + s) / (1 + s)) * Math.PI / 2) ** 2; ab = range(p.T).map((i) => Math.min(Math.max(f(i + 1) / f(0), 1e-5), 0.9999)); }
    const ab0 = [1, ...ab], t = range(p.T + 1), snr = ab0.map((a) => a / (1 - a));
    const xs = linspace(0, 1, 160), x0 = xs.map((x) => (x < 0.3 ? -0.8 : x < 0.6 ? 0.9 : Math.sin(18 * x) * 0.6));
    const a = ab0[Math.min(p.t, p.T)], eps = xs.map(() => rnd.normal()), xt = x0.map((v, i) => Math.sqrt(a) * v + Math.sqrt(1 - a) * eps[i]);
    const half = ab0.findIndex((v) => v < 0.5);
    return {
      plots: [
        { title: "ᾱ_t (sinyal payı) ve 1 − ᾱ_t", xlabel: "t", ylabel: "", ylim: [0, 1], series: [{ name: "ᾱ_t", x: t, y: ab0 }, { name: "1 − ᾱ_t", x: t, y: ab0.map((v) => 1 - v), c: 3 }], marks: [{ x: p.t, label: "t" }] },
        { title: "SNR = ᾱ/(1 − ᾱ)", xlabel: "t", ylabel: "SNR", ylog: true, series: [{ name: "SNR", x: t.slice(1), y: snr.slice(1) }], marks: [{ y: 1, label: "SNR = 1" }, { x: p.t, label: "t" }] },
        { title: `x_t = √ᾱ x₀ + √(1−ᾱ) ε  (t = ${p.t})`, xlabel: "konum", ylabel: "değer", ylim: [-3.5, 3.5], series: [{ name: "x₀", x: xs, y: x0, c: 1 }, { name: "x_t", x: xs, y: xt, c: 0 }] },
      ],
      stats: [{ label: "ᾱ_t (seçili t)", value: a }, { label: "ᾱ_T (son adım)", value: ab0[p.T] }, { label: "SNR = 1 olduğu (ᾱ < 0.5) ilk t", value: half }, { label: "Son adımda kalan sinyal √ᾱ_T", value: Math.sqrt(ab0[p.T]) }],
      note: ab0[p.T] > 0.01 ? `ᾱ_T = ${fmt(ab0[p.T])}: son adımda veri hâlâ seçilebiliyor; örnekleme saf N(0, I)'dan başladığı için eğitim–çıkarım uyuşmazlığı oluşur (T'yi ya da β_T'yi artır, "zero terminal SNR").` : `Doğrusal takvim SNR'yi erken söndürür: adımların büyük kısmı neredeyse saf gürültüde geçer. Kosinüs takvimi log-SNR'yi daha dengeli azaltır (küçük görüntülerde belirgin kazanç).`,
    };
  },
};
