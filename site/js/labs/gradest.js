// M5.1 — Örnekleme içinden gradyan: ∂/∂μ E_{z~N(μ,σ²)}[f(z)]'yi n örnekle kestirmenin iki yolu.
// Reparametrizasyon (yol türevi): z = μ + σε, ĝ = (1/n) Σ f'(μ + σε). Skor fonksiyonu (REINFORCE): ĝ = (1/n) Σ (f(z) − b)(z − μ)/σ².
// R tekrarla ortalama (yanlılık) ve standart sapma (gürültü) ölçülür; f türevlenemezse yol türevi yanlış sonuç verir.
import { fmt, range } from "./plot.js";

const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
export const F = [
  { name: "f(z) = z²", f: (z) => z * z, d: (z) => 2 * z, g: (mu) => 2 * mu },
  { name: "f(z) = sin z", f: Math.sin, d: Math.cos, g: (mu, s) => Math.cos(mu) * Math.exp(-0.5 * s * s) },
  { name: "f(z) = 1[z > 0] (basamak)", f: (z) => (z > 0 ? 1 : 0), d: () => 0, g: (mu, s) => pdf(mu / s) / s },
];

export default {
  params: [
    { id: "f", label: "Fonksiyon f", options: F.map((x) => x.name), value: 0 },
    { id: "mu", label: "μ", min: -2, max: 2, step: 0.05, value: 0.5 },
    { id: "sigma", label: "σ", min: 0.05, max: 3, step: 0.05, value: 1 },
    { id: "n", label: "Örnek sayısı n (kestirim başına)", min: 1, max: 1000, value: 10, int: true, log: true },
    { id: "base", label: "Skor kestiricisinde taban çizgisi b", options: ["yok (b = 0)", "örneklerin ortalaması"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const Fn = F[q.f], R = 400, true_ = Fn.g(q.mu, q.sigma), rp = [], sc = [], sc0 = [], sc1 = [];
    for (let k = 0; k < R; k++) {
      const eps = range(q.n).map(() => r.normal()), z = eps.map((e) => q.mu + q.sigma * e), fz = z.map(Fn.f);
      const fbar = fz.reduce((a, c) => a + c, 0) / q.n, est = (b) => z.reduce((a, zz, i) => a + (fz[i] - b) * (zz - q.mu) / q.sigma ** 2, 0) / q.n;
      rp.push(z.reduce((a, zz) => a + Fn.d(zz), 0) / q.n);
      sc0.push(est(0)); sc1.push(est(fbar)); sc.push(q.base ? sc1[k] : sc0[k]);
    }
    const st = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length, s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)); return { m, s }; };
    const A = st(rp), B = st(sc), B0 = st(sc0), B1 = st(sc1), ratio = A.s > 1e-9 ? B.s / A.s : null, lo = Math.min(...rp, ...sc), hi = Math.max(...rp, ...sc), nb = 40, w = (hi - lo) / nb || 1;
    const hist = (v) => { const h = new Array(nb).fill(0); v.forEach((x) => h[Math.min(nb - 1, Math.floor((x - lo) / w))]++); return h.map((c) => c / R); };
    const cs = range(nb).map((i) => lo + (i + 0.5) * w);
    const P = [`Gerçek gradyan ∂/∂μ E[f(z)] = ${fmt(true_, 4)}. ${R} tekrarda reparametrizasyon kestiricisinin ortalaması ${fmt(A.m, 4)}, standart sapması ${fmt(A.s, 3)}; skor fonksiyonu kestiricisinin ortalaması ${fmt(B.m, 4)}, standart sapması ${fmt(B.s, 3)} ${ratio == null ? "" : `(oran ${fmt(ratio, 3)})`}.`];
    if (q.f === 2) P.push(`<b>Basamak fonksiyonu</b> neredeyse her yerde türevi 0 olan bir fonksiyondur: yol türevi her örnekte 0 verir, oysa gerçek gradyan ${fmt(true_, 3)}. Reparametrizasyon, f'nin z'ye göre türevlenebilir olmasını <b>şart koşar</b>; ayrık çıktılı (ör. kategorik) değişkenlerde skor fonksiyonu ya da sürekli gevşetmeler (Gumbel-softmax) gerekir.`);
    else P.push(`Reparametrizasyon yansızdır; skor fonksiyonu f'nin türevini değil değerini kullandığı için genellikle daha gürültülüdür. Taban çizgisi: b = 0 ile std ${fmt(B0.s, 3)} (ortalama ${fmt(B0.m, 4)}), b = örnek ortalaması ile std ${fmt(B1.s, 3)} (ortalama ${fmt(B1.m, 4)}). ${B1.s < B0.s ? "Bu ayarda taban çizgisi varyansı düşürüyor." : "Bu ayarda taban çizgisi varyansı <b>düşürmüyor</b>: f'nin ortalaması zaten küçükken çıkarılan gürültülü ortalama yalnız gürültü ekliyor."} Örnek ortalaması aynı örneklerden hesaplandığı için kestiriciye (n − 1)/n çarpanı kadar yanlılık getirir; yansız sürüm her örnek için diğer n − 1 örneğin ortalamasını kullanır. VAE reparametrizasyon kullanır: kod çözücü z'ye göre türevlenebilirdir.`);
    return {
      plots: [
        { title: `${R} kestirimin dağılımı (n = ${q.n})`, xlabel: "kestirilen gradyan", ylabel: "oran", yzero: true, series: [
          { name: "reparametrizasyon", x: cs, y: hist(rp), kind: "bar", c: 0, bw: 0.45, off: -0.22 },
          { name: "skor fonksiyonu", x: cs, y: hist(sc), kind: "bar", c: 2, bw: 0.45, off: 0.22 }], marks: [{ x: true_, label: "gerçek" }] },
      ],
      stats: [
        { label: "Gerçek gradyan", value: true_ },
        { label: "Reparametrizasyon: ortalama ± std", value: `${fmt(A.m, 4)} ± ${fmt(A.s, 3)}` },
        { label: "Skor fonksiyonu: ortalama ± std", value: `${fmt(B.m, 4)} ± ${fmt(B.s, 3)}` },
        { label: "Std oranı (skor / reparam.)", value: ratio == null ? "— (reparam. std = 0)" : ratio },
      ],
      note: P.join(" "),
    };
  },
};
