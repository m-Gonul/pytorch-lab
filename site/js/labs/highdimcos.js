// M0.1 — Yüksek boyutta rastgele vektörler arasındaki kosinüs neden 0'a yığılır?
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "d", label: "Boyut d", min: 1, max: 2000, log: true, value: 3, int: true },
    { id: "dist", label: "Bileşen dağılımı", options: ["N(0, 1) (sıfır ortalamalı)", "Düzgün [0, 1] (hepsi pozitif, ör. ReLU çıktıları)"], value: 0 },
  ],
  seed: true,
  run(p, rnd) {
    const n = 1500, cs = [];
    const draw = () => Array.from({ length: p.d }, () => (p.dist === 0 ? rnd.normal() : rnd()));
    for (let k = 0; k < n; k++) { const a = draw(), b = draw(); let s = 0, na = 0, nb = 0; for (let i = 0; i < p.d; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } cs.push(s / Math.sqrt(na * nb)); }
    const m = cs.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(cs.reduce((a, b) => a + (b - m) ** 2, 0) / n);
    const bins = 40, h = new Array(bins).fill(0);
    for (const c of cs) h[Math.min(bins - 1, Math.floor(((c + 1) / 2) * bins))] += 1 / n;
    return {
      plots: [{ title: `${n} rastgele çiftin kosinüs benzerliği`, xlabel: "cos θ", ylabel: "oran", yzero: true, xlim: [-1, 1], series: [{ name: "histogram", x: range(bins).map((i) => -1 + (2 * i + 1) / bins), y: h, kind: "bar" }], marks: [{ x: 0, label: "dik" }] }],
      stats: [{ label: "Ortalama cos θ", value: m }, { label: "Std", value: sd }, { label: "Kuram (N(0,1) için) 1/√d", value: 1 / Math.sqrt(p.d) }, { label: "Açı (ortalamadan)", value: `${fmt((Math.acos(m) * 180) / Math.PI, 3)}°` }],
      note: p.dist === 0 ? `Bileşenler bağımsız ve sıfır ortalamalı: a·b'nin std'si ≈ √d, ‖a‖‖b‖ ≈ d, dolayısıyla cos θ'nın std'si ≈ 1/√d = ${fmt(1 / Math.sqrt(p.d), 3)}. Yüksek boyutta rastgele yönler neredeyse diktir ("ölçünün yoğunlaşması"). Attention'da skorları √d'ye bölmenin nedeni aynı hesap (M4.6).` : `Hepsi pozitif bileşenlerde vektörler aynı "çeyrekte" yaşar: cos θ 0'a değil ≈ ${fmt(m, 3)}'e yığılır (d büyüdükçe 3/4'e). Benzerlik ölçmeden önce merkezleme (ortalamayı çıkarma) bu ortak bileşeni kaldırır — o zaman kosinüs korelasyona dönüşür.`,
    };
  },
};
