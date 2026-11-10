// M9.3 — Yumuşatma süzgeçlerini karşılaştır: 1-B sinyal (basamak kenarları + ince bir çizgi + düz bölgeler), gürültü: Gauss, tuz-biber
// ya da ikisi. Süzgeçler: kutu (k), Gauss (σ, çekirdek ±3σ), medyan (k), ağırlıklı medyan (merkez ağırlığı w). Ölçüt: temiz sinyale
// MSE, kenar genişliği (%10–%90 yükselme mesafesi) ve ince çizginin tepe yüksekliği.
import { fmt, range } from "./plot.js";

const N = 240, clean = range(N).map((i) => (i >= 60 && i < 150 ? 0.8 : 0.2) + (i >= 190 && i < 192 ? 0.6 : 0));
const F = ["kutu", "Gauss", "medyan", "ağırlıklı medyan (merkez × w)"];
export function filt(x, q) {
  const k = q.k, r = Math.floor(k / 2), at = (i) => x[Math.max(0, Math.min(N - 1, i))];
  if (q.f === 0) return range(N).map((i) => range(k).reduce((s, j) => s + at(i - r + j), 0) / k);
  if (q.f === 1) { const R = Math.ceil(3 * q.sigma), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / q.sigma) ** 2)), S = w.reduce((a, b) => a + b, 0); return range(N).map((i) => w.reduce((s, v, j) => s + v * at(i - R + j), 0) / S); }
  return range(N).map((i) => { const v = []; for (let j = -r; j <= r; j++) { const rep = q.f === 3 && j === 0 ? q.w : 1; for (let t = 0; t < rep; t++) v.push(at(i + j)); } v.sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; });
}
export default {
  params: [
    { id: "f", label: "Süzgeç", options: F, value: 2 },
    { id: "k", label: "Pencere boyu k (tek)", min: 3, max: 15, step: 2, value: 5, int: true, on: { f: [0, 2, 3] } },
    { id: "sigma", label: "Gauss σ", min: 0.5, max: 5, step: 0.1, value: 1.5, on: { f: [1] } },
    { id: "w", label: "Merkez ağırlığı w", min: 1, max: 9, value: 3, int: true, on: { f: [3] } },
    { id: "noise", label: "Gürültü", options: ["Gauss (σ = 0.08)", "tuz-biber (%10)", "ikisi birden"], value: 1 },
  ],
  seed: true,
  run(q, r) {
    const x = clean.map((v) => { let y = v; if (q.noise !== 1) y += 0.08 * r.normal(); if (q.noise !== 0 && r() < 0.1) y = r() < 0.5 ? 0 : 1; return y; });
    const y = filt(x, q), mse = (a) => a.reduce((s, v, i) => s + (v - clean[i]) ** 2, 0) / N;
    const rise = (a) => { const seg = a.slice(45, 76), lo = 0.2 + 0.06, hi = 0.8 - 0.06; const i1 = seg.findIndex((v) => v >= lo), i2 = seg.findIndex((v) => v >= hi); return i1 < 0 || i2 < 0 ? NaN : i2 - i1; };
    const yc = filt(clean, q), ew = rise(yc), peak = Math.max(...y.slice(186, 196)) - 0.2;
    const all = [0, 1, 2, 3].map((f) => mse(filt(x, { ...q, f })));
    const P = [`${F[q.f]}: temiz sinyale MSE ${fmt(mse(y), 3)} (girdi ${fmt(mse(x), 3)}). Süzgecin temiz kenarı yayma genişliği (%10–%90) ${isNaN(ew) ? "ölçülemedi" : ew + " örnek"}; 2 örnek genişliğindeki çizginin tepesi ${fmt(peak, 3)} (temizde 0.6).`];
    P.push(`Aynı gürültüde MSE: kutu ${fmt(all[0], 3)}, Gauss ${fmt(all[1], 3)}, medyan ${fmt(all[2], 3)}, ağırlıklı medyan ${fmt(all[3], 3)}.`);
    if (q.noise === 1) P.push("Tuz-biber gürültüsü aykırı değerlerdir: doğrusal süzgeçler onları komşulara yayar (ortalamaya karışır), medyan ise sıralamada uçlara attığı için doğrudan yok eder.");
    if (q.f >= 2) P.push("Medyan basamak kenarlarını korur (kenar genişliği değişmez) ama pencerenin yarısından dar yapıları (ince çizgi) siler. Ağırlıklı medyan merkezi tekrarlayarak ince ayrıntıyı biraz daha korur, gürültüyü biraz daha az bastırır.");
    if (q.f <= 1 && q.noise === 0) P.push("Gauss gürültüsünde doğrusal yumuşatma iyi çalışır; bedeli kenarların yayılmasıdır.");
    return {
      plots: [{ title: "Sinyal, gürültülü girdi ve süzgeç çıkışı", xlabel: "konum", ylabel: "değer", wide: true, size: { w: 620, h: 240 }, series: [{ name: "gürültülü", x: range(N), y: x, kind: "scatter", c: 4, r: 1.6 }, { name: "temiz", x: range(N), y: clean, c: 2, dash: true }, { name: "çıkış", x: range(N), y: y, c: 0 }] }],
      stats: [{ label: "MSE (çıkış / girdi)", value: `${fmt(mse(y), 3)} / ${fmt(mse(x), 3)}` }, { label: "Kenar genişliği (temiz kenarda)", value: isNaN(ew) ? "—" : ew }, { label: "İnce çizginin tepesi (temiz 0.6)", value: peak }],
      note: P.join(" "),
    };
  },
};
