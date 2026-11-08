// M9.12 — 1B'de Gauss ve bilateral süzgeç: kenar korunumu, σ_r, yineleme
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "ss", label: "Uzamsal σ_s (örnek)", min: 0.5, max: 12, step: 0.5, value: 4 },
    { id: "sr", label: "Değer σ_r", min: 0.01, max: 2, log: true, value: 0.15, help: "σ_r → ∞: bilateral Gauss'a dönüşür. σ_r ≪ gürültü: süzgeç hiçbir şey yapmaz." },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.4, step: 0.01, value: 0.08 },
    { id: "it", label: "Yineleme sayısı", min: 1, max: 8, step: 1, value: 1, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const n = 200, x = range(n), clean = x.map((i) => (i < 70 ? 0.2 : i < 130 ? 0.8 : 0.5 + 0.004 * (i - 130)) + (i > 95 && i < 99 ? 0.25 : 0));
    const noisy = clean.map((v) => v + p.noise * rnd.normal());
    const R = Math.ceil(3 * p.ss);
    const filt = (s, rangeW) => s.map((c, i) => { let a = 0, w = 0; for (let k = -R; k <= R; k++) { const j = Math.min(n - 1, Math.max(0, i + k)); const ww = Math.exp(-(k * k) / (2 * p.ss * p.ss)) * (rangeW ? Math.exp(-((s[j] - c) ** 2) / (2 * p.sr * p.sr)) : 1); a += ww * s[j]; w += ww; } return a / w; });
    let g = noisy, b = noisy;
    for (let t = 0; t < p.it; t++) { g = filt(g, false); b = filt(b, true); }
    const mse = (a) => a.reduce((s, v, i) => s + (v - clean[i]) ** 2, 0) / n;
    const edge = (a) => Math.max(...range(n - 1).map((i) => Math.abs(a[i + 1] - a[i])));
    return {
      plots: [{ title: "Sinyal (adım kenarları, ince çizgi, eğim)", xlabel: "konum", ylabel: "yoğunluk", series: [{ name: "gürültülü", x, y: noisy, c: 4 }, { name: "temiz", x, y: clean, c: 1, dash: true }, { name: "Gauss", x, y: g, c: 3 }, { name: "bilateral", x, y: b, c: 0 }] }],
      stats: [{ label: "MSE gürültülü", value: mse(noisy) }, { label: "MSE Gauss", value: mse(g) }, { label: "MSE bilateral", value: mse(b) }, { label: "En dik adım: temiz / Gauss / bilateral", value: `${fmt(edge(clean), 3)} / ${fmt(edge(g), 3)} / ${fmt(edge(b), 3)}` }],
      note: p.sr < p.noise ? "σ_r gürültü genliğinden küçük: komşuların çoğu 'farklı' sayıldığı için bilateral neredeyse hiç düzleştirmiyor." : p.sr > 1 ? "σ_r çok büyük: değer ağırlığı ≈ 1, bilateral Gauss'tan ayırt edilemez (kenarlar yayılır)." : "Bilateral, değer farkı σ_r'yi aşan komşuları dışarıda bıraktığı için kenarlarda ortalama almaz. İnce çizgi (konum ≈ 97) ise σ_r'den küçük kontrastlıysa silinir; yinelemek düz bölgeleri 'basamaklı' yapar (karikatürleşme).",
    };
  },
};
