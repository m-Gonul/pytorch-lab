// M9.2 — Histogram eşleme (specification). Girdi yoğunluklarının dağılımı p_r, istenen dağılım p_z. T(r) = (L−1)·CDF_r(r) (eşitleme),
// G(z) = (L−1)·CDF_z(z); eşleme z = G⁻¹(T(r)): her r için G(z) ≥ T(r) olan en küçük z. Ayrık hâlde histogram tam istenen biçimi
// almaz (yalnız birleştirme yapılır, kova bölünmez). Girdi: N piksellik sentetik örneklem (karanlık, düşük karşıtlıklı ya da iki tepeli).
import { fmt, range } from "./plot.js";

const L = 256, SRC = ["karanlık (tek tepe, düşük değerler)", "düşük karşıtlık (orta gri, dar)", "iki tepeli (koyu arka plan + açık nesne)"];
const TGT = ["düzgün (= histogram eşitleme)", "Gauss (ortalama μ, std σ)", "iki tepeli (μ ± 60)"];
const clamp = (v) => Math.max(0, Math.min(L - 1, Math.round(v)));
export function pdfTarget(k, mu, sd) {
  const g = (m) => range(L).map((z) => Math.exp(-0.5 * ((z - m) / sd) ** 2));
  const p = k === 0 ? range(L).map(() => 1) : k === 1 ? g(mu) : g(mu - 60).map((v, i) => v + g(mu + 60)[i]);
  const s = p.reduce((a, b) => a + b, 0); return p.map((v) => v / s);
}
export function matchLUT(hist, pz) {
  const N = hist.reduce((a, b) => a + b, 0); let c = 0; const T = hist.map((h) => { c += h; return ((L - 1) * c) / N; });
  let cz = 0; const G = pz.map((p) => { cz += p; return (L - 1) * cz; });
  return T.map((t) => { let z = 0; while (z < L - 1 && G[z] < t - 1e-9) z++; return z; });
}
export default {
  params: [
    { id: "src", label: "Girdi dağılımı", options: SRC, value: 0 },
    { id: "tgt", label: "İstenen dağılım", options: TGT, value: 1 },
    { id: "mu", label: "İstenen ortalama μ", min: 40, max: 215, value: 128, int: true, on: { tgt: [1, 2] } },
    { id: "sd", label: "İstenen std σ", min: 5, max: 80, value: 40, int: true, on: { tgt: [1, 2] } },
    { id: "N", label: "Piksel sayısı", min: 500, max: 50000, value: 10000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const px = range(q.N).map(() => clamp(q.src === 0 ? 40 + 18 * r.normal() : q.src === 1 ? 120 + 8 * r.normal() : r() < 0.7 ? 50 + 12 * r.normal() : 190 + 15 * r.normal()));
    const h = new Array(L).fill(0); px.forEach((v) => h[v]++);
    const pz = pdfTarget(q.tgt, q.mu, q.sd), lut = matchLUT(h, pz), ho = new Array(L).fill(0); px.forEach((v) => ho[lut[v]]++);
    const mean = (hh) => hh.reduce((s, c, i) => s + c * i, 0) / q.N, std = (hh) => { const m = mean(hh); return Math.sqrt(hh.reduce((s, c, i) => s + c * (i - m) ** 2, 0) / q.N); };
    const used = (hh) => hh.filter((c) => c > 0).length;
    let ci = 0, cz = 0, ks = 0; for (let i = 0; i < L; i++) { ci += ho[i] / q.N; cz += pz[i]; ks = Math.max(ks, Math.abs(ci - cz)); }
    const bin = 4, nb = L / bin, agg = (hh, norm) => range(nb).map((b) => hh.slice(b * bin, b * bin + bin).reduce((a, c) => a + c, 0) / norm);
    const P = [`Girdi: ortalama ${fmt(mean(h), 4)}, std ${fmt(std(h), 3)}, kullanılan seviye ${used(h)}. Çıkış: ortalama ${fmt(mean(ho), 4)}, std ${fmt(std(ho), 3)}, kullanılan seviye ${used(ho)}. Çıkışın birikimli dağılımı istenenden en fazla ${fmt(ks, 3)} sapıyor (Kolmogorov–Smirnov uzaklığı).`];
    P.push("Eşleme iki eşitlemenin bileşimidir: önce girdi T ile düzgün dağılıma, sonra G⁻¹ ile istenen dağılıma. Ayrık seviyeler bölünemediği için (aynı r'deki bütün pikseller aynı z'ye gider) çıkış histogramı istenen biçime yalnız yaklaşır; kullanılan seviye sayısı girdininkini aşamaz.");
    if (q.tgt === 0) P.push("İstenen dağılım düzgünken bu tam olarak histogram eşitlemedir (G özdeşlik).");
    if (q.src === 1) P.push("Dar girdi histogramı: az sayıda seviye geniş bir aralığa yayılıyor, çıkış histogramında aralarında boşluklar (dişler) kalıyor.");
    return {
      plots: [
        { title: "Histogramlar (4'lük kovalar)", xlabel: "yoğunluk", ylabel: "oran", series: [{ name: "girdi", x: range(nb).map((b) => b * bin), y: agg(h, q.N), c: 4 }, { name: "çıkış", x: range(nb).map((b) => b * bin), y: agg(ho, q.N), c: 0 }, { name: "istenen", x: range(nb).map((b) => b * bin), y: agg(pz, 1), c: 1, dash: true }] },
        { title: "Eşleme eğrisi z = G⁻¹(T(r))", xlabel: "r", ylabel: "z", ylim: [0, 255], series: [{ name: "LUT", x: range(L), y: lut, c: 0 }, { name: "özdeşlik", x: [0, 255], y: [0, 255], c: 4, dash: true }] },
      ],
      stats: [
        { label: "Ortalama: girdi → çıkış", value: `${fmt(mean(h), 4)} → ${fmt(mean(ho), 4)}` },
        { label: "Std: girdi → çıkış", value: `${fmt(std(h), 3)} → ${fmt(std(ho), 3)}` },
        { label: "Kullanılan seviye: girdi / çıkış", value: `${used(h)} / ${used(ho)}` },
        { label: "KS uzaklığı (çıkış, istenen)", value: ks },
      ],
      note: P.join(" "),
    };
  },
};
