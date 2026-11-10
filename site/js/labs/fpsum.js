// M10.5 — Kayan nokta toplamının birleşmeli olmaması: N tane fp32 sayının toplamı üç yolla fp32'de hesaplanır (Math.fround ile her
// adımda yuvarlama): sıralı (soldan sağa), ikili (indirgeme ağacı: komşu çiftler, sonra çiftlerin çiftleri), Kahan (telafili). Başvuru:
// aynı fp32 değerlerin float64'te Neumaier toplamı (fp32 hatasından mertebelerce küçük). Ayrıca blok başına 256 öğe sıralı + blok
// sonuçlarının ağaçla toplanması (GPU tarzı iki düzeyli). Veri: düzgün [0, 1), karışık işaretli, geniş dinamik aralık (10^-6–10^6).
import { fmt, range } from "./plot.js";

const f = Math.fround;
function seqSum(a) { let s = 0; for (const x of a) s = f(s + x); return s; }
function pairSum(a) { let v = a.map(f); while (v.length > 1) { const n = []; for (let i = 0; i + 1 < v.length; i += 2) n.push(f(v[i] + v[i + 1])); if (v.length % 2) n.push(v[v.length - 1]); v = n; } return v[0]; }
function kahan(a) { let s = 0, c = 0; for (const x of a) { const y = f(x - c), t = f(s + y); c = f(f(t - s) - y); s = t; } return s; }
function blockSum(a, B = 256) { const parts = []; for (let i = 0; i < a.length; i += B) parts.push(seqSum(a.slice(i, i + B))); return pairSum(parts); }
function ref(a) { let s = 0, c = 0; for (const x of a) { const t = s + x; c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s; s = t; } return s + c; }
export default {
  params: [
    { id: "logN", label: "Öğe sayısı N = 10^k", min: 2, max: 6, step: 0.5, value: 5 },
    { id: "dist", label: "Veri", options: ["düzgün [0, 1)", "karışık işaretli (−1, 1)", "geniş aralık (10^-6 – 10^6, pozitif)"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const N = Math.round(10 ** q.logN), a = range(N).map(() => f(q.dist === 0 ? r() : q.dist === 1 ? 2 * r() - 1 : 10 ** (12 * r() - 6)));
    const R = ref(a), names = ["sıralı", "ikili (ağaç)", "Kahan", "blok (256) + ağaç"], vals = [seqSum(a), pairSum(a), kahan(a), blockSum(a)];
    const absSum = a.reduce((s, x) => s + Math.abs(x), 0), eps = 2 ** -24, rel = vals.map((v) => Math.abs(v - R) / absSum);
    const Ns = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6].map((k) => Math.round(10 ** k)), curves = [0, 1, 2].map(() => []);
    for (const n of Ns) { const b = range(n).map(() => f(q.dist === 0 ? r() : q.dist === 1 ? 2 * r() - 1 : 10 ** (12 * r() - 6))), rr = ref(b), as = b.reduce((s, x) => s + Math.abs(x), 0); [seqSum, pairSum, kahan].forEach((fn, k) => curves[k].push(Math.max(1e-12, Math.abs(fn(b) - rr) / as))); }
    const P = [`N = ${N}: göreli hata (Σ|x|'e göre) sıralı ${fmt(rel[0], 3)}, ikili ${fmt(rel[1], 3)}, Kahan ${fmt(rel[2], 3)}, blok + ağaç ${fmt(rel[3], 3)}; fp32 birim yuvarlaması ε = ${fmt(eps, 3)}.`];
    P.push(`Kuramsal üst sınırlar: sıralı ≈ (N − 1)ε = ${fmt((N - 1) * eps, 3)}, ikili ≈ ⌈log₂N⌉ε = ${fmt(Math.ceil(Math.log2(N)) * eps, 3)}, Kahan ≈ 2ε. Pratikte hatalar rastgele işaretli birikir ve sınırın altında kalır (≈ √N ve √log N ölçeği).`);
    if (q.dist === 0 && N >= 1e5) P.push("Düzgün pozitif veride sıralı toplam büyüdükçe küçük sayılar eklenince yuvarlanıp kayboluyor: hata sistematik birikiyor.");
    return {
      plots: [{ title: "Göreli hata ve N", xlabel: "log₁₀ N", ylabel: "göreli hata", wide: true, ylog: true, size: { w: 620, h: 230 }, series: [...[0, 1, 2].map((k) => ({ name: names[k], x: Ns.map(Math.log10), y: curves[k], c: k, dots: true })), { name: "ε", x: [2, 6], y: [eps, eps], c: 4, dash: true }], marks: [{ x: q.logN, label: "seçili N" }] }],
      stats: names.map((nm, k) => ({ label: `Göreli hata: ${nm}`, value: fmt(rel[k], 3) })),
      note: P.join(" "),
    };
  },
};
