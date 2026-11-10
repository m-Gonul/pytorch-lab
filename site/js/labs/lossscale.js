// M7.3 — Gradyan alt taşması ve dinamik kayıp ölçekleme (Micikevicius ve ark. 2018; PyTorch GradScaler). Gradyan büyüklükleri log-normal:
// log₁₀|g| ~ N(μ, s²). fp16'ya dönüşümde |g| < 2^−25 olanlar sıfır olur (alt taşma), 2^−24 … 2^−14 alt normaldir (hassasiyet kaybı),
// > 65504 taşar. Ölçek S ile çarpılınca dağılım log₁₀S kadar kayar. Dinamik ölçekleyici: taşma (inf/NaN) görülen adım atlanır ve S
// yarıya iner; aralıkla ardışık iyi adımdan sonra S iki katına çıkar. Taşmalar ara sıra gelen gradyan sıçramalarıyla modellenir.
import { fmt, range } from "./plot.js";

const Phi = (z) => 0.5 * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z ** 3)));
const LMIN = Math.log10(2 ** -25), LSUB = Math.log10(2 ** -14), LMAX = Math.log10(65504);
export function fractions(mu, s, logS) { const m = mu + logS; return { under: Phi((LMIN - m) / s), sub: Phi((LSUB - m) / s) - Phi((LMIN - m) / s), over: 1 - Phi((LMAX - m) / s) }; }
export default {
  params: [
    { id: "mu", label: "Gradyan büyüklüğünün medyanı: 10^μ, μ", min: -10, max: -1, step: 0.1, value: -6 },
    { id: "s", label: "Yayılım (log₁₀ std)", min: 0.2, max: 3, step: 0.1, value: 1.2 },
    { id: "init", label: "Başlangıç ölçeği log₂ S₀", min: 0, max: 24, value: 16, int: true },
    { id: "interval", label: "Büyütme aralığı (ardışık iyi adım)", min: 1, max: 2000, value: 200, int: true, log: true },
    { id: "spike", label: "Adım başına gradyan sıçraması olasılığı", min: 0, max: 0.05, step: 0.001, value: 0.005 },
    { id: "steps", label: "Adım", min: 100, max: 5000, value: 2000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    let lS = q.init, good = 0, skipped = 0; const hist = [], skips = [];
    for (let t = 0; t < q.steps; t++) {
      const spike = r() < q.spike ? 2 + 3 * r() : 0;                         // bu adımda en büyük gradyan 10^(μ + 4s + spike) civarı
      const maxlog = q.mu + 4 * q.s + spike + lS * Math.log10(2);
      if (maxlog > LMAX) { lS -= 1; good = 0; skipped++; skips.push(t); } else { good++; if (good >= q.interval) { lS += 1; good = 0; } }
      hist.push(lS);
    }
    const f0 = fractions(q.mu, q.s, 0), fS = fractions(q.mu, q.s, (hist.reduce((a, b) => a + b, 0) / hist.length) * Math.log10(2));
    const xs = range(141).map((i) => -14 + i * 0.15), pdf = (m) => xs.map((x) => Math.exp(-0.5 * ((x - m) / q.s) ** 2) / (q.s * Math.sqrt(2 * Math.PI)));
    const mS = (hist.reduce((a, b) => a + b, 0) / hist.length);
    const P = [];
    P.push(`Ölçeksiz: fp16'da sıfıra inen gradyan oranı ${fmt(f0.under, 3)}, alt normal bölgedeki oran ${fmt(f0.sub, 3)}. Dinamik ölçekleyicinin ortalama ölçeği 2^${fmt(mS, 3)}: sıfıra inen oran ${fmt(fS.under, 3)}, taşan oran ${fmt(fS.over, 3)}.`);
    P.push(`${q.steps} adımda ${skipped} adım taşma yüzünden atlandı (optimizasyon adımı yapılmadı, ölçek yarıya indi). Ölçekleyici, taşmanın hemen altında durmaya çalışır: aralık büyüdükçe daha az deneme ama daha yavaş toparlanma.`);
    if (q.mu + 4 * q.s > LMAX) P.push("Gradyanların kendisi bile fp16 aralığını aşıyor: ölçek 1'in altına inmek zorunda. Böyle modellerde bf16 (aralığı fp32 ile aynı) daha uygundur.");
    return {
      plots: [
        { title: "log₁₀|g| dağılımı ve fp16 sınırları", xlabel: "log₁₀ |g|", ylabel: "yoğunluk", series: [{ name: "ölçeksiz", x: xs, y: pdf(q.mu), c: 4 }, { name: "ortalama ölçekle", x: xs, y: pdf(q.mu + mS * Math.log10(2)), c: 0 }],
          marks: [{ x: LMIN, label: "sıfır" }, { x: LSUB, label: "normal" }, { x: LMAX, label: "65504" }] },
        { title: "Ölçek (log₂ S) ve atlanan adımlar", xlabel: "adım", ylabel: "log₂ S", series: [{ name: "log₂ S", x: range(q.steps), y: hist, c: 0 }, { name: "atlanan adım", x: skips, y: skips.map((t) => hist[t]), kind: "scatter", c: 1, r: 3 }] },
      ],
      stats: [
        { label: "Sıfıra inen oran: ölçeksiz / ölçekli", value: `${fmt(f0.under, 3)} / ${fmt(fS.under, 3)}` },
        { label: "Ortalama log₂ S", value: mS },
        { label: "Atlanan adım", value: skipped },
      ],
      note: P.join(" "),
    };
  },
};
