// M6.5 — Roofline modeli (Williams, Waterman, Patterson 2009): ulaşılabilir hız = min(tepe FLOP/s, aritmetik yoğunluk × bant genişliği).
// Aritmetik yoğunluk (AI) = FLOP / taşınan bayt. İşlemler: k işlemlik eleman bazlı zincir (füzyonsuz: her işlem ayrı çekirdek, N okur
// N yazar; füzyonlu: bir kez okur bir kez yazar), n × n × n matris çarpımı (2n³ FLOP, 3n² eleman), satır softmax (≈ 5 FLOP/eleman; füzyonsuz: max okur, exp okur-yazar, sum okur, bölme okur-yazar = 6N).
// Donanım değerleri üretici veri sayfalarından yaklaşık (yoğun, seyreklik yok).
import { fmt, range } from "./plot.js";

const OPS = ["eleman bazlı zincir (k işlem)", "matris çarpımı n × n × n", "satır softmax (N eleman)"];
export function opCost(q) {
  const b = q.dt === 0 ? 4 : 2;
  if (q.op === 0) { const N = 2 ** q.logN, flops = q.k * N; return q.fused ? { flops, bytes: 2 * N * b, kernels: 1 } : { flops, bytes: 2 * q.k * N * b, kernels: q.k }; }
  if (q.op === 1) { const n = 2 ** q.logN; return { flops: 2 * n ** 3, bytes: 3 * n * n * b, kernels: 1 }; }
  const N = 2 ** q.logN; return q.fused ? { flops: 5 * N, bytes: 2 * N * b, kernels: 1 } : { flops: 5 * N, bytes: 6 * N * b, kernels: 4 };
}
export default {
  params: [
    { id: "op", label: "İşlem", options: OPS, value: 0 },
    { id: "k", label: "Zincirdeki işlem sayısı k", min: 1, max: 20, value: 3, int: true, on: { op: [0] } },
    { id: "fused", label: "Füzyon", options: ["yok (her işlem ayrı çekirdek)", "var (tek çekirdek)"], value: 0, on: { op: [0, 2] } },
    { id: "logN", label: "Boyut: log₂ N (matris çarpımında log₂ n)", min: 4, max: 28, value: 20, int: true },
    { id: "dt", label: "Veri tipi", options: ["float32", "bf16"], value: 0 },
    { id: "peak", label: "Tepe hesap (TFLOP/s)", min: 1, max: 2000, value: 312, log: true },
    { id: "bw", label: "Bellek bant genişliği (TB/s)", min: 0.05, max: 5, value: 2, log: true },
    { id: "launch", label: "Çekirdek başlatma ek yükü (µs)", min: 0, max: 20, step: 0.5, value: 5 },
  ],
  run(q) {
    const c = opCost(q), ai = c.flops / c.bytes, ridge = (q.peak * 1e12) / (q.bw * 1e12), att = Math.min(q.peak * 1e12, ai * q.bw * 1e12);
    const tMem = c.bytes / (q.bw * 1e12), tComp = c.flops / (q.peak * 1e12), t = Math.max(tMem, tComp) + c.kernels * q.launch * 1e-6;
    const memBound = ai < ridge, us = (s) => fmt(s * 1e6, 3);
    const P = [`${OPS[q.op]}: ${fmt(c.flops, 3)} FLOP, ${fmt(c.bytes, 3)} bayt, aritmetik yoğunluk ${fmt(ai, 3)} FLOP/bayt. Makinenin dirsek noktası tepe / bant = ${fmt(ridge, 3)} FLOP/bayt: işlem <b>${memBound ? "bellek bağımlı" : "hesap bağımlı"}</b>; ulaşılabilir hız ${fmt(att / 1e12, 3)} TFLOP/s (tepenin %${fmt((100 * att) / (q.peak * 1e12), 3)} kadarı).`];
    P.push(`Süre tahmini: max(bellek ${us(tMem)} µs, hesap ${us(tComp)} µs) + ${c.kernels} çekirdek × ${q.launch} µs başlatma = ${us(t)} µs.`);
    if (q.op === 0) { const other = opCost({ ...q, fused: 1 - q.fused }), r = q.fused ? other.bytes / c.bytes : c.bytes / other.bytes; P.push(`Füzyon taşınan baytı ${fmt(r, 3)} kat azaltır (k = ${q.k}); FLOP aynıdır. Bellek bağımlı bir işlemde hız kazancının üst sınırı da budur.`); }
    if (q.op === 1) P.push(`Matris çarpımında yoğunluk n/3 (float32'de n/6) ile büyür: büyük matrisler hesap bağımlıdır ve füzyonun ona katacağı az şey vardır; kazanç, çarpımın ardından gelen eleman bazlı işlemleri (bias, aktivasyon) onun çekirdeğine katmaktan gelir.`);
    if (q.logN < 12 && q.op !== 1) P.push("Küçük boyutta süreyi başlatma ek yükü belirler: bu rejimde füzyon çekirdek sayısını azalttığı için daha da çok kazandırır; CUDA graph'ları (torch.compile mode=\"reduce-overhead\") ek yükü ayrıca düşürür.");
    const xs = range(61).map((i) => 10 ** (-2 + i * 0.1)), roof = xs.map((x) => Math.min(q.peak, (x * q.bw * 1e12) / 1e12));
    return {
      plots: [{ title: "Roofline", xlabel: "aritmetik yoğunluk (FLOP/bayt, log₁₀)", ylabel: "TFLOP/s", ylog: true, wide: true, size: { w: 620, h: 260 }, series: [
        { name: "çatı: min(tepe, AI × bant)", x: xs.map(Math.log10), y: roof, c: 4 },
        { name: "bu işlem", x: [Math.log10(ai)], y: [att / 1e12], kind: "scatter", c: 1, r: 6 }],
        marks: [{ x: Math.log10(ridge), label: "dirsek" }], xticks: [-2, -1, 0, 1, 2, 3, 4].map((v) => [v, `10^${v}`]) }],
      stats: [
        { label: "Aritmetik yoğunluk (FLOP/bayt)", value: ai },
        { label: "Dirsek noktası (FLOP/bayt)", value: ridge },
        { label: "Ulaşılabilir hız (TFLOP/s)", value: att / 1e12 },
        { label: "Tahmini süre (µs)", value: t * 1e6 },
      ],
      note: P.join(" "),
    };
  },
};
