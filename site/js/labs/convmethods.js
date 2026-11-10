// M10.7 — Evrişim yöntemlerinin maliyet karşılaştırması (H × W görüntü, C_in → C_out kanal, K × K filtre, 'same' dolgu):
// doğrudan: 2·H·W·C_in·C_out·K² FLOP; im2col + GEMM: aynı FLOP, ek bellek H·W·C_in·K² öğe (sütun matrisi); FFT: her kanal için
// 2-B FFT (5·N log₂N, N = H'·W', H' = H + K − 1'e kadar 2'nin kuvvetine dolgu), frekansta karmaşık çarpma-toplama 8·N·C_in·C_out,
// ters FFT; ayrılabilir (rank-1 filtre): 2·H·W·C_in·C_out·2K. Süre tahmini = FLOP / (verim × tepe) + bayt / bant genişliği
// (A100 benzeri: 19.5 TFLOPS fp32, 1.5 TB/s); GEMM verimi 0.8, doğrudan çekirdek verimi K ile 0.3–0.6, FFT 0.3.
import { fmt, range } from "./plot.js";

const PK = 19.5e12, BW = 1.5e12;
export function costs(H, Cin, Cout, K) {
  const W = H, pix = H * W, direct = 2 * pix * Cin * Cout * K * K;
  const np = 2 ** Math.ceil(Math.log2(H + K - 1)), N = np * np, fft = 5 * N * Math.log2(N) * (Cin + Cout) + 8 * N * Cin * Cout;
  const sep = 2 * pix * Cin * Cout * 2 * K;
  const bytesIn = 4 * pix * (Cin + Cout), col = 4 * pix * Cin * K * K;
  const effD = 0.3 + 0.3 * Math.min(1, (K * K) / 49);
  return [
    { name: "doğrudan", flop: direct, mem: bytesIn, t: direct / (effD * PK) + bytesIn / BW },
    { name: "im2col + GEMM", flop: direct, mem: bytesIn + 2 * col, t: direct / (0.8 * PK) + (bytesIn + 2 * col) / BW },
    { name: "FFT", flop: fft, mem: bytesIn + 8 * N * (Cin + Cout), t: fft / (0.3 * PK) + (bytesIn + 16 * N * (Cin + Cout)) / BW },
    { name: "ayrılabilir (rank-1)", flop: sep, mem: 2 * bytesIn, t: sep / (effD * PK) + (2 * bytesIn) / BW },
  ];
}
export default {
  params: [
    { id: "logH", label: "Görüntü kenarı H = 2^k", min: 6, max: 11, value: 9, int: true },
    { id: "logC", label: "Kanal sayısı C_in = C_out = 2^k", min: 0, max: 9, value: 6, int: true },
    { id: "K", label: "Filtre boyu K", min: 1, max: 31, step: 2, value: 3, int: true },
  ],
  run(q) {
    const H = 2 ** q.logH, C = 2 ** q.logC, cs = costs(H, C, C, q.K), best = cs.slice(0, 3).reduce((a, b) => (b.t < a.t ? b : a)), colMiB = (4 * H * H * C * q.K * q.K) / 2 ** 20;
    const Ks = range(16).map((k) => 2 * k + 1), curves = [0, 1, 2].map((m) => Ks.map((k) => costs(H, C, C, k)[m].t * 1e3));
    const P = [`H = ${H}, C = ${C}, K = ${q.K}: doğrudan ${fmt(cs[0].flop / 1e9, 3)} GFLOP; im2col sütun matrisi ${fmt((4 * H * H * C * q.K * q.K) / 2 ** 20, 4)} MiB; FFT ${fmt(cs[2].flop / 1e9, 3)} GFLOP. Genel yöntemlerden tahmini en hızlı: ${best.name} (${fmt(best.t * 1e3, 3)} ms); filtre ayrılabilir olsaydı ${fmt(cs[3].t * 1e3, 3)} ms.`];
    if (colMiB > 40 * 1024) P.push(`im2col sütun matrisi (${fmt(colMiB / 1024, 3)} GiB) GPU belleğine sığmaz: pratikte örtük GEMM (sütunlar anında üretilir) kullanılır.`);
    P.push(`Doğrudan ve im2col'un işi K² ile büyür, FFT'ninki K'dan neredeyse bağımsızdır (yalnız dolgu): büyük K'da FFT kazanır. 3 × 3 gibi küçük filtrelerde iyi eniyilenmiş GEMM ya da doğrudan çekirdekler (Winograd) öndedir. Ayrılabilir filtreler (Gauss gibi) ayrı bir sınıftır: 2K maliyetle hepsini geçer.`);
    if (C === 1) P.push("Tek kanalda GEMM'in iç boyutu yalnız K²: matris çarpımı verimsizleşir, doğrudan ya da FFT daha uygundur.");
    return {
      plots: [{ title: "Tahmini süre ve filtre boyu", xlabel: "K", ylabel: "ms", wide: true, ylog: true, size: { w: 620, h: 230 }, series: [0, 1, 2].map((m) => ({ name: cs[m].name, x: Ks, y: curves[m], c: m, dots: true })), marks: [{ x: q.K, label: "seçili K" }] }],
      stats: cs.map((c) => ({ label: `${c.name}: GFLOP / ms`, value: `${fmt(c.flop / 1e9, 3)} / ${fmt(c.t * 1e3, 3)}` })),
      note: P.join(" "),
    };
  },
};
