// M7.1 — Tiled matmul: global bellek okumalarını say. Naif: her C_ij, A'nın satırını ve B'nin sütununu okur (2n³ eleman). T × T tile ile
// her block k boyunca n/T adımda iki T × T tile'ı shared memory'ye yükler: 2n³/T. Küçük n için okumalar döngüyle sayılır (sayaç),
// büyük n için formül. Tile'ın aritmetik yoğunluğu: adım başına 2T³ FLOP / (2T² · eleman boyu) = T / eleman boyu FLOP/bayt.
import { fmt, range } from "./plot.js";

export function countReads(n, T) {
  let reads = 0; const nb = n / T;
  if (T === 0) return 2 * n ** 3;
  for (let bi = 0; bi < nb; bi++) for (let bj = 0; bj < nb; bj++) for (let k = 0; k < nb; k++) { for (let a = 0; a < T * T; a++) reads += 2; }
  return reads;
}
export default {
  params: [
    { id: "logn", label: "Matris boyu: log₂ n", min: 2, max: 12, value: 5, int: true },
    { id: "logT", label: "Tile boyu: log₂ T (0 = naif)", min: 0, max: 7, value: 2, int: true },
    { id: "es", label: "Eleman boyu", options: ["2 bayt (bf16)", "4 bayt (fp32)"], value: 1 },
    { id: "smem", label: "Block başına shared memory sınırı (KiB)", min: 16, max: 228, value: 48, int: true },
    { id: "ridge", label: "Donanımın dirsek noktası (FLOP/bayt)", min: 1, max: 300, value: 10, log: true },
  ],
  run(q) {
    const n = 2 ** q.logn, T = Math.min(2 ** q.logT, n), es = q.es ? 4 : 2, naive = q.logT === 0;
    const counted = n <= 64 ? (naive ? (() => { let r = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) r += 2 * n; return r; })() : countReads(n, T)) : null;
    const formula = naive ? 2 * n ** 3 : (2 * n ** 3) / T, reads = counted ?? formula;
    const smemB = naive ? 0 : 2 * T * T * es, threads = naive ? 0 : T * T, ai = naive ? 1 / es : T / es;
    const okSmem = smemB <= q.smem * 1024, okThreads = threads <= 1024, tMax = Math.floor(Math.sqrt((q.smem * 1024) / (2 * es)));
    const Ts = [1, 2, 4, 8, 16, 32, 64, 128].filter((t) => t <= n);
    const P = [`n = ${n}, ${naive ? "naif" : `T = ${T}`}: global okuma ${fmt(reads, 4)} eleman${counted !== null ? " (döngüyle sayıldı, formülle aynı: " + fmt(formula, 4) + ")" : " (formül)"}; naif ${fmt(2 * n ** 3, 4)}, oran ${fmt((2 * n ** 3) / reads, 3)}.`];
    if (!naive) {
      P.push(`Block başına shared memory 2T² × ${es} = ${fmt(smemB / 1024, 3)} KiB (${okSmem ? "sınırın %${fmt((100 * smemB) / (q.smem * 1024), 3)} kadarı" : "<b>sınırı aşıyor</b>"}; ${q.smem} KiB ile en büyük T = ${tMax}); thread başına bir çıktı ile ${threads} thread (${okThreads ? "≤ 1024" : "<b>1024 sınırını aşıyor</b>"}).`);
      P.push(`Tile'ın aritmetik yoğunluğu T / ${es} = ${fmt(ai, 3)} FLOP/bayt; dirsek ${q.ridge}. ${ai >= q.ridge ? "Hesap bağımlı bölgeye ulaşıldı." : `Hâlâ bellek bağımlı: T ≥ ${Math.ceil(q.ridge * es)} gerekir.`}`);
      if (!okThreads || ai < q.ridge) P.push("Thread başına bir çıktı ile T en fazla 32 olabilir (1024 thread). Daha büyük etkin tile için her thread birden çok çıktıyı yazmaçlarda biriktirir (yazmaç döşemesi, 2-B blocktiling); hızlı matmul çekirdekleri böyle çalışır (Boehm'ün yazısındaki adımlar).");
    } else P.push("Naif çekirdekte her eleman n kez okunur; yoğunluk eleman başına 2 FLOP / 2 okuma = 1 / eleman boyu FLOP/bayt. Önbellekler bunun bir kısmını kurtarır ama tahmin edilemez biçimde.");
    return {
      plots: [
        { title: `Global okuma ve tile boyu (n = ${n})`, xlabel: "T (log₂)", ylabel: "eleman", ylog: true, series: [{ name: "2n³ / T", x: Ts.map(Math.log2), y: Ts.map((t) => (2 * n ** 3) / t), c: 0 }],
          marks: naive ? [] : [{ x: Math.log2(T), label: "T" }], xticks: Ts.map((t) => [Math.log2(t), String(t)]) },
        { title: "Tile'ın aritmetik yoğunluğu ve dirsek", xlabel: "T (log₂)", ylabel: "FLOP/bayt", ylog: true, series: [{ name: "T / eleman boyu", x: Ts.map(Math.log2), y: Ts.map((t) => t / es), c: 1 }, { name: "dirsek", x: Ts.map(Math.log2), y: Ts.map(() => q.ridge), c: 4, dash: true }],
          marks: [{ x: 5, label: "1024 thread" }], xticks: Ts.map((t) => [Math.log2(t), String(t)]) },
      ],
      stats: [
        { label: "Global okuma (eleman)", value: reads },
        { label: "Naife göre azalma", value: (2 * n ** 3) / reads },
        { label: "Shared memory / block (KiB) ve sınırın izin verdiği en büyük T", value: `${fmt(smemB / 1024, 3)} / T ≤ ${tMax}` },
        { label: "Aritmetik yoğunluk (FLOP/bayt)", value: ai },
      ],
      note: P.join(" "),
    };
  },
};
