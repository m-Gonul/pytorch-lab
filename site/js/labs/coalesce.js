// M7.1 — Bellek birleştirme (coalescing): bir warp'ın (32 thread) global bellek erişimleri 32 baytlık sektörlere düşer; donanım warp
// başına dokunulan sektörleri getirir. Thread t, eleman (offset + t·stride)'ı okur. Verim = gereken bayt / getirilen bayt.
// Ayrıca ızgara hesabı: N eleman, blockDim → ⌈N/blockDim⌉ block, boşta kalan thread'ler (sınır kontrolü).
import { fmt, range } from "./plot.js";

export function warpSectors(stride, offset, esize, sector = 32) {
  const addrs = range(32).map((t) => (offset + t * stride) * esize), secs = new Set(addrs.map((a) => Math.floor(a / sector)));
  return { addrs, sectors: secs.size, useful: 32 * esize, fetched: secs.size * sector };
}
export default {
  params: [
    { id: "stride", label: "Thread'ler arası adım (eleman)", min: 1, max: 64, value: 1, int: true, log: true },
    { id: "offset", label: "Başlangıç kayması (eleman)", min: 0, max: 31, value: 0, int: true },
    { id: "es", label: "Eleman boyu", options: ["1 bayt (int8)", "2 bayt (bf16)", "4 bayt (fp32)", "8 bayt (fp64)"], value: 2 },
    { id: "N", label: "Eleman sayısı N (ızgara hesabı)", min: 1, max: 100000, value: 1000, int: true, log: true },
    { id: "bd", label: "blockDim.x", options: ["32", "64", "128", "256", "512", "1024"], value: 3 },
  ],
  run(q) {
    const es = [1, 2, 4, 8][q.es], bd = [32, 64, 128, 256, 512, 1024][q.bd], W = warpSectors(q.stride, q.offset, es), eff = W.useful / W.fetched;
    const grid = Math.ceil(q.N / bd), idle = grid * bd - q.N, ideal = Math.ceil((32 * es) / 32);
    const strides = [1, 2, 4, 8, 16, 32, 64], effs = strides.map((s) => { const w = warpSectors(s, q.offset, es); return w.useful / w.fetched; });
    const P = [`Bir warp ${W.useful} bayt istiyor; erişimler ${W.sectors} sektöre (${W.fetched} bayt) düşüyor: verim %${fmt(100 * eff, 3)}. En iyi durum ${ideal} sektör.`];
    if (q.stride === 1 && q.offset === 0) P.push("Ardışık thread'ler ardışık adreslere erişiyor: erişim tam birleşik. Satır-ana bir tensörün son boyutu boyunca thread atamanın nedeni budur.");
    if (q.stride > 1) P.push(`Adım ${q.stride}: thread'ler arasında ${q.stride * es} bayt boşluk var; getirilen sektörlerin çoğu kullanılmayan veri taşıyor. Satır-ana bir matrisin sütununu thread başına okumak (ya da transpoze görünüm üzerinde çalışmak) bu desendir; shared memory üzerinden transpoze ya da erişim sırasını değiştirmek çözümdür.`);
    if (q.offset > 0 && q.stride === 1) P.push(`Kayma ${q.offset}: erişim bir sektör sınırından başlamıyor, bir sektör fazladan getiriliyor. Etkisi küçük ama sistematik; tensör hizalaması bu yüzden önemlidir.`);
    P.push(`Izgara: N = ${q.N}, blockDim = ${bd} → ${grid} block, ${grid * bd} thread; ${idle} thread sınır kontrolüne (if (i < N)) takılıp hiçbir şey yapmaz. Global indeks i = blockIdx.x · blockDim.x + threadIdx.x.`);
    return {
      plots: [
        { title: "İlk warp'ın erişimleri (sektörler gri çizgiler)", xlabel: "bayt adresi", ylabel: "thread", series: [{ name: "erişilen adres", x: W.addrs, y: range(32), kind: "scatter", c: 0, r: 3 }],
          marks: range(Math.min(W.sectors + 2, 70)).map((k) => ({ x: Math.floor(W.addrs[0] / 32) * 32 + k * 32, label: "" })).filter((m) => m.x <= W.addrs[31] + 32) },
        { title: "Verim ve adım", xlabel: "adım (log₂)", ylabel: "gereken / getirilen", ylim: [0, 1.05], series: [{ name: "verim", x: strides.map(Math.log2), y: effs, c: 1 }], xticks: strides.map((s) => [Math.log2(s), String(s)]), marks: [{ x: Math.log2(q.stride), label: "adım" }] },
      ],
      stats: [
        { label: "Sektör (getirilen / en iyi)", value: `${W.sectors} / ${ideal}` },
        { label: "Verim", value: eff },
        { label: "Block sayısı / boşta thread", value: `${grid} / ${idle}` },
      ],
      note: P.join(" "),
    };
  },
};
