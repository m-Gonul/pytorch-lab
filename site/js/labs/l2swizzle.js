// M7.2 — Program sırası ve L2 yeniden kullanımı (Triton matmul eğitimindeki "gruplu sıra"). C, (Mb × Nb) tile'a bölünmüş; her program
// C'nin bir tile'ını hesaplar ve A'nın satır bloğundan Kb, B'nin sütun bloğundan Kb tile okur. Aynı anda W program çalışır (bir dalga);
// bir dalgadaki programların ortak tile'ları DRAM'den bir kez gelir (L2'de paylaşılır), dalgalar arası önbellek yok sayılır.
// Satır-ana sıra ile GROUP_M gruplu sıra karşılaştırılır. 9 × 9 örneği: ilk 9 çıktı için 90'a karşı 54 tile.
import { fmt, range } from "./plot.js";

export function order(Mb, Nb, G) {
  if (G <= 1) return range(Mb * Nb).map((p) => [Math.floor(p / Nb), p % Nb]);
  return range(Mb * Nb).map((pid) => { const per = G * Nb, gid = Math.floor(pid / per), first = gid * G, gs = Math.min(Mb - first, G); return [first + ((pid % per) % gs), Math.floor((pid % per) / gs)]; });
}
export function traffic(Mb, Nb, Kb, G, W) {
  const ord = order(Mb, Nb, G); let tot = 0; const waves = [];
  for (let s = 0; s < ord.length; s += W) { const w = ord.slice(s, s + W), rows = new Set(w.map((p) => p[0])), cols = new Set(w.map((p) => p[1])), t = (rows.size + cols.size) * Kb; tot += t; waves.push(t); }
  return { tot, waves, ord };
}
export default {
  params: [
    { id: "Mb", label: "C'nin satır blok sayısı (M / BM)", min: 1, max: 64, value: 9, int: true },
    { id: "Nb", label: "C'nin sütun blok sayısı (N / BN)", min: 1, max: 64, value: 9, int: true },
    { id: "Kb", label: "K blok sayısı (K / BK)", min: 1, max: 64, value: 9, int: true },
    { id: "G", label: "GROUP_M (1 = satır-ana sıra)", min: 1, max: 16, value: 3, int: true },
    { id: "W", label: "Aynı anda çalışan program (dalga)", min: 1, max: 256, value: 9, int: true, log: true },
  ],
  run(q) {
    const R = traffic(q.Mb, q.Nb, q.Kb, 1, q.W), S = traffic(q.Mb, q.Nb, q.Kb, q.G, q.W), naive = q.Mb * q.Nb * 2 * q.Kb;
    const Gs = range(16).map((g) => g + 1), sw = Gs.map((g) => traffic(q.Mb, q.Nb, q.Kb, g, q.W).tot);
    const img = new Array(q.Mb * q.Nb).fill(0); S.ord.forEach(([i, j], k) => (img[i * q.Nb + j] = Math.floor(k / q.W)));
    const nw = Math.max(1, Math.ceil(S.ord.length / q.W));
    const P = [`${q.Mb} × ${q.Nb} = ${q.Mb * q.Nb} program, dalga başına ${q.W}. DRAM'den okunan tile: satır-ana ${R.tot}, GROUP_M = ${q.G} ile ${S.tot} (oran ${fmt(R.tot / S.tot, 3)}); hiç paylaşım olmasaydı ${naive}.`];
    P.push(`İlk dalga: satır-ana ${R.waves[0]}, gruplu ${S.waves[0]} tile. Satır-ana sırada bir dalga A'nın tek bir satır bloğunu ama B'nin çok sayıda sütun bloğunu ister; gruplu sırada dalga yaklaşık kare bir C bölgesini kapsar ve hem A hem B tile'ları dalga içinde paylaşılır.`);
    if (q.Mb === 9 && q.Nb === 9 && q.Kb === 9 && q.W === 9 && q.G === 3) P.push("Triton eğitimindeki örnek: ilk 9 çıktı tile'ı için satır-ana 90, 3'lü gruplu 54 tile.");
    if (q.G >= q.Mb) P.push("GROUP_M ≥ satır blok sayısı: gruplu sıra sütun-ana sıraya dönüşür; dalga bu kez B'nin tek sütun bloğunu, A'nın çok satır bloğunu ister.");
    return {
      plots: [
        { title: "Program sırası (renk: dalga numarası)", xlabel: "pid_n", ylabel: "pid_m", size: { w: 300, h: 280 }, side: true, img: { w: q.Nb, h: q.Mb, data: img.map((v) => 0.1 + (0.75 * v) / Math.max(1, nw - 1)) }, series: [] },
        { title: "DRAM tile okuması ve GROUP_M", xlabel: "GROUP_M", ylabel: "tile", yzero: true, series: [{ name: "gruplu", x: Gs, y: sw, c: 0 }, { name: "satır-ana", x: Gs, y: Gs.map(() => R.tot), c: 4, dash: true }], marks: [{ x: q.G, label: "G" }] },
      ],
      stats: [
        { label: "Tile okuması: satır-ana / gruplu", value: `${R.tot} / ${S.tot}` },
        { label: "İlk dalga: satır-ana / gruplu", value: `${R.waves[0]} / ${S.waves[0]}` },
        { label: "Paylaşımsız üst sınır", value: naive },
      ],
      note: P.join(" "),
    };
  },
};
