// M10.5 — Blok içi indirgeme ağacı: T thread, 2T öğe. Eşlemeler: (0) basit: adım 1, 2, 4, …; thread t, t % adım == 0 ise
// input[2t] += input[2t + adım] (sahip konumlar giderek uzaklaşır); (1) yakınsak: adım T, T/2, …, 1; thread t < adım ise
// input[t] += input[t + adım] (etkin thread'ler bitişik). Her yinelemede: etkin thread, etkin warp, ayrışan warp, SIMT verimi
// (etkin thread / (32 × iş yapan warp)), global bellek isteği (iki okuma + bir yazma; 32 baytlık sektör sayısı, 4 baytlık öğe).
import { fmt, range } from "./plot.js";

export function simulate(T, map) {
  const iters = [];
  const steps = map === 0 ? range(Math.log2(2 * T)).map((k) => 2 ** k) : range(Math.log2(2 * T)).map((k) => T >> k);
  for (const s of steps) {
    const act = range(T).filter((t) => (map === 0 ? t % s === 0 : t < s));
    const warps = new Map(); for (const t of act) warps.set(t >> 5, (warps.get(t >> 5) || 0) + 1);
    const W = warps.size, div = [...warps.values()].filter((c) => c < 32).length;
    let sectors = 0; for (const [w] of warps) { const ts = act.filter((t) => t >> 5 === w); const own = ts.map((t) => (map === 0 ? 2 * t : t)), oth = own.map((o) => o + s); for (const arr of [own, oth, own]) sectors += new Set(arr.map((i) => Math.floor((4 * i) / 32))).size; }
    iters.push({ s, act: act.length, W, div, eff: W ? act.length / (32 * W) : 0, sectors, useful: act.length * 3 * 4 });
  }
  return iters;
}
export default {
  params: [
    { id: "map", label: "Thread eşlemesi", options: ["basit (adım 1, 2, 4, …)", "yakınsak (adım T, T/2, …, 1)"], value: 0 },
    { id: "logT", label: "Blok boyu T = 2^k (2T öğe)", min: 5, max: 10, value: 7, int: true },
  ],
  run(q) {
    const T = 2 ** q.logT, it = simulate(T, q.map), other = simulate(T, 1 - q.map);
    const warpSteps = it.reduce((a, x) => a + x.W, 0), divSteps = it.filter((x) => x.div > 0).length, sec = it.reduce((a, x) => a + x.sectors, 0), useful = it.reduce((a, x) => a + x.useful, 0);
    const warpO = other.reduce((a, x) => a + x.W, 0), secO = other.reduce((a, x) => a + x.sectors, 0);
    const P = [`${2 * T} öğe, ${T} thread, ${it.length} yineleme. Toplam ${warpSteps} warp-yinelemesi (diğer eşleme ${warpO}); ayrışan warp içeren yineleme ${divSteps}. Global bellek: ${sec} sektör (${fmt((useful / (32 * sec)) * 100, 3)}% yararlı bayt; diğer eşleme ${secO}).`];
    P.push(q.map === 0 ? "Basit eşlemede etkin thread'ler adım kadar aralıklı: her warp ayrışıyor ve erişimler dağınık; warp'lar ancak adım 32'yi geçince bütünüyle boşalıyor." : `Yakınsak eşlemede etkin thread'ler bitişik: ilk ${Math.max(0, it.length - 5)} yinelemede ayrışma yok, warp'lar bütünüyle boşalıyor; son 5 yineleme (16, 8, 4, 2, 1 thread) tek warp'ta ayrışıyor.`);
    return {
      plots: [
        { title: "Yineleme başına iş yapan warp", xlabel: "yineleme", ylabel: "warp", side: true, size: { w: 300, h: 230 }, xlim: [0.4, it.length + 0.6], series: [{ name: "iş yapan warp", x: range(it.length).map((i) => i + 1), y: it.map((x) => x.W), kind: "bar", c: 0 }, { name: "ayrışan", x: range(it.length).map((i) => i + 1), y: it.map((x) => x.div), kind: "bar", c: 1, bw: 0.4 }] },
        { title: "Yineleme başına SIMT verimi", xlabel: "yineleme", ylabel: "verim", side: true, size: { w: 300, h: 230 }, ylim: [0, 1.05], series: [{ name: "bu eşleme", x: range(it.length).map((i) => i + 1), y: it.map((x) => x.eff), c: 0, dots: true }, { name: "diğer eşleme", x: range(other.length).map((i) => i + 1), y: other.map((x) => x.eff), c: 2, dash: true }] },
      ],
      stats: [{ label: "Warp-yinelemesi (bu / diğer)", value: `${warpSteps} / ${warpO}` }, { label: "Ayrışan yineleme", value: divSteps }, { label: "Bellek sektörü (bu / diğer)", value: `${sec} / ${secO}` }],
      note: P.join(" "),
    };
  },
};
