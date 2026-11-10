// M10.4 — Doluluk (occupancy) hesaplayıcısı. SM başına sınırlar mimariye göre: en çok thread, en çok blok, yazmaç dosyası, paylaşımlı
// bellek; blok başına en çok thread. Bir bloğun SM'de yer kaplaması: warp sayısı ⌈T/32⌉; yazmaçlar warp başına ayrılır (R·32, 256'lık
// birimlere yuvarlanır; G80/GT200'de blok başına 512'lik birim); paylaşımlı bellek 256 baytlık birimlerde. SM'de aynı anda kalabilen blok
// sayısı dört sınırın en küçüğüdür; doluluk = etkin warp / en çok warp.
import { fmt, range } from "./plot.js";

const ARCH = [
  { n: "G80 (1.0)", thr: 768, blk: 8, regs: 8192, smem: 16, tpb: 512, maxR: 124, old: true },
  { n: "GT200 (1.3)", thr: 1024, blk: 8, regs: 16384, smem: 16, tpb: 512, maxR: 124, old: true },
  { n: "Fermi (2.0)", thr: 1536, blk: 8, regs: 32768, smem: 48, tpb: 1024, maxR: 63 },
  { n: "Kepler (3.5)", thr: 2048, blk: 16, regs: 65536, smem: 48, tpb: 1024, maxR: 255 },
  { n: "Ampere A100 (8.0)", thr: 2048, blk: 32, regs: 65536, smem: 164, tpb: 1024, maxR: 255 },
  { n: "Hopper H100 (9.0)", thr: 2048, blk: 32, regs: 65536, smem: 228, tpb: 1024, maxR: 255 },
];
export function occ(a, T, R, S) {
  const w = Math.ceil(T / 32), byThr = Math.floor(a.thr / (w * 32)), byBlk = a.blk;
  const regBlk = a.old ? Math.ceil((R * T) / 512) * 512 : w * Math.ceil((R * 32) / 256) * 256, byReg = R === 0 ? Infinity : Math.floor(a.regs / regBlk);
  const smBlk = Math.ceil((S * 1024) / 256) * 256, bySm = S === 0 ? Infinity : Math.floor((a.smem * 1024) / smBlk);
  const b = Math.min(byThr, byBlk, byReg, bySm), names = ["thread", "blok", "yazmaç", "paylaşımlı bellek"], lim = [byThr, byBlk, byReg, bySm];
  return { b, warps: b * w, maxW: a.thr / 32, occ: (b * w) / (a.thr / 32), limit: names[lim.indexOf(b)], lim };
}
export default {
  params: [
    { id: "arch", label: "Mimari", options: ARCH.map((a) => a.n), value: 4 },
    { id: "T", label: "Blok başına thread", min: 32, max: 1024, step: 32, value: 256, int: true },
    { id: "R", label: "Thread başına yazmaç", min: 8, max: 255, value: 32, int: true },
    { id: "S", label: "Blok başına paylaşımlı bellek (KB)", min: 0, max: 96, step: 0.5, value: 8 },
  ],
  validate(q) { const a = ARCH[q.arch], e = []; if (q.T > a.tpb) e.push({ level: "error", msg: `Blok başına thread: ${a.n} en çok ${a.tpb} thread'e izin verir.` }); if (q.S > a.smem) e.push({ level: "error", msg: `Blok başına paylaşımlı bellek: ${a.n}'de SM başına ${a.smem} KB var.` }); if (q.R > a.maxR) e.push({ level: "warn", msg: `Thread başına yazmaç ${a.maxR}'i aşıyor: derleyici fazlasını yerel belleğe taşır (spill).` }); return e; },
  run(q) {
    const a = ARCH[q.arch], o = occ(a, q.T, Math.min(q.R, a.maxR), q.S);
    const Rs = range(31).map((k) => 8 + k * 8).filter((r) => r <= a.maxR), Ts = range(32).map((k) => 32 * (k + 1)).filter((t) => t <= a.tpb);
    const P = [`${a.n}: SM'de ${o.b} blok, ${o.warps}/${o.maxW} warp → doluluk ${fmt(100 * o.occ, 3)}%. Sınırlayan: ${o.limit} (thread ${o.lim[0]}, blok ${o.lim[1]}, yazmaç ${Number.isFinite(o.lim[2]) ? o.lim[2] : "—"}, paylaşımlı bellek ${Number.isFinite(o.lim[3]) ? o.lim[3] : "—"} blok).`];
    if (o.b === 0) P.push("Bu yapılandırmada SM'ye tek bir blok bile sığmıyor: çekirdek başlatılamaz.");
    else if (o.occ < 0.5) P.push("Doluluk düşük: bellek gecikmesini gizleyecek hazır warp az olabilir. Yine de doluluk tek başına hız değildir; az warp'la çok yazmaç kullanan (ILP'li) çekirdekler de hızlı olabilir.");
    return {
      plots: [
        { title: "Doluluk ve thread başına yazmaç", xlabel: "yazmaç / thread", ylabel: "doluluk", side: true, size: { w: 300, h: 230 }, ylim: [0, 1.05], series: [{ name: "doluluk", x: Rs, y: Rs.map((r) => occ(a, q.T, r, q.S).occ), kind: "step", c: 0 }], marks: [{ x: q.R, label: "seçili" }] },
        { title: "Doluluk ve blok boyu", xlabel: "thread / blok", ylabel: "doluluk", side: true, size: { w: 300, h: 230 }, ylim: [0, 1.05], series: [{ name: "doluluk", x: Ts, y: Ts.map((t) => occ(a, t, Math.min(q.R, a.maxR), q.S).occ), c: 2, dots: true }], marks: [{ x: q.T, label: "seçili" }] },
      ],
      stats: [{ label: "Doluluk", value: `${fmt(100 * o.occ, 3)}%` }, { label: "SM başına blok / warp", value: `${o.b} / ${o.warps}` }, { label: "Sınırlayan kaynak", value: o.limit }],
      note: P.join(" "),
    };
  },
};
