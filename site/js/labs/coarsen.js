// M10.5 — Hiyerarşik indirgemede thread kabalaştırma (coarsening). N öğe, T thread'lik bloklar, kabalaştırma çarpanı C: her blok
// 2·T·C öğe işler (her thread 2C öğeyi önce yazmaçta toplar), sonra paylaşımlı bellekte log₂T adımlık ağaç ve bir atomicAdd.
// Maliyet modeli (çevrim, A100 benzeri: 108 SM, SM başına 2048 thread, global bant genişliği ~1000 bayt/çevrim): bir bloğun süresi =
// 2C (yükleme ve toplama, thread başına) + log₂T · 30 (her ağaç adımı: eşzamanlama + paylaşımlı bellek) + 500 (blok başlatma ve atomik);
// aynı anda SM·(2048/T) blok çalışır (dalgalar); toplam süre ≥ 4N / (1000 · doluluk), doluluk = min(1, blok / kapasite). Ayrıca atomicAdd'ler tek adreste sıraya girer
// (her biri 10 çevrim).
import { fmt, range } from "./plot.js";

export function model(N, T, C, sm = 108) {
  const blocks = Math.ceil(N / (2 * T * C)), conc = sm * Math.floor(2048 / T), waves = Math.ceil(blocks / conc);
  // Bant genişliğini doldurmak için yeterli sayıda blok (thread) yerleşik olmalı: tek dalgada kapasitenin ancak blocks/conc'u dolu.
  const fill = Math.min(1, blocks / conc), tBlock = 2 * C + Math.log2(T) * 30 + 500, comp = waves * tBlock, bw = (4 * N) / (1000 * fill), atom = blocks * 10;
  return { blocks, conc, waves, T: Math.max(comp, bw) + atom, comp, bw, atom, tBlock };
}
export default {
  params: [
    { id: "logN", label: "Öğe sayısı N = 2^k", min: 16, max: 28, value: 24, int: true },
    { id: "T", label: "Blok başına thread", options: ["128", "256", "512", "1024"], value: 1 },
    { id: "logC", label: "Kabalaştırma çarpanı C = 2^k", min: 0, max: 10, value: 0, int: true },
    { id: "sm", label: "SM sayısı", min: 8, max: 144, value: 108, int: true },
  ],
  run(q) {
    const N = 2 ** q.logN, T = [128, 256, 512, 1024][q.T], C = 2 ** q.logC, m = model(N, T, C, q.sm);
    const Cs = range(11).map((k) => 2 ** k), ts = Cs.map((c) => model(N, T, c, q.sm).T), best = Cs[ts.indexOf(Math.min(...ts))];
    const P = [`N = ${N}, T = ${T}, C = ${C}: ${m.blocks} blok, aynı anda ${m.conc} blok çalışabiliyor → ${m.waves} dalga. Süre ≈ ${fmt(m.T, 4)} çevrim (hesap/ek yük ${fmt(m.comp, 4)}, bant genişliği alt sınırı ${fmt(m.bw, 4)}, atomikler ${fmt(m.atom, 4)}).`];
    P.push(`Bu boyutta en iyi C = ${best}. ${C < best ? "C küçük: çok sayıda blok başlatılıyor, her blok ağaç adımlarının ve atomik işlemin sabit maliyetini ödüyor." : C > best ? "C büyük: blok sayısı eşzamanlı kapasitenin altına düşüyor, SM'ler boş kalıyor." : "Bu C, blok başına sabit maliyetlerle donanımı doldurmak arasında en iyi denge."}`);
    if (m.blocks < m.conc) P.push(`Blok sayısı (${m.blocks}) eşzamanlı kapasitenin (${m.conc}) altında: donanımın bir kısmı boş.`);
    return {
      plots: [{ title: "Süre ve kabalaştırma çarpanı", xlabel: "log₂ C", ylabel: "çevrim", wide: true, ylog: true, size: { w: 620, h: 230 }, series: [{ name: "toplam süre", x: range(11), y: ts, c: 0, dots: true }, { name: "tam dolulukta bant genişliği sınırı", x: [0, 10], y: [(4 * N) / 1000, (4 * N) / 1000], c: 4, dash: true }], marks: [{ x: q.logC, label: "seçili C" }] }],
      stats: [{ label: "Blok / eşzamanlı kapasite", value: `${m.blocks} / ${m.conc}` }, { label: "Süre (çevrim)", value: fmt(m.T, 4) }, { label: "En iyi C", value: best }],
      note: P.join(" "),
    };
  },
};
