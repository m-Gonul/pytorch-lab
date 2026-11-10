// M10.7 — Yazmaç karolamalı fp32 matmul (SIMT, A100 benzeri) için kaba başarım modeli. Blok C'nin B × B karosunu hesaplar, k boyunca
// BK'lık dilimlerle A ve B karolarını paylaşımlı belleğe yükler; her thread TM × TN çıktıyı yazmaçta biriktirir (thread = B²/(TM·TN)).
// Tavanlar: (1) tepe FMA hızı × gecikme gizleme (etkin warp × ILP eşiği), (2) DRAM: blok düzeyi yoğunluk B/4 FLOP/bayt × 1.5 TB/s,
// (3) paylaşımlı bellek: thread her k adımında TM + TN öğe okuyup TM·TN FMA yapar → 2·TM·TN/(4(TM+TN)) FLOP/bayt × 19.5 TB/s,
// (4) bariyer kaybı: tek tamponda her BK diliminde yükle–bariyer–hesapla sıralıdır, açıkta kalan pay ≈ 0.3·(8/BK)/(SM'deki blok);
// çift tamponda 0.05·(8/BK)/(blok). (5) dalga nicemlemesi: (n/B)² blok, 108 SM × SM başına blok. Yazmaç ≈ TM·TN + TM + TN + 24 (adres, sayaç); 255'i aşarsa taşma.
import { fmt, range } from "./plot.js";

const PK = 19.5e12, BW = 1.5e12, SBW = 19.5e12, NSM = 108;
const BS = [16, 32, 64, 128, 256], TS = [1, 2, 4, 8, 16], BKS = [8, 16, 32];
export function model(B, BK, TM, TN, dbuf, n) {
  const thr = (B * B) / (TM * TN), regs = TM * TN + TM + TN + 24, smem = 2 * B * BK * 4 * (dbuf ? 2 : 1);
  const spill = regs > 255, r = Math.min(regs, 255), w = Math.ceil(thr / 32);
  const byThr = Math.floor(2048 / (w * 32)), byReg = Math.floor(65536 / (w * Math.ceil((r * 32) / 256) * 256)), bySm = Math.floor((164 * 1024) / smem);
  const blk = Math.max(0, Math.min(byThr, 32, byReg, bySm)), warps = blk * w;
  const ilp = Math.min(TM * TN, 8) * (dbuf ? 1.5 : 1), lat = Math.min(1, (warps * ilp) / 32);
  const cPeak = PK * lat * (spill ? 0.35 : 1), cDram = (B / 4) * BW, cSmem = (SBW * 2 * TM * TN) / (4 * (TM + TN));
  const tiles = Math.ceil(n / B) ** 2, slots = NSM * Math.max(blk, 1), waves = Math.ceil(tiles / slots), wq = tiles / (waves * slots);
  const sync = blk ? 1 - ((db_ => (db_ ? 0.05 : 0.3))(dbuf) * (8 / BK)) / blk : 0, est = Math.min(cPeak, cDram, cSmem) * sync * wq;
  const lim = [["tepe × gecikme gizleme", cPeak], ["DRAM bant genişliği", cDram], ["paylaşımlı bellek bant genişliği", cSmem]].reduce((a, b) => (b[1] < a[1] ? b : a))[0];
  return { thr, regs, smem, spill, sync, blk, warps, lat, cPeak, cDram, cSmem, tiles, waves, wq, est, lim };
}
export default {
  params: [
    { id: "bt", label: "Blok karosu B × B", options: BS.map((b) => `${b} × ${b}`), value: 3 },
    { id: "BK", label: "k dilimi BK", options: BKS.map(String), value: 0 },
    { id: "TM", label: "Thread karosu TM (satır)", options: TS.map(String), value: 3 },
    { id: "TN", label: "Thread karosu TN (sütun)", options: TS.map(String), value: 3 },
    { id: "db", label: "Paylaşımlı bellek tamponu", options: ["tek", "çift (yükleme ile hesap örtüşür)"], value: 0 },
    { id: "logn", label: "Matris boyu n = 2^k", min: 9, max: 13, value: 12, int: true },
  ],
  validate(q) {
    const B = BS[q.bt], TM = TS[q.TM], TN = TS[q.TN], e = [];
    if (TM > B || TN > B) return [{ level: "error", msg: "Thread karosu: blok karosundan büyük olamaz." }];
    const m = model(B, BKS[q.BK], TM, TN, q.db === 1, 2 ** q.logn);
    if (m.thr > 1024) e.push({ level: "error", msg: `Blok karosu: ${m.thr} thread gerekir, en çok 1024 olabilir (thread karosunu büyüt).` });
    else if (m.thr < 32) e.push({ level: "warn", msg: `Blokta yalnız ${m.thr} thread: warp'ın bir kısmı boş kalır.` });
    if (m.smem > 164 * 1024) e.push({ level: "error", msg: `Paylaşımlı bellek tamponu: blok ${fmt(m.smem / 1024, 3)} KB ister, SM'de 164 KB var.` });
    if (m.thr <= 1024 && m.smem <= 164 * 1024 && m.blk === 0) e.push({ level: "error", msg: `Thread karosu: ${m.thr} thread × ${Math.min(m.regs, 255)} yazmaç = ${m.thr * Math.min(m.regs, 255)} > 65536: SM'ye tek blok sığmaz.` });
    if (m.spill) e.push({ level: "warn", msg: `Thread karosu: ≈ ${m.regs} yazmaç gerekir (en çok 255): fazlası yerel belleğe taşar, model tepe hızı 0.35 ile çarpar.` });
    return e;
  },
  run(q) {
    const B = BS[q.bt], BK = BKS[q.BK], TM = TS[q.TM], TN = TS[q.TN], db = q.db === 1, n = 2 ** q.logn, m = model(B, BK, TM, TN, db, n);
    const T = (v) => v / 1e12;
    const curve = (t) => BS.map((b) => (t > b || (b * b) / (t * t) > 1024 || 2 * b * BK * 4 * (db ? 2 : 1) > 164 * 1024 ? NaN : ((m_) => (m_.blk === 0 ? NaN : T(m_.est)))(model(b, BK, t, t, db, n))));
    const P = [`B = ${B}, BK = ${BK}, thread karosu ${TM} × ${TN}: blokta ${m.thr} thread, thread başına ≈ ${m.regs} yazmaç, ${fmt(m.smem / 1024, 3)} KB paylaşımlı bellek; SM'de ${m.blk} blok (${m.warps} warp). Tavanlar: tepe × gecikme ${fmt(T(m.cPeak), 3)}, DRAM ${fmt(T(m.cDram), 3)}, paylaşımlı bellek ${fmt(T(m.cSmem), 3)} TFLOP/s; sınırlayan ${m.lim}.`];
    P.push(`Bariyer verimi ${fmt(100 * m.sync, 3)}% (${db ? "çift" : "tek"} tampon, BK = ${BK}, SM'de ${m.blk} blok). n = ${n}: ${m.tiles} blok, ${m.waves} dalga, dalga verimi ${fmt(100 * m.wq, 3)}% → tahmini ${fmt(T(m.est), 3)} TFLOP/s (tepenin ${fmt((100 * m.est) / PK, 3)}%'i).`);
    if (TM * TN === 1) P.push("Thread karosu 1 × 1: her FMA için iki paylaşımlı bellek okuması (8 bayt / 2 FLOP); paylaşımlı bellek tepe hızın dörtte birinde tıkanır.");
    if (m.wq < 0.8) P.push("Büyük blok karosu küçük matriste az blok demek: son dalga SM'lerin çoğunu boş bırakır (split-K ya da daha küçük karo).");
    P.push("Sınır: kaba bir tavan modeli; komut gönderme, bank çakışması, L2 isabeti ve Tensor Core yoktur. Gerçek SGEMM'ler (cuBLAS) bu donanımda tepenin %80–90'ına ulaşır.");
    return {
      plots: [
        { title: "Tavanlar (TFLOP/s)", xlabel: "", ylabel: "TFLOP/s", side: true, size: { w: 300, h: 230 }, xlim: [-0.6, 3.6], ylim: [0, Math.max(T(m.cPeak), T(m.cDram), T(m.cSmem), T(PK)) * 1.1], xticks: [[0, "tepe×gec."], [1, "DRAM"], [2, "paylaşımlı"], [3, "tahmin"]],
          series: [{ name: "tavan", x: [0, 1, 2, 3], y: [m.cPeak, m.cDram, m.cSmem, NaN].map((v) => (Number.isNaN(v) ? v : T(v))), kind: "bar", c: 0, bw: 0.45 }, { name: "tahmin (dalga dahil)", x: [0, 1, 2, 3], y: [NaN, NaN, NaN, T(m.est)], kind: "bar", c: 2, bw: 0.15 }], marks: [{ y: T(PK), label: "fp32 tepe" }] },
        { title: "Tahmin ve blok karosu (kare thread karoları)", xlabel: "blok karosu B", ylabel: "TFLOP/s", side: true, size: { w: 300, h: 230 }, ylim: [0, 20.5], xlim: [-0.3, 4.3], xticks: BS.map((b, i) => [i, String(b)]),
          series: [1, 4, 8].map((t, i) => ({ name: `${t} × ${t}`, x: [0, 1, 2, 3, 4], y: curve(t), c: i, dots: true })), marks: [{ x: q.bt, label: "seçili" }] },
      ],
      stats: [{ label: "Tahmini hız", value: `${fmt(T(m.est), 3)} TFLOP/s` }, { label: "Sınırlayan", value: m.lim }, { label: "SM başına blok / warp", value: `${m.blk} / ${m.warps}` }, { label: "Bariyer / dalga verimi", value: `${fmt(100 * m.sync, 3)}% / ${fmt(100 * m.wq, 3)}%` }],
      note: P.join(" "),
    };
  },
};
