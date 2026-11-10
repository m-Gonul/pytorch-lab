// M7.2 — Triton'un blok programları: grid ve maskeler. Vektör: cdiv(N, BLOCK) program, program p'nin ofsetleri p·BLOCK + arange(BLOCK),
// maske ofs < N. Matmul: (cdiv(M, BM), cdiv(N, BN)) program, her biri cdiv(K, BK) adımda A ve B tile'larını maskeli yükler (other = 0).
// Maske verimi = gerçek eleman / işlenen şerit. Görüntü: C'nin kapsanması (açık/koyu: komşu tile'lar, siyah: maskelenen şeritler).
import { fmt, range } from "./plot.js";

const POW = [8, 16, 32, 64, 128, 256, 512, 1024];
export default {
  params: [
    { id: "op", label: "Çekirdek", options: ["vektör toplama (1-B)", "matris çarpımı (2-B grid)"], value: 1 },
    { id: "M", label: "M (vektörde N)", min: 1, max: 2048, value: 70, int: true, log: true },
    { id: "N", label: "N (C'nin sütun sayısı)", min: 1, max: 2048, value: 40, int: true, log: true, on: { op: [1] } },
    { id: "K", label: "K (iç boyut)", min: 1, max: 2048, value: 50, int: true, log: true, on: { op: [1] } },
    { id: "B", label: "BLOCK / BM = BN", options: POW.map(String), value: 1 },
    { id: "BK", label: "BK", options: POW.map(String), value: 1, on: { op: [1] } },
  ],
  run(q) {
    const B = POW[q.B], cd = (a, b) => Math.ceil(a / b);
    if (q.op === 0) {
      const progs = cd(q.M, B), lanes = progs * B, eff = q.M / lanes, valid = q.M - (progs - 1) * B;
      const effs = POW.map((b) => q.M / (cd(q.M, b) * b));
      return {
        plots: [{ title: "Maske verimi ve BLOCK", xlabel: "BLOCK (log₂)", ylabel: "gerçek / işlenen", ylim: [0, 1.05], series: [{ name: "verim", x: POW.map(Math.log2), y: effs, c: 0 }], marks: [{ x: Math.log2(B), label: "BLOCK" }], xticks: POW.map((b) => [Math.log2(b), String(b)]) }],
        stats: [{ label: "Program sayısı", value: progs }, { label: "Son programda geçerli eleman", value: valid }, { label: "Maske verimi", value: eff }],
        note: `N = ${q.M}, BLOCK = ${B}: grid = cdiv(N, BLOCK) = ${progs} program. Son program ${valid} geçerli eleman işler; maskeli ${B - valid} şerit yükleme ve yazma yapmaz. Maske verimi %${fmt(100 * eff, 3)}. BLOCK büyüdükçe program başına ek yük azalır ama son programdaki boş şeritler artabilir; Triton'da BLOCK bir derleme zamanı sabitidir (tl.constexpr) ve 2'nin kuvveti olmalıdır.`,
      };
    }
    const BK = POW[q.BK], gm = cd(q.M, B), gn = cd(q.N, B), ks = cd(q.K, BK), progs = gm * gn, eff = (q.M * q.N) / (progs * B * B), effK = q.K / (ks * BK);
    const H = gm * B, Wd = gn * B, sc = Math.max(1, Math.ceil(Math.max(H, Wd) / 128)), h = Math.ceil(H / sc), w = Math.ceil(Wd / sc);
    const img = range(h * w).map((k) => { const i = Math.floor(k / w) * sc, j = (k % w) * sc; if (i >= q.M || j >= q.N) return 0; return (Math.floor(i / B) + Math.floor(j / B)) % 2 ? 0.55 : 0.85; });
    const flops = 2 * q.M * q.N * q.K, done = 2 * progs * B * B * ks * BK;
    return {
      plots: [{ title: `C'nin kapsanması (${gm} × ${gn} program; siyah: maskeli)`, xlabel: "sütun", ylabel: "satır", size: { w: 320, h: 300 }, side: true, img: { w, h, data: img }, series: [] },
        { title: "Maske verimi ve BM = BN", xlabel: "BM (log₂)", ylabel: "gerçek / işlenen", ylim: [0, 1.05], series: [{ name: "C tile'ları", x: POW.map(Math.log2), y: POW.map((b) => (q.M * q.N) / (cd(q.M, b) * cd(q.N, b) * b * b)), c: 0 }], marks: [{ x: Math.log2(B), label: "BM" }], xticks: POW.map((b) => [Math.log2(b), String(b)]) }],
      stats: [{ label: "Program sayısı (grid)", value: `${gm} × ${gn} = ${progs}` }, { label: "Program başına K adımı", value: ks }, { label: "Maske verimi: C / K", value: `${fmt(eff, 3)} / ${fmt(effK, 3)}` }, { label: "Yapılan / gereken FLOP", value: done / flops }],
      note: `C (${q.M} × ${q.N}) = A (${q.M} × ${q.K}) · B (${q.K} × ${q.N}), BM = BN = ${B}, BK = ${BK}: grid (cdiv(M, BM), cdiv(N, BN)) = (${gm}, ${gn}) → ${progs} program; her biri K boyunca ${ks} adım tl.dot yapar. Kenar tile'larında maskeli yüklemeler other = 0 verir: sıfırlar toplama katkı yapmaz, sonuç doğru kalır; ama tl.dot dolgu dahil bütün tile'ı hesaplar: yapılan FLOP gerekenin ${fmt(done / flops, 3)} katı. Kenarlarda bu israf küçük matrislerde belirgindir; büyük matrislerde kaybolur.`,
    };
  },
};
