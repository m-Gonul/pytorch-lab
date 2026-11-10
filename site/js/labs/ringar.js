// M7.4 — Ring all-reduce adım adım (Patarasuk–Yuan; Baidu/Gibiansky). N işçi, vektör N parçaya bölünmüş. Reduce-scatter, adım s:
// işçi i parça (i − s) mod N'yi i + 1'e gönderir, alıcı ekler. N − 1 adım sonra işçi i, parça (i + 1) mod N'nin tam toplamını tutar.
// All-gather, adım s: işçi i parça (i + 1 − s) mod N'yi gönderir, alıcı üzerine yazar. Toplam 2(N − 1) adım; işçi başına gönderilen
// 2(N − 1)/N · boyut. Süre modeli (α–β): T = adım sayısı · α + gönderilen bayt / bant. Karşılaştırma: merkezi sunucu (hepsi bir işçiye
// gönderir, o toplayıp geri yollar): sunucunun bağlantısı 2(N − 1) · boyut taşır.
import { fmt, range } from "./plot.js";

export function simulate(N, upto) {
  // durum[i][c] = işçi i'nin parça c'sinde toplanmış katkıların kümesi (bit maskesi)
  let st = range(N).map((i) => range(N).map(() => 1 << i)); const sent = new Array(N).fill(0); let step = 0;
  for (let s = 0; s < N - 1 && step < upto; s++, step++) { const msg = range(N).map((i) => [(i + 1) % N, ((i - s) % N + N) % N, st[i][((i - s) % N + N) % N]]); msg.forEach(([to, c, v], i) => { st[to][c] |= v; sent[i]++; }); }
  for (let s = 0; s < N - 1 && step < upto; s++, step++) { const msg = range(N).map((i) => [(i + 1) % N, ((i + 1 - s) % N + N) % N, st[i][((i + 1 - s) % N + N) % N]]); msg.forEach(([to, c, v], i) => { st[to][c] = v; sent[i]++; }); }
  return { st, sent, step };
}
const pop = (v) => { let c = 0; while (v) { c += v & 1; v >>= 1; } return c; };
export default {
  params: [
    { id: "N", label: "İşçi sayısı N", min: 2, max: 16, value: 4, int: true },
    { id: "step", label: "Gösterilen adım (0 … 2(N − 1))", min: 0, max: 30, value: 3, int: true },
    { id: "MB", label: "Gradyan boyutu (MB)", min: 1, max: 10000, value: 100, log: true },
    { id: "alpha", label: "Mesaj başına gecikme α (µs)", min: 1, max: 1000, value: 10, log: true },
    { id: "bw", label: "Bağlantı bant genişliği (GB/s)", min: 1, max: 900, value: 25, log: true },
  ],
  validate(q) { return q.step > 2 * (q.N - 1) ? [{ level: "warn", msg: `N = ${q.N} için en fazla ${2 * (q.N - 1)} adım var; son durum gösteriliyor.` }] : []; },
  run(q) {
    const total = 2 * (q.N - 1), s = Math.min(q.step, total), S = simulate(q.N, s), full = simulate(q.N, total);
    const img = S.st.flatMap((row) => row.map((v) => 0.95 - (0.8 * (pop(v) - 1)) / (q.N - 1)));
    const perWorkerMB = (2 * (q.N - 1) / q.N) * q.MB, ringT = total * q.alpha * 1e-6 + perWorkerMB / 1e3 / q.bw, psT = 2 * q.alpha * 1e-6 + (2 * (q.N - 1) * q.MB) / 1e3 / q.bw;
    const Ns = range(15).map((i) => i + 2), ring = Ns.map((n) => 2 * (n - 1) * q.alpha * 1e-6 + ((2 * (n - 1)) / n) * q.MB / 1e3 / q.bw), ps = Ns.map((n) => 2 * q.alpha * 1e-6 + (2 * (n - 1) * q.MB) / 1e3 / q.bw);
    const done = full.st.every((row) => row.every((v) => pop(v) === q.N));
    const phase = s === 0 ? "başlangıç: her işçi yalnız kendi gradyanını tutuyor" : s <= q.N - 1 ? `reduce-scatter, adım ${s} / ${q.N - 1}` : `all-gather, adım ${s - (q.N - 1)} / ${q.N - 1}`;
    const P = [`${phase}. Görüntüde satır = işçi, sütun = parça, koyuluk = o parçada toplanmış katkı sayısı (siyah: tam toplam). ${s === q.N - 1 ? "Reduce-scatter bitti: her işçi tam olarak bir parçanın tam toplamını tutuyor (köşegen kaydırılmış)." : ""}${s === total ? " All-gather bitti: bütün işçiler bütün parçaların tam toplamını tutuyor." : ""}`];
    P.push(`Tam algoritma ${total} adım; işçi başına ${full.sent[0]} parça (= ${fmt(perWorkerMB, 4)} MB) gönderilir ve bu, N büyüdükçe 2 · ${q.MB} MB'ye yaklaşır: bant genişliği maliyeti N'den neredeyse bağımsızdır. ${done ? "Sonuçta herkes tam toplamı tutuyor (doğrulandı)." : ""}`);
    P.push(`α–β tahmini: ring ${fmt(ringT * 1e3, 4)} ms, merkezi sunucu ${fmt(psT * 1e3, 4)} ms. ${ringT < psT ? "Bant terimi baskın: ring kazanıyor." : "<b>Gecikme terimi baskın</b>: ring'in 2(N − 1) adımlık gecikmesi, merkezi yaklaşımın bant tasarrufunu aşıyor."} Küçük mesajlarda ve çok sayıda işçide ağaç tabanlı algoritmalar (log N adım) tercih edilir; NCCL ikisi arasında seçer.`);
    return {
      plots: [
        { title: `Parça durumları, adım ${s}`, xlabel: "parça", ylabel: "işçi", size: { w: 300, h: 280 }, side: true, img: { w: q.N, h: q.N, data: img }, series: [] },
        { title: "Süre tahmini ve işçi sayısı", xlabel: "N", ylabel: "ms", series: [{ name: "ring", x: Ns, y: ring.map((t) => t * 1e3), c: 0 }, { name: "merkezi sunucu", x: Ns, y: ps.map((t) => t * 1e3), c: 1 }], marks: [{ x: q.N, label: "N" }] },
      ],
      stats: [
        { label: "Toplam adım", value: total },
        { label: "İşçi başına gönderilen (MB)", value: perWorkerMB },
        { label: "Süre: ring / merkezi (ms)", value: `${fmt(ringT * 1e3, 4)} / ${fmt(psT * 1e3, 4)}` },
      ],
      note: P.join(" "),
    };
  },
};
