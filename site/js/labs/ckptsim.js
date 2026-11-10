// M6.4 — Aktivasyon checkpointing benzetimi (Chen ve ark. 2016). n katmanlı zincir, her katmanın çıktısı 1 birim (ya da "transformer
// benzeri": dikkat katmanları 3 birim). Her k katmanda bir segment; ileri geçişte yalnız segment sonları saklanır. Geri geçiş segmentleri
// sondan başa işler: segmentin iç çıktılarını yeniden hesaplar, sonra katman katman geri gider ve çıktıları serbest bırakır.
// Tepe = max_j [segment 0..j sonları + segment j'nin yeniden hesaplanan iç çıktıları] (aşama b'deki tanım). k = 1: checkpoint yok.
import { fmt, range } from "./plot.js";

export function sizesOf(q) { return range(q.n).map((i) => (q.kind === 1 && i % 2 === 0 ? 3 : 1)); }
export function simulate(sizes, k) {
  const n = sizes.length, segs = []; for (let s = 0; s < n; s += k) segs.push(range(Math.min(k, n - s)).map((i) => s + i));
  const keep = new Set(segs.map((sg) => sg.at(-1))); if (k === 1) range(n).forEach((i) => keep.add(i));
  const tl = []; let mem = 0;
  for (let i = 0; i < n; i++) { if (keep.has(i)) mem += sizes[i]; tl.push(mem); }      // ileri
  let rec = 0, peak = Math.max(...tl, 0);
  for (let j = segs.length - 1; j >= 0; j--) {
    const sg = segs[j];
    if (k > 1) for (const i of sg.slice(0, -1)) { mem += sizes[i]; rec++; tl.push(mem); peak = Math.max(peak, mem); }  // yeniden hesap
    for (let t = sg.length - 1; t >= 0; t--) { mem -= sizes[sg[t]]; tl.push(mem); }      // geri geçiş: çıktıyı serbest bırak
  }
  return { tl, peak, rec, segs: segs.length };
}
export default {
  params: [
    { id: "n", label: "Katman sayısı n", min: 2, max: 128, value: 16, int: true, log: true },
    { id: "k", label: "Segment uzunluğu k (1 = checkpoint yok)", min: 1, max: 128, value: 4, int: true, log: true },
    { id: "kind", label: "Katman çıktı boyutları", options: ["hepsi 1 birim", "dönüşümlü 3 ve 1 birim (dikkat + MLP benzeri)"], value: 0 },
    { id: "unit", label: "Bir birim (MiB)", min: 1, max: 4096, value: 64, int: true, log: true },
  ],
  run(q) {
    const sizes = sizesOf(q), k = Math.min(q.k, q.n), S = simulate(sizes, k), N = simulate(sizes, 1), total = sizes.reduce((s, v) => s + v, 0);
    const ks = range(q.n).map((i) => i + 1), sw = ks.map((kk) => simulate(sizes, kk)), best = ks[sw.reduce((bi, s, i) => (s.peak < sw[bi].peak ? i : bi), 0)];
    const P = [`n = ${q.n}, k = ${k}: ${S.segs} segment. Tepe bellek ${S.peak} birim (${fmt(S.peak * q.unit, 4)} MiB); checkpoint'siz ${N.peak} birim. Yeniden hesaplanan katman sayısı ${S.rec} (ileri geçişin %${fmt((100 * S.rec) / q.n, 3)} kadarı).`];
    P.push(`Kuram: tek tip katmanlarda tepe ≈ n/k + k, en küçüğü k = √n = ${fmt(Math.sqrt(q.n), 3)}'te ≈ 2√n. Bu ayarlarda en küçük tepe k = ${best}'te ${sw[best - 1].peak} birim.`);
    if (k === 1) P.push("k = 1: her çıktı saklanır, yeniden hesap yok. Bellek n ile doğrusal.");
    if (k >= q.n) P.push("k = n: yalnız son çıktı saklanır ama geri geçişte tüm zincir yeniden hesaplanır ve bu sırada hepsi bellekte durur: tepe yine ≈ n. Tek bir dev segment işe yaramaz.");
    if (q.kind === 1) P.push("Katman boyutları farklıyken segment sınırlarının nereye düştüğü de önemlidir: büyük çıktıları segment sonu yapmamak (onları yeniden hesaplamak) daha az bellek harcar. Seçici yeniden hesaplama (Korthikanti ve ark. 2022) yalnız pahalı olmayan ama büyük dikkat ara sonuçlarını yeniden hesaplar.");
    return {
      plots: [
        { title: "Bellek zaman çizelgesi (ileri, sonra geri)", xlabel: "adım", ylabel: "saklı birim", yzero: true, series: [
          { name: "checkpoint'siz", x: range(N.tl.length), y: N.tl, c: 4, dash: true }, { name: `k = ${k}`, x: range(S.tl.length), y: S.tl, c: 0 }] },
        { title: "Tepe bellek ve yeniden hesap, k'ya göre", xlabel: "k", ylabel: "birim / katman", yzero: true, series: [
          { name: "tepe bellek", x: ks, y: sw.map((s) => s.peak), c: 0 }, { name: "yeniden hesaplanan katman", x: ks, y: sw.map((s) => s.rec), c: 1 }],
          marks: [{ x: Math.sqrt(q.n), label: "√n" }, { x: k, label: "k" }] },
      ],
      stats: [
        { label: "Tepe: checkpoint'li / checkpoint'siz (birim)", value: `${S.peak} / ${N.peak}` },
        { label: "Yeniden hesaplanan katman", value: S.rec },
        { label: "En iyi k (bu ayarlarda)", value: best },
        { label: "Toplam çıktı (birim)", value: total },
      ],
      note: P.join(" "),
    };
  },
};
