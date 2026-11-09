// M4.12 — Zipf dağılımlı bir sözlükte (f_r ∝ r^−s) iki word2vec ayarı: negatif örneklerin gürültü dağılımı P_n ∝ f^α ve sık
// kelimelerin alt-örneklemesi (makale: atılma 1 − √(t/f); word2vec.c: tutma (√(f/t) + 1)·t/f). Tüm sayılar V kelimenin hepsi
// üzerinden hesaplanır; grafikler log-aralıklı sıralarda çizilir.
import { fmt, range } from "./plot.js";

export function zipf(V, s) { const w = range(V).map((i) => (i + 1) ** -s), z = w.reduce((a, b) => a + b, 0); return w.map((x) => x / z); }
export function keepProb(f, t, form) {
  if (form === 0) return 1;
  if (form === 1) return Math.min(1, Math.sqrt(t / f));
  return Math.min(1, Math.sqrt(t / f) + t / f);
}
const share = (p, a, b) => p.slice(a, b).reduce((x, y) => x + y, 0);

export default {
  params: [
    { id: "V", label: "Sözlük boyu V", min: 100, max: 100000, value: 10000, int: true, log: true },
    { id: "s", label: "Zipf üssü s (f ∝ sıra^−s)", min: 0.5, max: 1.5, step: 0.05, value: 1 },
    { id: "alpha", label: "Gürültü üssü α (P_n ∝ f^α)", min: 0, max: 1, step: 0.05, value: 0.75 },
    { id: "form", label: "Alt-örnekleme", options: ["kapalı", "makale: atılma 1 − √(t/f)", "word2vec.c: tutma √(t/f) + t/f"], value: 1 },
    { id: "tl", label: "Eşik log₁₀ t", min: -6, max: -2, step: 0.1, value: -5, on: { form: [1, 2] } },
  ],
  validate(q) { return q.form && 10 ** q.tl > 1 / Math.min(q.V, 10) ? [{ level: "warn", msg: "t, en sık kelimelerin frekansından büyük: alt-örnekleme hemen hiçbir şeyi atmıyor." }] : []; },
  run(q) {
    const f = zipf(q.V, q.s), t = 10 ** q.tl, pa = f.map((x) => x ** q.alpha), za = pa.reduce((a, b) => a + b, 0), pn = pa.map((x) => x / za);
    const keep = f.map((x) => keepProb(x, t, q.form)), fk = f.map((x, i) => x * keep[i]), kept = fk.reduce((a, b) => a + b, 0), fs = fk.map((x) => x / kept);
    const tail = Math.min(1000, q.V - 1);
    const idx = [...new Set(range(160).map((i) => Math.min(q.V - 1, Math.round(Math.exp((Math.log(q.V) * i) / 159)) - 1)))];
    const lx = idx.map((i) => Math.log10(i + 1)), xt = range(Math.floor(Math.log10(q.V)) + 1).map((e) => [e, e < 3 ? String(10 ** e) : e === 3 ? "1K" : e === 4 ? "10K" : "100K"]);
    const top10 = [share(f, 0, 10), share(pn, 0, 10), share(fs, 0, 10)], tl = [share(f, tail, q.V), share(pn, tail, q.V), share(fs, tail, q.V)];
    const firstCut = keep.findIndex((k) => k >= 1); // ilk tam tutulan sıra
    const P = [];
    P.push(`En sık 10 kelimenin metindeki token payı %${fmt(100 * top10[0], 3)}, α = ${q.alpha} ile negatif örneklerdeki payı %${fmt(100 * top10[1], 3)}${q.alpha < 1 ? ` (α = 1 olsaydı %${fmt(100 * top10[0], 3)} olurdu)` : ""}. ${tail < q.V - 1 ? `Sırası ${tail}'den büyük kuyruğun token payı %${fmt(100 * tl[0], 3)}, negatiflerdeki payı %${fmt(100 * tl[1], 3)}.` : ""} α küçüldükçe dağılım düzleşir: α = 0 her kelimeye 1/V verir (sık kelimeler neredeyse hiç ceza almaz), α = 1 ham frekanstır (nadir kelimeler hemen hiç negatif olmaz).`);
    if (q.form) {
      const disc = 1 - kept;
      P.push(`Alt-örnekleme (t = ${fmt(t, 2)}) ile atılan token oranı %${fmt(100 * disc, 3)}; en sık kelimenin tutulma olasılığı ${fmt(keep[0], 3)}. ${firstCut > 0 ? `Sırası ${firstCut + 1} ve sonrası olan kelimeler (f ≤ ${q.form === 1 ? "t" : "φ²·t ≈ 2.618 t"}) tam tutuluyor.` : firstCut === 0 ? "Hiçbir kelime atılmıyor (en sık kelimenin frekansı bile eşiğin altında)." : "Sözlüğün tamamı kısmen atılıyor."} Atılmayan kısımda f·tutma ${q.form === 1 ? "= √(t·f)" : "= √(t·f) + t"}: sık kelimelerin etkin frekansı karekökle sıkışır, log–log grafikte baş kısmın eğimi −s'den yaklaşık −s/2'ye iner. Sonuç: en sık 10 kelimenin token payı %${fmt(100 * top10[0], 3)} → %${fmt(100 * top10[2], 3)}; aynı epoch sayısında nadir kelimeler görece daha çok güncelleme alır ve eğitim kısalır.`);
    } else P.push("Alt-örnekleme kapalı: \"ve\", \"bir\" gibi işlev kelimeleri hemen her pencereye girer ve çiftlerin büyük kısmını oluşturur.");
    return {
      plots: [
        { title: "Olasılık — sıra (log–log)", xlabel: "kelime sırası (log₁₀ ekseni)", ylabel: "olasılık", ylog: true, xticks: xt, series: [
          { name: "unigram f", x: lx, y: idx.map((i) => f[i]), c: 4 },
          { name: `gürültü P_n (α = ${q.alpha})`, x: lx, y: idx.map((i) => pn[i]), c: 5 },
          ...(q.form ? [{ name: "alt-örnekleme sonrası frekans", x: lx, y: idx.map((i) => fs[i]), c: 0, dash: true }] : [])] },
        { title: "Tutulma olasılığı", xlabel: "kelime sırası (log₁₀ ekseni)", ylabel: "P(tut)", ylim: [0, 1.05], xticks: xt, series: [
          { name: q.form ? ["", "makale", "word2vec.c"][q.form] : "kapalı (hepsi tutulur)", x: lx, y: idx.map((i) => keep[i]), c: 0 }],
          marks: q.form ? [{ x: Math.log10(Math.max(1, Math.min(q.V, (f.findIndex((x) => x <= t) + 1) || q.V))), label: "f = t" }] : [] },
      ],
      stats: [
        { label: "En sık 10 kelimenin token payı", value: top10[0] },
        { label: "… negatif örneklerdeki payı", value: top10[1] },
        { label: "… alt-örnekleme sonrası token payı", value: top10[2] },
        { label: `Kuyruğun (sıra > ${tail}) negatiflerdeki payı`, value: tl[1] },
        { label: "Atılan token oranı", value: 1 - kept },
        { label: "En sık kelimenin tutulma olasılığı", value: keep[0] },
      ],
      note: P.join(" "),
    };
  },
};
