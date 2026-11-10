// M8.2 — İki modeli dürüstçe karşılaştırmak: aynı n test örneğinde örnek başı skorlar (ör. Dice) a_i, b_i; ortalama fark δ, örnekler
// arası ortak zorluk (ilişki ρ). Eşleştirilmemiş bootstrap: A ve B'yi ayrı yeniden örnekle; eşleştirilmiş: aynı indekslerle
// yeniden örnekle ve d_i = b_i − a_i farklarının ortalamasını al. %95 yüzdelik güven aralıkları karşılaştırılır.
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "n", label: "Test örneği sayısı n", min: 10, max: 2000, value: 100, int: true, log: true },
    { id: "delta", label: "Gerçek ortalama fark δ (B − A)", min: -0.1, max: 0.1, step: 0.005, value: 0.02 },
    { id: "sigma", label: "Örnekler arası skor std'si", min: 0.01, max: 0.3, step: 0.01, value: 0.15 },
    { id: "rho", label: "İki modelin örnek başı skor ilişkisi ρ", min: 0, max: 0.99, step: 0.01, value: 0.9 },
    { id: "B", label: "Bootstrap tekrarı", min: 200, max: 5000, value: 2000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const z = range(q.n).map(() => r.normal()), a = z.map((v) => 0.7 + q.sigma * v), b = z.map((v) => 0.7 + q.delta + q.sigma * (q.rho * v + Math.sqrt(1 - q.rho ** 2) * r.normal()));
    const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length, obs = mean(b) - mean(a);
    const pd = [], ud = [];
    for (let k = 0; k < q.B; k++) {
      let sp = 0, sa = 0, sb = 0;
      for (let i = 0; i < q.n; i++) { const j = Math.floor(r() * q.n); sp += b[j] - a[j]; sa += a[Math.floor(r() * q.n)]; sb += b[Math.floor(r() * q.n)]; }
      pd.push(sp / q.n); ud.push((sb - sa) / q.n);
    }
    const ci = (x) => { const s = [...x].sort((u, v) => u - v); return [s[Math.floor(0.025 * x.length)], s[Math.floor(0.975 * x.length) - 1]]; };
    const P_ = ci(pd), U = ci(ud), wP = P_[1] - P_[0], wU = U[1] - U[0], exP = P_[0] > 0 || P_[1] < 0, exU = U[0] > 0 || U[1] < 0;
    const nb = 40, lo = Math.min(...ud, ...pd), hi = Math.max(...ud, ...pd), bw = (hi - lo) / nb || 1e-9, hist = (x) => { const h = new Array(nb).fill(0); x.forEach((v) => h[Math.min(nb - 1, Math.floor((v - lo) / bw))]++); return h; };
    const P = [`Gözlenen fark ${fmt(obs, 3)} (gerçek ${q.delta}). %95 aralık: eşleştirilmiş [${fmt(P_[0], 3)}, ${fmt(P_[1], 3)}] (genişlik ${fmt(wP, 3)}), eşleştirilmemiş [${fmt(U[0], 3)}, ${fmt(U[1], 3)}] (genişlik ${fmt(wU, 3)}). ${exP ? "Eşleştirilmiş aralık 0'ı dışlıyor" : "Eşleştirilmiş aralık 0'ı içeriyor"}; ${exU ? "eşleştirilmemiş de dışlıyor" : "eşleştirilmemiş içeriyor"}.`];
    P.push(`İki model aynı örneklerde değerlendirildiği için örnek zorluğu ortaktır (ρ = ${q.rho}); eşleştirme bu ortak gürültüyü farkta yok eder. Farkın varyansı 2σ²(1 − ρ)/n, eşleştirilmemiş tahminde 2σ²/n: genişlik oranı ≈ √(1 − ρ) = ${fmt(Math.sqrt(1 - q.rho), 3)} (ölçülen ${fmt(wP / wU, 3)}).`);
    if (!exP && Math.abs(q.delta) > 0) P.push("Gerçek bir fark olduğu hâlde aralık 0'ı içeriyor: test kümesi bu farkı ayırt etmeye yetmiyor. n'yi büyütmek ya da tohumları da yeniden örneklemek (eğitim rastgeleliği) gerekir.");
    const xs = range(nb).map((k) => lo + (k + 0.5) * bw);
    return {
      plots: [{ title: "Farkın bootstrap dağılımı", xlabel: "B − A", ylabel: "tekrar", wide: true, size: { w: 620, h: 230 }, series: [{ name: "eşleştirilmemiş", x: xs, y: hist(ud), c: 4 }, { name: "eşleştirilmiş", x: xs, y: hist(pd), c: 0 }], marks: [{ x: 0, label: "0" }, { x: q.delta, label: "δ" }] }],
      stats: [
        { label: "Eşleştirilmiş %95 aralık", value: `[${fmt(P_[0], 3)}, ${fmt(P_[1], 3)}]` },
        { label: "Eşleştirilmemiş %95 aralık", value: `[${fmt(U[0], 3)}, ${fmt(U[1], 3)}]` },
        { label: "Genişlik oranı (ölçülen / √(1 − ρ))", value: `${fmt(wP / wU, 3)} / ${fmt(Math.sqrt(1 - q.rho), 3)}` },
      ],
      note: P.join(" "),
    };
  },
};
