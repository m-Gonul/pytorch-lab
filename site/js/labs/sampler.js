// M4.7 — Bir sonraki token'ın örneklenmesi: sıcaklık, top-k ve top-p (nucleus) dağılımı nasıl değiştirir? Sabit bir dil modeli
// çıktısı (12 aday token'ın logit'leri) üzerinde dönüştürülmüş olasılıklar, ondan çekilen örneklerin sıklığı ve entropi.
import { fmt, range } from "./plot.js";

export const TOK = ["üstüne", "üzerine", "altına", "yanına", "önüne", "arkasına", "kenarına", "ortasına", "başına", "içine", "mavi", "uçtu"];
export const LOGIT = [3.2, 2.9, 2.4, 1.8, 1.2, 1.0, 0.9, 0.3, 0.2, -0.5, -2.5, -3.5];
export function transform(z, { tau, k, p }) {
  if (tau <= 0) { const m = z.indexOf(Math.max(...z)); return z.map((_, i) => (i === m ? 1 : 0)); }
  const s = z.map((v) => v / tau), m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), Z = e.reduce((a, b) => a + b, 0);
  let P = e.map((v) => v / Z);
  const order = range(z.length).sort((a, b) => P[b] - P[a]), keep = new Set();
  // önce top-k, sonra top-p (Hugging Face sırası: sıcaklık → top-k → top-p); top-p eşiği aşan ilk token'ı da tutar
  let cum = 0; for (let j = 0; j < Math.min(k, order.length); j++) { keep.add(order[j]); cum += P[order[j]]; if (cum >= p) break; }
  P = P.map((v, i) => (keep.has(i) ? v : 0)); const S = P.reduce((a, b) => a + b, 0);
  return P.map((v) => v / S);
}
export const entropy = (P) => -P.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);

export default {
  params: [
    { id: "tau", label: "Sıcaklık τ", min: 0.05, max: 5, value: 1, log: true, help: "Logit'ler τ'ya bölünür. τ < 1 dağılımı sivriltir, τ > 1 düzleştirir; τ → 0 açgözlü (argmax) seçim." },
    { id: "k", label: "top-k (12 = kapalı)", min: 1, max: 12, value: 12, int: true },
    { id: "p", label: "top-p / nucleus (1 = kapalı)", min: 0.05, max: 1, step: 0.01, value: 1 },
    { id: "n", label: "Çekilen örnek sayısı", min: 10, max: 20000, value: 500, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const P0 = transform(LOGIT, { tau: 1, k: 12, p: 1 }), P = transform(LOGIT, { tau: q.tau, k: Math.round(q.k), p: q.p });
    const cdf = []; P.reduce((a, v, i) => (cdf[i] = a + v), 0);
    const cnt = TOK.map(() => 0), n = Math.round(q.n);
    for (let t = 0; t < n; t++) { const u = r(); let i = cdf.findIndex((c) => u < c); if (i < 0) i = P.length - 1 - [...P].reverse().findIndex((v) => v > 0); cnt[i]++; }
    const taus = range(61).map((i) => 0.05 * 100 ** (i / 60)), H = taus.map((t) => entropy(transform(LOGIT, { tau: t, k: Math.round(q.k), p: q.p })));
    const kept = P.filter((v) => v > 0).length, xs = range(12);
    return {
      plots: [
        { title: "\"Kedi masanın ___\": olasılıklar", size: { w: 780, h: 300 }, wide: true, xlabel: "", ylabel: "olasılık", yzero: true, ylim: [0, 1], xlim: [-0.6, 11.6], xticks: TOK.map((t, i) => [i, t]), series: [
          { name: "model (τ = 1, kesme yok)", x: xs, y: P0, kind: "bar", c: 4, bw: 0.28, off: -1 },
          { name: "dönüştürülmüş dağılım", x: xs, y: P, kind: "bar", c: 0, bw: 0.28, off: 0 },
          { name: `${n} örneğin sıklığı`, x: xs, y: cnt.map((c) => c / n), kind: "bar", c: 2, bw: 0.28, off: 1 }] },
        { title: "Entropi – sıcaklık (geçerli top-k/top-p ile)", xlabel: "τ (log₁₀ ekseni)", ylabel: "entropi (bit)", ylim: [0, Math.log2(12) * 1.05], xticks: [[-1, "0.1"], [0, "1"], [Math.log10(5), "5"]], series: [{ name: "H(τ)", x: taus.map(Math.log10), y: H, c: 0 }], marks: [{ x: Math.log10(q.tau), label: `τ = ${fmt(q.tau, 3)}` }, { y: Math.log2(12), label: "düzgün: log₂12" }] },
      ],
      stats: [
        { label: "Örneklenebilen token sayısı", value: `${kept} / 12` },
        { label: "Entropi (bit)", value: `${fmt(entropy(P), 3)} (model: ${fmt(entropy(P0), 3)})` },
        { label: "En olası token'ın olasılığı", value: fmt(Math.max(...P), 4) },
        { label: "Anlamsız adaylar (mavi, uçtu) toplam olasılığı", value: fmt(P[10] + P[11], 3) },
        { label: "Sıklık ile olasılık arası en büyük fark", value: fmt(Math.max(...P.map((v, i) => Math.abs(v - cnt[i] / n))), 3) },
      ],
      note: `Sıra: logit'ler τ'ya bölünür → softmax → en olası ${Math.round(q.k)} token dışındakiler atılır → kalanlardan birikimli olasılığı ${fmt(q.p, 3)}'e ilk ulaşan en küçük küme tutulur → yeniden normalize edilir. ${kept === 1 ? "<b>Tek aday kaldı:</b> örnekleme açgözlü seçime dönüştü — çıktı her seferinde aynı." : q.tau > 1.5 && q.k >= 12 && q.p >= 1 ? "<b>Yüksek sıcaklık, kesme yok:</b> \"mavi\" ve \"uçtu\" gibi anlamsız adayların payı büyüyor; uzun üretimde bunlar birikir (Holtzman ve ark.: dağılımın kuyruğu)." : "Kesmeler kuyruğu (düşük olasılıklı, çoğu kez anlamsız devamları) atar; sıcaklık ise kalan dağılımın ne kadar sivri olacağını ayarlar."} Top-p'nin top-k'ya üstünlüğü uyarlanabilir olmasıdır: model eminse küme küçük, kararsızsa büyük olur. Örnek sıklıkları ${n} çekilişte olasılıklara ≈ 1/√n ≈ ${fmt(1 / Math.sqrt(n), 2)} mertebesinde yaklaşır.`,
    };
  },
};
