// M6.4 — Gradyan biriktirme ne zaman büyük batch'le birebir aynıdır? N dizi, dizi i'nin uzunluğu Lᵢ token. Her token: x ∈ R³,
// y = w*·x + gürültü. Model ŷ = w·norm(x); norm = özdeşlik ya da BN (özelliklerin batch'teki token ortalaması/std'si ile).
// Kayıp: batch'teki tüm tokenların ortalaması. Büyük batch gradyanı ile m dizilik mikro batch'lerden biriktirilen gradyan karşılaştırılır;
// mikro kayıplar üç biçimde ölçeklenir: örnek sayısıyla (n_j/N), token sayısıyla (T_j/T), eşit (1/J).
import { fmt, range } from "./plot.js";

const SC = ["örnek sayısıyla: n_j / N", "token sayısıyla: T_j / T", "eşit: 1 / J"];
function grad(tokens, w, bn) {
  let X = tokens.map((t) => t.x);
  if (bn) { const mu = [0, 1, 2].map((d) => X.reduce((s, x) => s + x[d], 0) / X.length), sd = [0, 1, 2].map((d) => Math.sqrt(X.reduce((s, x) => s + (x[d] - mu[d]) ** 2, 0) / X.length + 1e-5)); X = X.map((x) => x.map((v, d) => (v - mu[d]) / sd[d])); }
  const g = [0, 0, 0]; X.forEach((x, i) => { const e = x.reduce((s, v, d) => s + v * w[d], 0) - tokens[i].y; x.forEach((v, d) => (g[d] += (2 * e * v) / X.length)); });
  return g;
}
export function accumulate(seqs, w, m, bn, scale) {
  const N = seqs.length, T = seqs.reduce((s, q) => s + q.length, 0), chunks = []; for (let s = 0; s < N; s += m) chunks.push(seqs.slice(s, s + m));
  const g = [0, 0, 0];
  chunks.forEach((c) => { const tok = c.flat(), gj = grad(tok, w, bn), a = scale === 0 ? c.length / N : scale === 1 ? tok.length / T : 1 / chunks.length; gj.forEach((v, d) => (g[d] += a * v)); });
  return g;
}
const rel = (a, b) => Math.hypot(...a.map((v, i) => v - b[i])) / Math.max(1e-12, Math.hypot(...b));
export default {
  params: [
    { id: "N", label: "Batch: dizi sayısı N", min: 4, max: 128, value: 32, int: true, log: true },
    { id: "m", label: "Mikro batch: dizi sayısı m", min: 1, max: 128, value: 8, int: true, log: true },
    { id: "Lmax", label: "En uzun dizi (token; 1 = hepsi eşit)", min: 1, max: 64, value: 32, int: true, log: true },
    { id: "bn", label: "Normalizasyon", options: ["yok", "BatchNorm (batch istatistikleriyle)"], value: 0 },
    { id: "scale", label: "Mikro kaybın ölçeği", options: SC, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const ws = [1, -2, 0.5], w = [1.1, -2.1, 0.55]; // optimuma yakın: gradyan gürültü baskın, eğitimin çoğu böyle geçer
    const seqs = range(q.N).map(() => { const L = q.Lmax === 1 ? 1 : 1 + Math.floor(r() * q.Lmax); return range(L).map(() => { const x = [r.normal() + 1, 2 * r.normal(), r.normal() - 0.5]; return { x, y: x.reduce((s, v, d) => s + v * ws[d], 0) + 0.3 * r.normal() }; }); });
    const big = grad(seqs.flat(), w, q.bn === 1), m = Math.min(q.m, q.N), acc = accumulate(seqs, w, m, q.bn === 1, q.scale), err = rel(acc, big);
    const ms = [1, 2, 4, 8, 16, 32, 64, 128].filter((v) => v <= q.N), sweep = [0, 1, 2].map((s) => ms.map((mm) => Math.max(1e-16, rel(accumulate(seqs, w, mm, q.bn === 1, s), big))));
    const T = seqs.reduce((s, x) => s + x.length, 0), J = Math.ceil(q.N / m);
    const P = [`${q.N} dizi, toplam ${T} token, ${J} mikro batch. Biriktirilen gradyanın büyük batch gradyanından göreli farkı: ${fmt(err, 2)}.`];
    if (q.bn === 1) P.push("BatchNorm her mikro batch'te kendi ortalama ve std'sini kullanır; ileri geçiş büyük batch'tekinden farklıdır, bu yüzden hiçbir ölçekleme farkı sıfırlayamaz (yalnız m = N'de sıfır). Çareler: daha büyük mikro batch, LayerNorm/GroupNorm ya da senkron BN.");
    else if (err < 1e-12) P.push("Fark makine hassasiyetinde: kayıp tokenlar üzerinden toplanabilir olduğu için doğru ağırlıklarla biriktirme birebir aynı adımı verir.");
    else {
      if (q.scale === 0 && q.Lmax > 1) P.push("Örnek sayısıyla ölçekleme dizi başına ortalama için doğrudur; ama kayıp token ortalaması ve diziler farklı uzunlukta. Uzun dizilerin bulunduğu mikro batch'ler hak ettiğinden az ağırlık alır. Doğru ağırlık T_j / T'dir (dil modellerinde sık görülen hata).");
      if (q.scale === 2) P.push("Eşit ağırlık (mikro kayıpların basit ortalaması) yalnız mikro batch'ler aynı token sayısını taşıyorsa doğrudur; son parça küçükse ya da uzunluklar farklıysa sapar.");
    }
    return {
      plots: [{ title: "Göreli gradyan farkı ve mikro batch boyu", xlabel: "m (log₂)", ylabel: "‖g_acc − g_büyük‖ / ‖g_büyük‖", ylog: true, series: [
        { name: SC[0], x: ms.map(Math.log2), y: sweep[0], c: 0 }, { name: SC[1], x: ms.map(Math.log2), y: sweep[1], c: 2 }, { name: SC[2], x: ms.map(Math.log2), y: sweep[2], c: 1, dash: true }],
        marks: [{ x: Math.log2(m), label: "m" }], xticks: ms.map((v) => [Math.log2(v), String(v)]) }],
      stats: [
        { label: "Göreli fark (seçili ölçekleme)", value: err },
        { label: "Toplam token / mikro batch sayısı", value: `${T} / ${J}` },
      ],
      note: P.join(" "),
    };
  },
};
