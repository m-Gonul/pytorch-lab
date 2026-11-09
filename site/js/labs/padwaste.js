// M4.13 — Dolgu israfı: değişken uzunluklu cümleleri batch'lere koyunca her batch en uzun cümlesine kadar doldurulur.
// Üç strateji: rastgele batch'ler, tüm veriyi uzunluğa göre sıralayıp bölmek, "kovalama" (x2'deki gibi: karıştır, chunk·B'lik
// parçalara böl, parçayı sırala, batch'lere ayır, batch sırasını karıştır). İsraf = dolgu token / dolgulu toplam token.
import { fmt, range } from "./plot.js";

export function lengths(n, dist, mean, spread, r) {
  return range(n).map(() => {
    let L;
    if (dist === 0) L = Math.exp(Math.log(mean) - spread * spread / 2 + spread * r.normal());
    else if (dist === 1) L = 1 + r() * (2 * mean - 1);
    else L = r() < 0.7 ? Math.exp(Math.log(mean * 0.6) + 0.25 * r.normal()) : Math.exp(Math.log(mean * 2.5) + 0.25 * r.normal());
    return Math.max(1, Math.round(L));
  });
}
const perm = (n, r) => { const p = range(n); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; };
export function batches(L, B, strat, chunk, r) {
  const n = L.length;
  if (strat === 0) { const p = perm(n, r); return range(Math.ceil(n / B)).map((i) => p.slice(i * B, i * B + B)); }
  const out = [], step = strat === 1 ? n : chunk * B, p = strat === 1 ? range(n) : perm(n, r);
  for (let i = 0; i < n; i += step) { const part = p.slice(i, i + step).sort((a, b) => L[a] - L[b] || a - b); for (let j = 0; j < part.length; j += B) out.push(part.slice(j, j + B)); }
  const o = perm(out.length, r); return o.map((i) => out[i]);
}
export function waste(L, bs) { let real = 0, tot = 0; for (const b of bs) { let m = 0; for (const i of b) { real += L[i]; m = Math.max(m, L[i]); } tot += m * b.length; } return { real, tot, w: 1 - real / tot }; }

export default {
  params: [
    { id: "n", label: "Cümle sayısı", min: 200, max: 20000, value: 5000, int: true, log: true },
    { id: "dist", label: "Uzunluk dağılımı", options: ["log-normal (doğal metin)", "düzgün", "iki tepeli (kısa + uzun)"], value: 0 },
    { id: "mean", label: "Ortalama uzunluk (token)", min: 3, max: 80, value: 20, int: true },
    { id: "spread", label: "Log-normal yayılım σ", min: 0.1, max: 1.2, step: 0.05, value: 0.6, on: { dist: [0] } },
    { id: "B", label: "Batch boyu B", min: 2, max: 512, value: 64, int: true, log: true },
    { id: "strat", label: "Batch oluşturma", options: ["rastgele", "tümünü sırala", "kovalama (parça içinde sırala)"], value: 2 },
    { id: "chunk", label: "Kovalama parçası (kaç batch'lik)", min: 1, max: 200, value: 20, int: true, log: true, on: { strat: [2] } },
  ],
  seed: true,
  validate(q) { return q.B > q.n ? [{ level: "error", msg: "Batch boyu B cümle sayısından büyük olamaz." }] : []; },
  run(q, r) {
    const L = lengths(q.n, q.dist, q.mean, q.spread, r), bs = batches(L, q.B, q.strat, q.chunk, r), W = waste(L, bs);
    const wr = waste(L, batches(L, q.B, 0, 1, r)).w, ws = waste(L, batches(L, q.B, 1, 1, r)).w;
    const chunks = [1, 2, 5, 10, 20, 50, 100, 200].filter((c) => c * q.B <= 4 * q.n), wc = chunks.map((c) => waste(L, batches(L, q.B, 2, c, r)).w);
    const maxL = Math.max(...L), hb = Math.min(60, maxL), edges = range(hb + 1).map((i) => (i * maxL) / hb), hist = range(hb).map(() => 0);
    L.forEach((x) => hist[Math.min(hb - 1, Math.floor(((x - 0.5) / maxL) * hb))]++);
    const mx = bs.map((b) => Math.max(...b.map((i) => L[i]))), mean = L.reduce((a, b) => a + b, 0) / L.length;
    const cv = Math.sqrt(L.reduce((s, x) => s + (x - mean) ** 2, 0) / L.length) / mean;
    const name = ["rastgele", "tümünü sırala", `kovalama (parça ${q.chunk} batch)`][q.strat];
    const P = [`${q.n} cümle, ortalama uzunluk ${fmt(mean, 3)}, en uzun ${maxL}, değişim katsayısı ${fmt(cv, 2)}. ${name}: dolgulu tensörlerde ${W.tot.toLocaleString("tr-TR")} hücre, bunların ${W.real.toLocaleString("tr-TR")} tanesi gerçek token — israf %${fmt(100 * W.w, 3)}. GPU bu dolgu hücrelerini de hesaplar (maske yalnız sonucu yok sayar), yani süre ≈ dolgulu toplamla orantılıdır: gerçek token başına ${fmt(W.tot / W.real, 3)} kat iş.`];
    P.push(`Karşılaştırma: rastgele %${fmt(100 * wr, 3)}, tümünü sıralamak %${fmt(100 * ws, 3)}. Tümünü sıralamak en az israfı verir ama her epoch'ta aynı cümleler aynı batch'e düşer ve batch'ler uzunluğa göre homojenleşir (kısa batch'lerin gradyanı başka, uzunlarınki başka bir alt kümeye ait); kovalama israfın çoğunu kurtarırken parça içindeki rastgeleliği korur.`);
    if (q.B <= 4) P.push("Çok küçük batch'te israf zaten düşük: dolgu, batch içindeki uzunluk farkından gelir.");
    return {
      plots: [
        { title: "Cümle uzunluğu dağılımı", xlabel: "uzunluk (token)", ylabel: "cümle sayısı", series: [{ name: "histogram", x: edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2), y: hist, kind: "bar", c: 0 }] },
        { title: "Kovalama parçası büyüdükçe israf", xlabel: "parça (batch sayısı, log₁₀ ekseni)", ylabel: "israf oranı", ylim: [0, Math.max(wr, 0.05) * 1.1], xticks: chunks.map((c) => [Math.log10(c), String(c)]), series: [
          { name: "kovalama", x: chunks.map(Math.log10), y: wc, c: 2, dots: true },
          { name: "rastgele", x: [0, Math.log10(chunks.at(-1) || 1)], y: [wr, wr], c: 4, dash: true },
          { name: "tümünü sırala", x: [0, Math.log10(chunks.at(-1) || 1)], y: [ws, ws], c: 1, dash: true }],
          marks: q.strat === 2 ? [{ x: Math.log10(q.chunk), label: `parça = ${q.chunk}` }] : [] },
      ],
      stats: [
        { label: `İsraf (${name})`, value: W.w },
        { label: "Gerçek token başına hesaplanan hücre", value: W.tot / W.real },
        { label: "İsraf, rastgele batch", value: wr },
        { label: "İsraf, tümünü sırala", value: ws },
        { label: "Batch sayısı", value: bs.length },
        { label: "Batch başına en uzun cümle, ortalama", value: mx.reduce((a, b) => a + b, 0) / mx.length },
      ],
      note: P.join(" "),
    };
  },
};
