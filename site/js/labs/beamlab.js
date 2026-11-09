// M4.13 — Açgözlü çözümleme, beam search ve kesin arama. Rastgele bigram dil modelleri (önceki token → sonraki token + EOS)
// üzerinde: beam (k = 1 açgözlüdür) en olası diziyi ne sıklıkla kaçırır, çıktılar k büyüdükçe neden kısalır, uzunluk
// normalizasyonu (skor = log p / L^α, L: EOS dahil token sayısı) bunu nasıl düzeltir. Bigram modelde kesin optimum Viterbi ile
// bulunur: her uzunluk ℓ için en olası tamamlanmış dizi, sonra ℓ üzerinden en iyi normalize skor.
import { fmt, range } from "./plot.js";

export function makeLM(V, sharp, eos, r) {
  // satır 0..V-1: önceki token, satır V: SOS; sütun V: EOS
  return range(V + 1).map(() => { const z = range(V + 1).map(() => sharp * r.normal()); z[V] += eos; const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => Math.log(x / s)); });
}
const score = (lp, L, a) => lp / L ** a;
export function exact(LM, V, T, a) {
  // best[v]: v ile biten, ℓ uzunluklu en olası önek (EOS'suz)
  let best = LM[V].slice(0, V), out = { lp: LM[V][V], L: 1, s: score(LM[V][V], 1, a) };
  for (let l = 1; l < T; l++) {
    for (let v = 0; v < V; v++) { const lp = best[v] + LM[v][V], s = score(lp, l + 1, a); if (s > out.s) out = { lp, L: l + 1, s }; }
    if (l + 1 >= T) break;
    best = range(V).map((w) => Math.max(...range(V).map((v) => best[v] + LM[v][w])));
  }
  return out;
}
export function beam(LM, V, T, k, a) {
  let open = [{ last: V, lp: 0, L: 0 }]; const done = [];
  for (let step = 0; step < T && open.length && done.length < k; step++) {
    const cand = [];
    for (const h of open) for (let w = h.L + 1 >= T ? V : 0; w <= V; w++) cand.push({ last: w, lp: h.lp + LM[h.last][w], L: h.L + 1 }); // son adımda yalnız EOS
    cand.sort((x, y) => y.lp - x.lp);
    open = [];
    for (const c of cand.slice(0, k)) (c.last === V ? done : open).push(c);
  }
  const pool = done.length ? done : open;
  return pool.reduce((b, c) => (score(c.lp, c.L, a) > score(b.lp, b.L, a) ? c : b));
}

export default {
  params: [
    { id: "V", label: "Sözlük boyu V (EOS hariç)", min: 3, max: 40, value: 12, int: true },
    { id: "sharp", label: "Model keskinliği σ (logit std'si)", min: 0.3, max: 4, step: 0.1, value: 2 },
    { id: "eos", label: "EOS logit kaydırması", min: -4, max: 3, step: 0.1, value: 0 },
    { id: "T", label: "En büyük uzunluk T (EOS dahil)", min: 3, max: 25, value: 15, int: true },
    { id: "k", label: "Beam genişliği k (1 = açgözlü)", min: 1, max: 30, value: 5, int: true },
    { id: "alpha", label: "Uzunluk normalizasyonu α (skor = log p / L^α)", min: 0, max: 1.5, step: 0.05, value: 0 },
  ],
  seed: true,
  run(q, r) {
    const N = 60, LMs = range(N).map(() => makeLM(q.V, q.sharp, q.eos, r));
    const ks = [...new Set([1, 2, 3, 5, 8, 12, 20, 30, q.k])].sort((a, b) => a - b);
    const ex0 = LMs.map((M) => exact(M, q.V, q.T, 0)), exA = LMs.map((M) => exact(M, q.V, q.T, q.alpha));
    const agg = (a) => ks.map((k) => { const B = LMs.map((M) => beam(M, q.V, q.T, k, a)), E = a === 0 ? ex0 : exA; return {
      L: B.reduce((s, b) => s + b.L, 0) / N, lp: B.reduce((s, b) => s + b.lp, 0) / N, err: B.filter((b, i) => score(b.lp, b.L, a) < E[i].s - 1e-9).length / N, empty: B.filter((b) => b.L === 1).length / N }; });
    const A0 = agg(0), AA = q.alpha === 0 ? A0 : agg(q.alpha), cur = AA[ks.indexOf(q.k)], g = AA[0];
    const mean = (xs, f) => xs.reduce((s, x) => s + f(x), 0) / xs.length, lk = ks.map((k) => Math.log2(k)), xt = ks.filter((k) => [1, 2, 5, 12, 30].includes(k) || k === q.k).map((k) => [Math.log2(k), String(k)]);
    const exL0 = mean(ex0, (e) => e.L), exLA = mean(exA, (e) => e.L), exE0 = ex0.filter((e) => e.L === 1).length / N;
    const P = [`${N} rastgele bigram modeli. k = ${q.k}${q.alpha ? `, α = ${q.alpha}` : ""}: ortalama çıktı uzunluğu ${fmt(cur.L, 3)} (EOS dahil), ortalama log p = ${fmt(cur.lp, 3)}; ${q.k === 1 ? "açgözlü çözümleme" : "beam"} modellerin %${fmt(100 * cur.err, 3)} kadarında kesin optimumu (aynı skorla) kaçırıyor. Açgözlüde bu oran %${fmt(100 * g.err, 3)}.`];
    P.push(`Kesin aramayla (α = 0, yani en olası dizi) ortalama uzunluk ${fmt(exL0, 3)}; modellerin %${fmt(100 * exE0, 3)} kadarında en olası "çeviri" <b>boş dizidir</b> (yalnız EOS). Her ek token log p'ye negatif bir terim eklediği için ham olasılık kısa dizileri kayırır: k büyüdükçe arama bu optimuma yaklaşır ve çıktılar kısalır (α = 0'da k = 1 → ${ks.at(-1)}: uzunluk ${fmt(A0[0].L, 3)} → ${fmt(A0.at(-1).L, 3)}, boş çıktı %${fmt(100 * A0[0].empty, 3)} → %${fmt(100 * A0.at(-1).empty, 3)}).`);
    if (q.alpha > 0) P.push(`α = ${q.alpha} ile skor log p / L^α: kesin optimumun ortalama uzunluğu ${fmt(exLA, 3)}, beam k = ${q.k}'de ${fmt(cur.L, 3)}, boş çıktı %${fmt(100 * cur.empty, 3)}. Normalizasyon "arama hatası"nı bir model hatasıyla takas eder: daha uzun ama daha az olası diziler seçilir.`);
    return {
      plots: [
        { title: "Çıktı uzunluğu ve beam genişliği", xlabel: "k (log₂ ekseni)", ylabel: "ortalama uzunluk (EOS dahil)", xticks: xt, yzero: true, series: [
          { name: "beam, α = 0", x: lk, y: A0.map((a) => a.L), c: 5, dots: true },
          { name: "kesin, α = 0", x: [lk[0], lk.at(-1)], y: [exL0, exL0], c: 5, dash: true },
          ...(q.alpha ? [{ name: `beam, α = ${q.alpha}`, x: lk, y: AA.map((a) => a.L), c: 0, dots: true }, { name: `kesin, α = ${q.alpha}`, x: [lk[0], lk.at(-1)], y: [exLA, exLA], c: 0, dash: true }] : [])],
          marks: [{ x: Math.log2(q.k), label: `k = ${q.k}` }] },
        { title: "Arama hatası ve boş çıktı oranı", xlabel: "k (log₂ ekseni)", ylabel: "oran", ylim: [0, 1], xticks: xt, series: [
          { name: `optimumu kaçırma (α = ${q.alpha})`, x: lk, y: AA.map((a) => a.err), c: 3, dots: true },
          { name: `boş çıktı (α = ${q.alpha})`, x: lk, y: AA.map((a) => a.empty), c: 2, dots: true }] },
      ],
      stats: [
        { label: `Ortalama log p, beam k = ${q.k}`, value: cur.lp },
        { label: "Ortalama log p, kesin optimum (α = 0)", value: mean(ex0, (e) => e.lp) },
        { label: `Arama hatası oranı (k = ${q.k})`, value: cur.err },
        { label: `Ortalama uzunluk (k = ${q.k})`, value: cur.L },
        { label: `Boş çıktı oranı (k = ${q.k})`, value: cur.empty },
        { label: "Boş dizinin en olası olduğu model oranı", value: exE0 },
      ],
      note: P.join(" "),
    };
  },
};
