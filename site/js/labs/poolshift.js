// M4.1 — Havuzlama ve kayma: girdi s piksel sağa kayınca max/ortalama havuzlamanın çıktısı ne kadar değişir?
// Küçük kaymalara kısmi değişmezlik (invariance), k'nın katı kaymalarda eşdeğişirlik (equivariance), aradaki durumlarda ise ikisi de değil.
import { fmt, range } from "./plot.js";

const N = 16;
export function pattern(kind, r) {
  const X = range(N).map(() => new Array(N).fill(0));
  if (kind === 0) for (let t = 0; t < 22; t++) X[Math.floor(r() * N)][Math.floor(r() * (N - 4))] = 0.4 + 0.6 * r();
  else if (kind === 1) for (let i = 4; i < 11; i++) for (let j = 3; j < 9; j++) X[i][j] = i === 4 || i === 10 || j === 3 || j === 8 ? 1 : 0;
  else for (let i = 0; i < N; i++) for (let j = 0; j < N - 4; j++) X[i][j] = (i + 2 * j) % 5 === 0 ? 1 : 0;
  return X;
}
export const shift = (X, s) => X.map((row) => row.map((_, j) => (j - s >= 0 ? row[j - s] : 0)));
export function pool(X, k, op) {
  const M = Math.floor(N / k);
  return range(M).map((a) => range(M).map((b) => { const v = []; for (let u = 0; u < k; u++) for (let w = 0; w < k; w++) v.push(X[a * k + u][b * k + w]); return op === 0 ? Math.max(...v) : v.reduce((p, c) => p + c, 0) / v.length; }));
}
const same = (A, B) => { let e = 0, n = 0; A.forEach((r, i) => r.forEach((v, j) => { n++; if (Math.abs(v - B[i][j]) < 1e-9) e++; })); return e / n; };
const flat = (X) => X.flat();

export default {
  params: [
    { id: "pat", label: "Desen", options: ["seyrek parlak noktalar", "kare çerçeve", "eğik çizgiler"], value: 0 },
    { id: "s", label: "Sağa kayma s (piksel)", min: 0, max: 4, value: 1, int: true },
    { id: "k", label: "Havuz çekirdeği k (adım = k)", min: 2, max: 4, value: 2, int: true },
    { id: "op", label: "Havuzlama", options: ["max", "ortalama"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const k = Math.round(q.k), s = Math.round(q.s), A = pattern(q.pat, r), B = shift(A, s), PA = pool(A, k, q.op), PB = pool(B, k, q.op), M = PA.length;
    const ss = range(5), sweepMax = ss.map((t) => same(pool(A, k, 0), pool(shift(A, t), k, 0))), sweepAvg = ss.map((t) => same(pool(A, k, 1), pool(shift(A, t), k, 1)));
    const equi = s % k === 0 ? same(PB.map((row) => row.slice(s / k)), PA.map((row) => row.slice(0, M - s / k))) : null;
    const mx = Math.max(1e-9, ...flat(PA), ...flat(PB)), nrm = (X) => flat(X).map((v) => v / mx);
    return {
      plots: [
        { title: "A: girdi", img: { w: N, h: N, data: flat(A) }, side: true },
        { title: `B: ${s} piksel sağa kaydırılmış`, img: { w: N, h: N, data: flat(B) }, side: true },
        { title: `${["max", "ortalama"][q.op]} havuz(A), ${k}×${k}`, img: { w: M, h: M, data: nrm(PA) }, side: true },
        { title: `${["max", "ortalama"][q.op]} havuz(B)`, img: { w: M, h: M, data: nrm(PB) }, side: true },
        { title: "Havuz çıktısında hücre hücre aynı kalan oran, kaymaya göre", xlabel: "kayma s (piksel)", ylabel: "aynı hücre oranı", ylim: [0, 1.05], series: [{ name: "max", x: ss, y: sweepMax, c: 0, dots: true }, { name: "ortalama", x: ss, y: sweepAvg, c: 2, dots: true }], marks: [{ x: s, label: "seçili s" }] },
      ],
      stats: [
        { label: "Havuz çıktısında aynı kalan hücre oranı", value: `%${fmt(same(PA, PB) * 100, 3)}` },
        { label: `Kayma k'nın katıyken: havuz(B) = havuz(A) ${s % k === 0 && s > 0 ? `${s / k} hücre kaydırılmış` : ""}`, value: equi == null ? "— (s, k'nın katı değil)" : s === 0 ? "s = 0 (aynı girdi)" : `%${fmt(equi * 100, 3)} eşleşme` },
      ],
      note: `Max havuzlama her k×k pencereden yalnız en büyük değeri geçirir: parlak bir nokta pencerenin içinde kaldığı sürece (s < k ve pencere sınırını geçmezse) çıktı değişmez — <b>küçük kaymalara kısmi değişmezlik</b>. Ortalama havuzlama pencere içindeki her değişimi görür; ${q.op === 1 ? "bu yüzden aynı kalan hücre oranı daha düşük." : "değişmezliği daha zayıftır."} Kayma tam k'nın katıysa çıktı yalnızca <b>kayar</b> (eşdeğişirlik); arada ise çıktı ne aynı kalır ne düzgünce kayar — modern CNN'lerin küçük kaymalara bile hassas olabilmesinin bir nedeni bu "örtüşme" (aliasing) etkisidir (Zhang 2019). Sınıflandırmada değişmezlik işe yarar; segmentasyon gibi çıktının girdiyle birlikte kayması gereken görevlerde istenen eşdeğişirliktir.`,
    };
  },
};
