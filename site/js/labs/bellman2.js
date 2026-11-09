// M5.5 — İki durumlu, iki eylemli bir MDP'de değer iterasyonu ve politika değerlendirme. Durumlar A, B; eylemler "kal", "geç".
// "kal": aynı durumda kalır (olasılık 1 − kayma), "geç": diğer duruma geçer (1 − kayma). Ödüller r(A, ·), r(B, ·) kullanıcıdan.
// V_{k+1}(s) = max_a [r(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]; kesin V* ve sabit bir politikanın değeri (I − γP_π)^{-1} r_π.
import { fmt, range } from "./plot.js";

export function mdp(q) {
  const P = (s, a) => { const other = 1 - s, same = s; const p = [0, 0]; if (a === 0) { p[same] += 1 - q.slip; p[other] += q.slip; } else { p[other] += 1 - q.slip; p[same] += q.slip; } return p; };
  const R = [[q.rAs, q.rAg], [q.rBs, q.rBg]];
  return { P, R };
}
const backup = (M, V, g) => [0, 1].map((s) => [0, 1].map((a) => M.R[s][a] + g * M.P(s, a).reduce((t, p, s2) => t + p * V[s2], 0)));
export function policyValue(M, pol, g) {
  const P = [0, 1].map((s) => M.P(s, pol[s])), r = [0, 1].map((s) => M.R[s][pol[s]]);
  const a = 1 - g * P[0][0], b = -g * P[0][1], c = -g * P[1][0], d = 1 - g * P[1][1], det = a * d - b * c;
  return [(d * r[0] - b * r[1]) / det, (-c * r[0] + a * r[1]) / det];
}

export default {
  params: [
    { id: "gamma", label: "İndirim γ", min: 0, max: 0.99, step: 0.01, value: 0.9 },
    { id: "slip", label: "Kayma olasılığı (eylemin ters sonucu)", min: 0, max: 0.5, step: 0.05, value: 0.1 },
    { id: "rAs", label: "r(A, kal)", min: -5, max: 5, step: 0.5, value: 1 },
    { id: "rAg", label: "r(A, geç)", min: -5, max: 5, step: 0.5, value: 0 },
    { id: "rBs", label: "r(B, kal)", min: -5, max: 5, step: 0.5, value: 0 },
    { id: "rBg", label: "r(B, geç)", min: -5, max: 5, step: 0.5, value: 2 },
    { id: "iters", label: "Değer iterasyonu adımı", min: 1, max: 200, value: 60, int: true, log: true },
  ],
  run(q) {
    const M = mdp(q), pols = [[0, 0], [0, 1], [1, 0], [1, 1]], vals = pols.map((p) => policyValue(M, p, q.gamma));
    const bi = vals.map((v, i) => [v[0] + v[1], i]).sort((a, b) => b[0] - a[0])[0][1], Vstar = vals[bi], pstar = pols[bi];
    let V = [0, 0]; const hist = [V], err = [Math.max(Math.abs(V[0] - Vstar[0]), Math.abs(V[1] - Vstar[1]))];
    for (let k = 0; k < q.iters; k++) { V = backup(M, V, q.gamma).map((qs) => Math.max(...qs)); hist.push(V); err.push(Math.max(Math.abs(V[0] - Vstar[0]), Math.abs(V[1] - Vstar[1]))); }
    const greedy = backup(M, V, q.gamma).map((qs) => (qs[1] > qs[0] ? 1 : 0)), nm = ["kal", "geç"], ks = range(q.iters + 1);
    const bound = ks.map((k) => err[0] * q.gamma ** k);
    const P = [`En iyi politika: A'da "${nm[pstar[0]]}", B'de "${nm[pstar[1]]}"; V*(A) = ${fmt(Vstar[0], 4)}, V*(B) = ${fmt(Vstar[1], 4)} (dört deterministik politikanın değerleri (I − γP_π)⁻¹r_π ile kesin hesaplandı).`];
    P.push(`${q.iters} değer iterasyonu adımından sonra V = (${fmt(V[0], 4)}, ${fmt(V[1], 4)}), hata ‖V_k − V*‖∞ = ${fmt(err.at(-1), 3)}; açgözlü politika ${greedy[0] === pstar[0] && greedy[1] === pstar[1] ? "en iyi politikayla aynı" : "henüz en iyi politika değil"}. Hata her adımda en az γ = ${q.gamma} kat küçülür (Bellman operatörü γ-büzülmedir): sınır ‖V₀ − V*‖·γᵏ.`);
    if (q.gamma > 0.95) P.push(`γ büyük: büzülme yavaş, %1 hataya inmek için ≈ ${Math.ceil(Math.log(0.01) / Math.log(q.gamma))} adım gerekir.`);
    return {
      plots: [
        { title: "Değer iterasyonu: V_k(A), V_k(B)", xlabel: "adım k", ylabel: "değer", series: [
          { name: "V_k(A)", x: ks, y: hist.map((v) => v[0]), c: 0 }, { name: "V_k(B)", x: ks, y: hist.map((v) => v[1]), c: 2 }],
          marks: [{ y: Vstar[0], label: "V*(A)" }, { y: Vstar[1], label: "V*(B)" }] },
        { title: "Hata ve γᵏ sınırı", xlabel: "adım k", ylabel: "‖V_k − V*‖∞", ylog: true, series: [
          { name: "hata", x: ks, y: err.map((e) => Math.max(e, 1e-12)), c: 3 }, { name: "‖V₀ − V*‖·γᵏ", x: ks, y: bound.map((e) => Math.max(e, 1e-12)), c: 4, dash: true }] },
      ],
      stats: [
        { label: "V*(A), V*(B)", value: `${fmt(Vstar[0], 4)}, ${fmt(Vstar[1], 4)}` },
        { label: "En iyi politika (A, B)", value: `${nm[pstar[0]]}, ${nm[pstar[1]]}` },
        { label: "Son hata ‖V_k − V*‖∞", value: err.at(-1) },
        { label: "Dört politikanın V(A)", value: vals.map((v) => fmt(v[0], 3)).join(" / ") },
      ],
      note: P.join(" "),
    };
  },
};
