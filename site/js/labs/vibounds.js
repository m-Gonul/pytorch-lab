// M11.2 — Değer yinelemesinin durma kuralı ve sınırları, rastgele (tohumlu) bir MDP'de. V₀ = 0'dan V_n = H*(V_{n−1}); ε_n = ‖V_n − V_{n−1}‖∞.
// Gerçek hata ‖V* − V_n‖∞ (V*, politika yinelemesiyle kesin), sıkı sınır γε_n/(1 − γ) ve kaynaktaki ε_n/(1 − γ); V_n'ye göre açgözlü
// politika π_n'nin kaybı ‖V* − V^{π_n}‖∞ (V^π = (I − γT_π)⁻¹R_π) ve sınırı 2γε_n/(1 − γ). MDP yapıları: yoğun rastgele geçişler, seyrek
// (her eylem 2 ardıla gider), halka (yavaş karışan: eylemler bir adım ileri/geri, küçük gürültü).
import { fmt, range } from "./plot.js";

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } }
  const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = s / M[i][i]; }
  return x;
}
export function mdp(S, A, kind, r) {
  const P = range(A).map(() => range(S).map(() => Array(S).fill(0))), R = range(A).map(() => range(S).map(() => 2 * r() - 1));
  for (let a = 0; a < A; a++) for (let s = 0; s < S; s++) {
    const row = P[a][s];
    if (kind === 0) { let t = 0; for (let j = 0; j < S; j++) { row[j] = -Math.log(r() + 1e-12); t += row[j]; } for (let j = 0; j < S; j++) row[j] /= t; }
    else if (kind === 1) { const j1 = Math.floor(r() * S), j2 = Math.floor(r() * S), w = r(); row[j1] += w; row[j2] += 1 - w; }
    else { const d = a % 2 === 0 ? 1 : -1; row[(s + d + S) % S] += 0.9; row[s] += 0.1; }
  }
  return { P, R };
}
const bell = (m, V, g) => { const S = V.length, A = m.P.length, out = Array(S), pi = Array(S); for (let s = 0; s < S; s++) { let best = -Infinity, ba = 0; for (let a = 0; a < A; a++) { let q = m.R[a][s]; const row = m.P[a][s]; for (let j = 0; j < S; j++) q += g * row[j] * V[j]; if (q > best + 1e-12) { best = q; ba = a; } } out[s] = best; pi[s] = ba; } return { V: out, pi }; };
const evalPi = (m, pi, g) => { const S = pi.length; return solve(range(S).map((s) => range(S).map((j) => (s === j ? 1 : 0) - g * m.P[pi[s]][s][j])), range(S).map((s) => m.R[pi[s]][s])); };
const nrm = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
export default {
  params: [
    { id: "kind", label: "MDP yapısı", options: ["yoğun rastgele", "seyrek (2 ardıl)", "halka (yavaş karışan)"], value: 2 },
    { id: "S", label: "Durum sayısı", min: 4, max: 40, value: 20, int: true },
    { id: "A", label: "Eylem sayısı", min: 2, max: 6, value: 3, int: true },
    { id: "g", label: "İndirim γ", min: 0.5, max: 0.99, step: 0.01, value: 0.9 },
    { id: "ltol", label: "Durma eşiği ε = 10^k", min: -8, max: -1, step: 0.5, value: -3 },
  ],
  seed: true,
  run(q, r) {
    const m = mdp(q.S, q.A, q.kind, r), g = q.g, tol = 10 ** q.ltol;
    // V*: politika yinelemesi
    let pi = Array(q.S).fill(0), Vs;
    for (let it = 0; it < 100; it++) { Vs = evalPi(m, pi, g); const nb = bell(m, Vs, g).pi; if (nb.every((a, i) => a === pi[i])) break; pi = nb; }
    const N = Math.min(2000, Math.ceil(Math.log(tol * (1 - g) / 4) / Math.log(g)) + 5);
    let V = Array(q.S).fill(0); const rows = []; let stop = -1;
    for (let n = 1; n <= N; n++) {
      const { V: nv, pi: p } = bell(m, V, g), eps = nrm(nv, V), Vp = evalPi(m, bell(m, nv, g).pi, g);
      rows.push({ n, eps, err: nrm(Vs, nv), loss: nrm(Vs, Vp) }); V = nv;
      if (stop < 0 && eps <= tol) { stop = n; }
      if (stop > 0 && n >= stop + Math.max(5, Math.ceil(stop * 0.25))) break;
    }
    const st = rows[stop - 1] || rows[rows.length - 1], ns = rows.map((x) => x.n), fl = (v) => Math.max(v, 1e-16);
    const P = [`ε = ${fmt(tol, 2)} eşiği ${stop > 0 ? `${stop}. yinelemede` : "bu sürede"} aşılıyor. O anda gerçek hata ‖V* − V_n‖ = ${fmt(st.err, 3)}; sıkı sınır γε/(1 − γ) = ${fmt((g * st.eps) / (1 - g), 3)}, kaynaktaki ε/(1 − γ) = ${fmt(st.eps / (1 - g), 3)}.`];
    P.push(`Açgözlü politikanın kaybı ‖V* − V^π‖ = ${fmt(st.loss, 3)} (sınır 2γε/(1 − γ) = ${fmt((2 * g * st.eps) / (1 - g), 3)})${st.loss === 0 ? ": politika zaten en iyi" : ""}.`);
    let li = rows.length; while (li > 0 && rows[li - 1].loss < 1e-9) li--; const firstOpt = li < rows.length ? rows[li] : null, early = rows.find((x) => x.loss < 1e-9);
    if (firstOpt) P.push(`Açgözlü politika ${firstOpt.n}. yinelemeden itibaren kalıcı olarak en iyi${early && early.n < firstOpt.n ? ` (ilk kez ${early.n}. yinelemede en iyi oldu, sonra yeniden saptı: yakınsama politika için tekdüze değil)` : ""}; değerler o anda hâlâ ${fmt(firstOpt.err, 3)} uzakta. Politika, değerlerden çok önce yakınsar (M11.3'teki politika yinelemesinin hızının nedeni).`);
    if (st.err > 0.95 * (g * st.eps) / (1 - g)) P.push("Sıkı sınır burada neredeyse eşitlikle sağlanıyor: V₀ = 0'dan başlayınca hataya bütün durumlarda aynı olan bir kayma hâkim ve bu kayma her yinelemede tam γ ile küçülür.");
    if (q.kind === 0) P.push("Yoğun rastgele geçişlerde bütün eylemler benzer ardıl dağılımlarına gider; kararı anlık ödül belirler ve açgözlü politika ilk yinelemede doğru çıkar.");
    if (q.kind === 2) P.push("Halka yapısında bilgi her yinelemede yalnız bir komşuya yayılır; hata oranı yine en çok γ, ama sınırlar daha gevşek kalır.");
    return {
      plots: [{ title: "Yineleme boyunca hata ve sınırlar (log)", xlabel: "yineleme n", ylabel: "∞-norm", wide: true, ylog: true, size: { w: 620, h: 250 },
        series: [{ name: "ε_n = ‖V_n − V_{n−1}‖", x: ns, y: rows.map((x) => fl(x.eps)), c: 4, dash: true }, { name: "‖V* − V_n‖", x: ns, y: rows.map((x) => fl(x.err)), c: 0 }, { name: "sınır γε/(1 − γ)", x: ns, y: rows.map((x) => fl((g * x.eps) / (1 - g))), c: 0, dash: true }, { name: "‖V* − V^{π_n}‖ (0 olduğunda çizilmez)", x: ns, y: rows.map((x) => (x.loss < 1e-12 ? NaN : x.loss)), c: 2, dots: true }],
        marks: stop > 0 ? [{ x: stop, label: "durma" }] : [] }],
      stats: [{ label: "Durma yinelemesi", value: stop > 0 ? String(stop) : "—" }, { label: "Gerçek hata / sınır", value: `${fmt(st.err, 3)} / ${fmt((g * st.eps) / (1 - g), 3)}` }, { label: "Politika kaybı", value: fmt(st.loss, 3) }, { label: "Politika kalıcı en iyi", value: firstOpt ? `n = ${firstOpt.n}` : "—" }],
      note: P.join(" "),
    };
  },
};
