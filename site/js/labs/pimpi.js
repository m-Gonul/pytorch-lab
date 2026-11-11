// M11.3 — Değer yinelemesi (VI), politika yinelemesi (PI) ve değiştirilmiş politika yinelemesinin (MPI, K değerlendirme taraması) aynı
// tohumlu MDP'de karşılaştırılması. Hata ‖V_n − V*‖∞ (PI'da V_n = V^{π_n}); iş, çarpma–toplama sayısıyla ve geçişlerin seyrekliği hesaba
// katılarak: bir yedekleme taraması Σ_a nnz(P_a), politikanın değerlendirme taraması nnz(P_π), PI'nin doğrusal sistemi |S|³/3 (yoğun LU).
// VI yinelemesi = tam tarama; PI = LU + tam tarama (iyileştirme); MPI = K · nnz(P_π) + tam tarama. V* politika yinelemesiyle kesin.
// İkinci grafik PI'nin monoton iyileşmesini gösterir: her yinelemede değerlerin ortalaması ve değişen eylem sayısı.
import { fmt, range } from "./plot.js";
import { mdp } from "./vibounds.js";

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } }
  const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = s / M[i][i]; }
  return x;
}
const q1 = (m, V, g, s, a) => { let q = m.R[a][s]; const row = m.P[a][s]; for (let j = 0; j < V.length; j++) q += g * row[j] * V[j]; return q; };
const greedy = (m, V, g) => V.map((_, s) => { let b = 0, bv = -Infinity; for (let a = 0; a < m.P.length; a++) { const v = q1(m, V, g, s, a); if (v > bv + 1e-12) { bv = v; b = a; } } return b; });
const backup = (m, V, g) => V.map((_, s) => Math.max(...range(m.P.length).map((a) => q1(m, V, g, s, a))));
const evalPi = (m, pi, g) => solve(pi.map((a, s) => pi.map((_, j) => (s === j ? 1 : 0) - g * m.P[a][s][j])), pi.map((a, s) => m.R[a][s]));
const nrm = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
export default {
  params: [
    { id: "kind", label: "MDP yapısı", options: ["yoğun rastgele", "seyrek (2 ardıl)", "halka (yavaş karışan)"], value: 2 },
    { id: "S", label: "Durum sayısı", min: 4, max: 40, value: 30, int: true },
    { id: "A", label: "Eylem sayısı", min: 2, max: 6, value: 3, int: true },
    { id: "g", label: "İndirim γ", min: 0.5, max: 0.99, step: 0.01, value: 0.95 },
    { id: "K", label: "MPI değerlendirme taraması K", min: 1, max: 60, value: 5, int: true },
  ],
  seed: true,
  run(q, r) {
    const m = mdp(q.S, q.A, q.kind, r), g = q.g, S = q.S, A = q.A, tol = 1e-8;
    let pi = Array(S).fill(0), Vs = null; const piHist = [];
    for (let it = 0; it < 200; it++) { Vs = evalPi(m, pi, g); const nb = greedy(m, Vs, g); piHist.push({ V: Vs, ch: nb.filter((a, i) => a !== pi[i]).length }); if (nb.every((a, i) => a === pi[i])) break; pi = nb; }
    const nnz = m.P.map((Pa) => Pa.reduce((t, row) => t + row.filter((v) => v > 0).length, 0)), full = nnz.reduce((a, b) => a + b, 0), perPi = full / A;
    const cVI = full, cPI = (S * S * S) / 3 + full, cMPI = q.K * perPi + full, cap = 4000;
    // VI
    const vi = []; let V = Array(S).fill(0);
    for (let n = 1; n <= cap; n++) { V = backup(m, V, g); const e = nrm(V, Vs); vi.push([n * cVI, e]); if (e < tol) break; }
    // PI (π₀ = hep eylem 0)
    const piR = piHist.map((h, n) => [(n + 1) * cPI, nrm(h.V, Vs)]);
    // MPI
    const mpi = []; let W = Array(S).fill(0), pm = greedy(m, W, g);
    for (let n = 1; n <= cap; n++) {
      for (let k = 0; k < q.K; k++) W = W.map((_, s) => q1(m, W, g, s, pm[s]));
      pm = greedy(m, W, g); const e = nrm(W, Vs); mpi.push([n * cMPI, e]); if (e < tol) break;
    }
    const at = (arr, e) => { const i = arr.findIndex((x) => x[1] < e); return i < 0 ? null : { n: i + 1, w: arr[i][0] }; };
    const t6 = [vi, piR, mpi].map((a) => at(a, 1e-6)), names = ["VI", "PI", `MPI (K = ${q.K})`];
    const fl = (v) => Math.max(v, 1e-12), M6 = 1e6;
    const P = [`10⁻⁶ hataya: ${names.map((nm, i) => `${nm} ${t6[i] ? `${t6[i].n} yineleme, ${fmt(t6[i].w / M6, 3)} M işlem` : "ulaşmadı"}`).join("; ")}.`];
    P.push(`Geçiş matrislerinde ${fmt(full / A, 4)} sıfır olmayan öğe var (eylem başına; yoğun olsaydı ${S * S}). Bir yedekleme taraması ${full} işlem, PI'nin doğrusal sistemi ≈ ${fmt((S * S * S) / 3, 4)} işlem.`);
    P.push(`PI ${piHist.length} yinelemede bitti (son yinelemede politika değişmedi); her yinelemede ${S} × ${S} doğrusal sistem çözülüyor.`);
    const best = t6.map((x, i) => (x ? [x.w, i] : [Infinity, i])).sort((a, b) => a[0] - b[0])[0][1];
    P.push(`Bu ayarda en az işle biten: ${names[best]}.`);
    if (q.K === 1) P.push("K = 1: MPI her yinelemede tek değerlendirme taraması yapar; yalnız açgözlü politikayla bir yedekleme demektir — değer yinelemesiyle aynıdır.");
    if (q.K >= 40) P.push("Büyük K: MPI her politikayı neredeyse tam değerlendirir ve politika yinelemesine yaklaşır; yineleme sayısı azalır, yineleme başına iş artar.");
    const mono = piHist.map((h) => h.V.reduce((a, b) => a + b, 0) / S);
    return {
      plots: [
        { title: "Hata ve hesap işi (log)", xlabel: "çarpma–toplama (milyon)", ylabel: "‖V − V*‖∞", wide: true, ylog: true, size: { w: 620, h: 240 },
          series: [{ name: "VI", x: vi.map((x) => x[0] / M6), y: vi.map((x) => fl(x[1])), c: 0 }, { name: "PI", x: piR.map((x) => x[0] / M6), y: piR.map((x) => fl(x[1])), c: 2, dots: true }, { name: `MPI (K = ${q.K})`, x: mpi.map((x) => x[0] / M6), y: mpi.map((x) => fl(x[1])), c: 1 }] },
        { title: "PI: değerlerin ortalaması (monoton) ve değişen eylem sayısı", xlabel: "PI yinelemesi", ylabel: "ort. V / değişen eylem", wide: true, size: { w: 620, h: 190 },
          series: [{ name: "ortalama V^{π_n}", x: range(piHist.length), y: mono, c: 2, dots: true }, { name: "değişen eylem", x: range(piHist.length), y: piHist.map((h) => h.ch), c: 4, kind: "bar" }] },
      ],
      stats: names.map((nm, i) => ({ label: `${nm}: yineleme / iş (M)`, value: t6[i] ? `${t6[i].n} / ${fmt(t6[i].w / M6, 3)}` : "—" })),
      note: P.join(" "),
    };
  },
};
