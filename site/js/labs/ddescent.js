// M3.8 — Çift iniş (double descent): gizli faktör modelinde (Hastie ve ark. 2022) en küçük normlu (ya da ridge) en küçük kareler.
// Veri 5 boyutlu gizli z'den üretilir: her öznitelik xⱼ = wⱼᵀz + ν·gürültü, hedef y = θᵀz + σ·gürültü. Model ilk p özniteliği kullanır.
// p, n'yi geçerken (ara değerleme eşiği p = n) test hatası tepe yapar, sonra yeniden düşer — burada klasik bölgenin en iyisinin de altına.
import { fmt, range } from "./plot.js";

// simetrik pozitif tanımlı A x = b (Cholesky); A küçük ve λ > 0 ile koşullandırılmış
export function cholSolve(A, b) {
  const n = A.length, Lm = A.map((r) => r.map(() => 0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
    if (i === j) { if (!(s > 0)) s = 1e-300; Lm[i][i] = Math.sqrt(s); } else Lm[i][j] = s / Lm[j][j];
  }
  const y = new Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lm[i][k] * y[k]; y[i] = s / Lm[i][i]; }
  const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= Lm[k][i] * x[k]; x[i] = s / Lm[i][i]; }
  return x;
}
// ridge çözümü w = argmin ‖Φw − y‖² + λ‖w‖²: p ≤ n ise (ΦᵀΦ + λI)w = Φᵀy, p > n ise w = Φᵀ(ΦΦᵀ + λI)⁻¹y (ikisi aynı çözüm)
export function ridge(Phi, y, lam) {
  const n = Phi.length, p = Phi[0].length;
  if (p <= n) { const A = range(p).map((i) => range(p).map((j) => Phi.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? lam : 0))), b = range(p).map((i) => Phi.reduce((s, r, k) => s + r[i] * y[k], 0)); return cholSolve(A, b); }
  const A = range(n).map((i) => range(n).map((j) => Phi[i].reduce((s, v, k) => s + v * Phi[j][k], 0) + (i === j ? lam : 0))), a = cholSolve(A, y);
  return range(p).map((k) => Phi.reduce((s, r, i) => s + r[k] * a[i], 0));
}

export default {
  params: [
    { id: "n", label: "Eğitim örneği sayısı n", min: 8, max: 60, value: 20, int: true },
    { id: "sigma", label: "Etiket gürültüsü σ", min: 0, max: 0.6, step: 0.02, value: 0.1 },
    { id: "nu", label: "Öznitelik gürültüsü ν (her öznitelik gizli faktörlerin gürültülü bir karışımı)", min: 0.05, max: 2, step: 0.05, value: 0.3 },
    { id: "lam", label: "Ridge λ (≈ 0 → en küçük normlu çözüm)", min: 1e-9, max: 10, value: 1e-9, log: true },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), P = 300, d = 5, D = 6, T = 300;
    const grid = [...new Set([1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 50, 60, 70, 85, 100, 120, 150, 200, 250, 300, n - 2, n - 1, n, n + 1, n + 2].filter((p) => p >= 1 && p <= P))].sort((a, b) => a - b);
    const te = grid.map(() => 0), tr = grid.map(() => 0), wn = grid.map(() => 0);
    for (let k = 0; k < D; k++) {
      const W = range(P).map(() => range(d).map(() => r.normal() / Math.sqrt(d))), th = range(d).map(() => r.normal() / Math.sqrt(d));
      const gen = (m) => { const X = [], Y = []; for (let i = 0; i < m; i++) { const z = range(d).map(() => r.normal()); X.push(W.map((w) => w.reduce((s, v, j) => s + v * z[j], 0) + q.nu * r.normal())); Y.push(th.reduce((s, v, j) => s + v * z[j], 0) + q.sigma * r.normal()); } return { X, Y }; };
      const A = gen(n), B = gen(T);
      grid.forEach((p, g) => {
        const w = ridge(A.X.map((x) => x.slice(0, p)), A.Y, q.lam), pr = (x) => { let s = 0; for (let j = 0; j < p; j++) s += x[j] * w[j]; return s; };
        te[g] += B.X.reduce((s, x, i) => s + (pr(x) - B.Y[i]) ** 2, 0) / T / D; tr[g] += A.X.reduce((s, x, i) => s + (pr(x) - A.Y[i]) ** 2, 0) / n / D; wn[g] += Math.hypot(...w) / D;
      });
    }
    const iN = grid.indexOf(n), under = grid.map((p, i) => [p, te[i]]).filter(([p]) => p < n - 2), over = grid.map((p, i) => [p, te[i]]).filter(([p]) => p > n + 2), best = (A) => A.reduce((m, v) => (v[1] < m[1] ? v : m), [0, Infinity]);
    const bu = best(under), bo = best(over), xt = [1, 3, 10, 30, 100, 300].map((v) => [Math.log10(v), String(v)]);
    return {
      plots: [
        { title: `Test ve eğitim hatası, öznitelik sayısına göre (${D} veri setinin ortalaması)`, xlabel: "öznitelik sayısı p (log)", ylabel: "MSE", ylog: true, xticks: xt, series: [{ name: "test", x: grid.map(Math.log10), y: te.map((v) => Math.max(v, 1e-6)), c: 2, dots: true }, { name: "eğitim", x: grid.map(Math.log10), y: tr.map((v) => Math.max(v, 1e-6)), c: 0, dots: true }], marks: [{ x: Math.log10(n), label: "p = n" }, ...(q.sigma > 0 ? [{ y: q.sigma * q.sigma, label: "σ²" }] : [])] },
        { title: "Çözümün normu ‖w‖", xlabel: "öznitelik sayısı p (log)", ylabel: "‖w‖", ylog: true, xticks: xt, series: [{ name: "‖w‖", x: grid.map(Math.log10), y: wn.map((v) => Math.max(v, 1e-6)), c: 3, dots: true }], marks: [{ x: Math.log10(n), label: "p = n" }] },
      ],
      stats: [
        { label: "Test MSE, p = n (ara değerleme eşiği)", value: te[iN] },
        { label: "En iyi test MSE, p < n − 2 (klasik bölge)", value: under.length ? `${fmt(bu[1], 3)} (p = ${bu[0]})` : "—" },
        { label: "En iyi test MSE, p > n + 2 (aşırı parametreli)", value: over.length ? `${fmt(bo[1], 3)} (p = ${bo[0]})` : "—" },
        { label: "Eğitim MSE, p = 2n", value: tr[grid.findIndex((p) => p >= 2 * n)] ?? tr[tr.length - 1] },
      ],
      note: `p < n: klasik bölge — öznitelik ekledikçe yanlılık azalır, ama p n'ye yaklaştıkça varyans patlar. p = n: denklem sistemi kare ve kötü koşullu; tam ara değerleyen tek çözüm gürültüyü de ezberlemek için çok büyük ağırlıklar ister (‖w‖ tepe yapar), test hatası tepe yapar. p > n: sonsuz sayıda ara değerleyen çözüm vardır; en küçük normlu olanı seçilir. p büyüdükçe bu çözümün normu küçülür ve gürültü birçok özniteliğe "yayılarak" zararsızlaşır — test hatası <b>yeniden düşer</b>${bo[1] < bu[1] ? `, bu ayarlarda klasik bölgenin en iyisinin (${fmt(bu[1], 3)}) de altına (${fmt(bo[1], 3)})` : ""}. ${q.lam > 1e-3 ? `<b>λ = ${fmt(q.lam, 2)}:</b> ridge tepeyi bastırır; iyi ayarlanmış düzenlileştirme çift inişi büyük ölçüde yok eder (Nakkiran ve ark. 2021).` : "λ'yı büyüt: tepe yumuşar."} Bu, iyi huylu ara değerlemenin (benign overfitting) bir oyuncak örneğidir; sonuç veri modeline bağlıdır (öznitelikler ilişkisiz ve her biri sinyalin küçük bir parçasını taşıyorsa ikinci iniş birincinin altına inmeyebilir).`,
    };
  },
};
