// M1.3 — Polinom regresyonu: derece ve ridge λ ile eğitim/test hatası (yanlılık–varyans)
import { fmt, linspace } from "./plot.js";
const truef = (x) => Math.sin(2.5 * x) + 0.3 * x;
function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < n; c++) { let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r; [M[c], M[piv]] = [M[piv], M[c]]; const d = M[c][c] || 1e-300; for (let r = 0; r < n; r++) if (r !== c) { const k = M[r][c] / d; for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j]; } } return M.map((r, i) => r[n] / (r[i] || 1e-300)); }
// Legendre tabanı: [-1,1]'de iyi koşullu
function feats(x, d) { const P = [1, x]; for (let k = 1; k < d; k++) P.push(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)); return P.slice(0, d + 1); }
function fit(X, Y, d, lam) { const F = X.map((x) => feats(x, d)), m = d + 1; const A = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => F.reduce((s, f) => s + f[i] * f[j], 0) + (i === j && i > 0 ? lam : 0))); const b = Array.from({ length: m }, (_, i) => F.reduce((s, f, k) => s + f[i] * Y[k], 0)); return solve(A, b); }
const pred = (w, x) => feats(x, w.length - 1).reduce((s, f, i) => s + f * w[i], 0);
export default {
  params: [
    { id: "d", label: "Polinom derecesi", min: 0, max: 15, step: 1, value: 9, int: true },
    { id: "lam", label: "Ridge λ", min: 1e-8, max: 10, log: true, value: 1e-8, hardMin: 0 },
    { id: "n", label: "Eğitim noktası n", min: 5, max: 200, step: 1, value: 15, int: true },
    { id: "sig", label: "Gürültü σ", min: 0, max: 1, step: 0.01, value: 0.25 },
  ],
  seed: true,
  validate: (p) => (p.d + 1 > p.n && p.lam < 1e-6 ? [{ level: "warn", msg: `Parametre sayısı (${p.d + 1}) ≥ veri sayısı (${p.n}) ve λ ≈ 0: çözüm tek değil, model veriyi tam ezberler (eğitim hatası ≈ 0).` }] : []),
  run(p, rnd) {
    const X = Array.from({ length: p.n }, () => 2 * rnd() - 1), Y = X.map((x) => truef(x) + p.sig * rnd.normal());
    const Xt = linspace(-1, 1, 200), Yt = Xt.map(truef);
    const mse = (w, xs, ys) => xs.reduce((s, x, i) => s + (pred(w, x) - ys[i]) ** 2, 0) / xs.length;
    const w = fit(X, Y, p.d, p.lam);
    const tr = mse(w, X, Y), te = mse(w, Xt, Yt) + p.sig ** 2;
    const lams = linspace(-8, 1, 28).map((e) => 10 ** e);
    const curve = lams.map((l) => { const ww = fit(X, Y, p.d, l); return [mse(ww, X, Y), mse(ww, Xt, Yt) + p.sig ** 2]; });
    const degs = Array.from({ length: 16 }, (_, d) => d);
    const dc = degs.map((d) => { const ww = fit(X, Y, d, p.lam); return [mse(ww, X, Y), mse(ww, Xt, Yt) + p.sig ** 2]; });
    const yl = [Math.min(...Yt) - 1, Math.max(...Yt) + 1];
    return {
      plots: [
        { title: "Veri ve uydurulan polinom", xlabel: "x", ylabel: "y", ylim: yl, series: [{ name: "gerçek f(x)", x: Xt, y: Yt, c: 1 }, { name: "eğitim verisi", x: X, y: Y, kind: "scatter", c: 3 }, { name: `derece ${p.d}`, x: Xt, y: Xt.map((x) => pred(w, x)), c: 0 }] },
        { title: "Hata – derece (bu λ ile)", xlabel: "derece", ylabel: "MSE", ylog: true, series: [{ name: "eğitim", x: degs, y: dc.map((c) => Math.max(c[0], 1e-12)), dots: true }, { name: "test (beklenen)", x: degs, y: dc.map((c) => c[1]), dots: true, c: 2 }], marks: [{ x: p.d, label: "seçili" }, { y: p.sig ** 2 || null, label: "gürültü tabanı σ²" }] },
        { title: "Hata – λ (bu dereceyle)", xlabel: "log₁₀ λ", ylabel: "MSE", ylog: true, series: [{ name: "eğitim", x: lams.map(Math.log10), y: curve.map((c) => Math.max(c[0], 1e-12)) }, { name: "test", x: lams.map(Math.log10), y: curve.map((c) => c[1]), c: 2 }], marks: [{ x: Math.log10(Math.max(p.lam, 1e-8)), label: "seçili" }] },
      ],
      stats: [{ label: "Eğitim MSE", value: tr }, { label: "Test MSE (gürültü dahil)", value: te }, { label: "Gürültü tabanı σ² (hiçbir model altına inemez)", value: p.sig ** 2 }, { label: "‖w‖₂", value: Math.hypot(...w) }],
      note: te > 3 * tr + p.sig ** 2 ? "Aşırı uyum: eğitim hatası test hatasından çok küçük — model gürültüyü öğreniyor (yüksek varyans). λ'yı artır, dereceyi düşür ya da n'yi büyüt." : tr > 2 * p.sig ** 2 + 0.02 ? "Eksik uyum: eğitim hatası bile gürültü tabanının çok üstünde (yüksek yanlılık)." : `Dengeli: test hatası gürültü tabanına (σ² = ${fmt(p.sig ** 2)}) yakın.`,
    };
  },
};
