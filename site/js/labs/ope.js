// M11.8 — Kayıtlı veriden politika dışı değerlendirme. Bağlam x ~ N(0, I_d), K kol, gerçek ödül r = θ_aᵀx + β·(x₁² − 1) + ε (β: doğrusal
// modelin yakalayamadığı terim), ε ~ N(0, 0.5²). Kayıt politikası: ε_log olasılıkla düzgün, aksi hâlde gürültülü bir modele göre
// açgözlü; eğilim (propensity) p(a|x) bilinir. Değerlendirilen politika: gerçek doğrusal kısma göre açgözlü. Kestiriciler: DM (her kol
// için kayıtlı veride sırt regresyonu, π'nin eyleminin öngörülen ödülü), IPS (r·1[a = π(x)]/p), SNIPS (öz-normalize IPS), DR
// (DM + IPS düzeltmesi). R tekrarda yanlılık, standart sapma ve RMSE; gerçek değer 20 000 taze bağlamla Monte Carlo.
import { fmt, range } from "./plot.js";

function ridge(X, y, lam) { const d = X[0] ? X[0].length : 0, A = range(d).map((i) => range(d).map((j) => (i === j ? lam : 0))), b = Array(d).fill(0);
  X.forEach((x, n) => { for (let i = 0; i < d; i++) { b[i] += x[i] * y[n]; for (let j = 0; j < d; j++) A[i][j] += x[i] * x[j]; } });
  const M = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < d; c++) { let p = c; for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = 0; r < d; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= d; k++) M[r][k] -= f * M[c][k]; } }
  return M.map((r, i) => r[d] / r[i]); }
export default {
  params: [
    { id: "n", label: "Kayıt sayısı n", min: 100, max: 5000, step: 100, value: 1000, int: true },
    { id: "eps", label: "Kayıt politikasının keşif oranı ε_log", min: 0.02, max: 1, step: 0.02, value: 0.2 },
    { id: "beta", label: "Modelin yakalayamadığı terim β", min: 0, max: 1, step: 0.05, value: 0.3 },
    { id: "R", label: "Tekrar", min: 10, max: 200, step: 10, value: 50, int: true },
  ],
  seed: true,
  run(q, rnd) {
    const d = 4, K = 4, th = range(K).map(() => range(d).map(() => rnd.normal() / Math.sqrt(d))), noisy = th.map((w) => w.map((v) => v + 0.7 * rnd.normal() / Math.sqrt(d)));
    const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0), arg = (W, x) => { let b = 0, bv = -Infinity; W.forEach((w, k) => { const v = dot(w, x); if (v > bv) { bv = v; b = k; } }); return b; };
    const reward = (a, x) => dot(th[a], x) + q.beta * (x[0] * x[0] - 1) + 0.5 * rnd.normal();
    const pi = (x) => arg(th, x), plog = (x) => { const g = arg(noisy, x); return range(K).map((k) => q.eps / K + (k === g ? 1 - q.eps : 0)); };
    let truth = 0; for (let i = 0; i < 20000; i++) { const x = range(d).map(() => rnd.normal()); truth += (dot(th[pi(x)], x)) / 20000; }
    const est = { DM: [], IPS: [], SNIPS: [], DR: [] };
    for (let k = 0; k < q.R; k++) {
      const X = [], A = [], Y = [], Pp = [];
      for (let i = 0; i < q.n; i++) { const x = range(d).map(() => rnd.normal()), p = plog(x); let u = rnd(), a = 0; while (a < K - 1 && u >= p[a]) { u -= p[a]; a++; } X.push(x); A.push(a); Y.push(reward(a, x)); Pp.push(p[a]); }
      const W = range(K).map((a) => { const idx = A.map((v, i) => (v === a ? i : -1)).filter((i) => i >= 0); return idx.length ? ridge(idx.map((i) => X[i]), idx.map((i) => Y[i]), 1) : Array(d).fill(0); });
      let dm = 0, ips = 0, wsum = 0, dr = 0;
      for (let i = 0; i < q.n; i++) { const ap = pi(X[i]), m = dot(W[ap], X[i]), w = A[i] === ap ? 1 / Pp[i] : 0; dm += m; ips += w * Y[i]; wsum += w; dr += m + w * (Y[i] - dot(W[A[i]], X[i])); }
      est.DM.push(dm / q.n); est.IPS.push(ips / q.n); est.SNIPS.push(wsum > 0 ? ips / wsum : 0); est.DR.push(dr / q.n);
    }
    const names = ["DM", "IPS", "SNIPS", "DR"], mean = (a) => a.reduce((x, y) => x + y, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
    const B = names.map((n) => mean(est[n]) - truth), SD = names.map((n) => sd(est[n])), RM = names.map((n, i) => Math.sqrt(B[i] ** 2 + SD[i] ** 2));
    const P = [`Gerçek değer V(π) = ${fmt(truth, 3)}. ${names.map((n, i) => `${n}: yanlılık ${fmt(B[i], 2)}, std ${fmt(SD[i], 2)}, RMSE ${fmt(RM[i], 2)}`).join("; ")}.`];
    P.push(q.beta > 0 ? `β = ${q.beta}: doğrusal ödül modeli yanlış belirlenmiş, DM yanlı. IPS ve DR yansız; DR'nin varyansı IPS'ten düşük, çünkü modelin açıkladığı kısmı çıkarıp yalnız artığı ağırlıklandırıyor.` : "β = 0: model doğru belirlenmiş; DM hem yansız hem en düşük varyanslı. IPS gereksiz yere gürültülü.");
    P.push(`En küçük eğilim ${fmt(q.eps / K, 2)}: ağırlıklar 1/p'ye kadar büyür. ε_log küçüldükçe IPS'in varyansı patlar.`);
    return {
      plots: [{ title: "Kestiricilerin tekrarlar boyunca değerleri", xlabel: "kestirici", ylabel: "kestirilen V(π)", wide: true, size: { w: 620, h: 240 }, xlim: [-0.6, 3.6], xticks: names.map((n, i) => [i, n]),
        series: [...names.map((n, i) => ({ name: i === 0 ? "tekrarlar" : "", x: est[n].map(() => i + (rnd() - 0.5) * 0.3), y: est[n], kind: "scatter", c: 0 })), { name: "gerçek değer", x: [-0.5, 3.5], y: [truth, truth], c: 2, dash: true }] }],
      stats: names.map((n, i) => ({ label: `${n}: yanlılık / RMSE`, value: `${fmt(B[i], 2)} / ${fmt(RM[i], 2)}` })),
      note: P.join(" "),
    };
  },
};
