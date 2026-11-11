// M11.8 — Doğrusal bağlamlı haydut. Her adımda d boyutlu bir bağlam x ~ N(0, I) gelir; K kolun her biri için gerçek ödül
// r = θ_aᵀx + ε, ε ~ N(0, σ²) (ayrık modeller: her kolun kendi θ_a'sı). Yöntemler: LinUCB (her kol için Bayesçi doğrusal regresyon:
// A_a = λ⁻²I + σ⁻² Σ x xᵀ, b_a = σ⁻² Σ r x, μ = A⁻¹b, üst sınır μᵀx + c√(xᵀA⁻¹x)), doğrusal Thompson (w ~ N(μ, A⁻¹), bir örnek, bütün
// kollar için aynı çekiliş), bağlamı yok sayan UCB1 ve bağlam kullanan açgözlü (c = 0). R çalıştırmanın birikimli pişmanlığı.
// Ters matrisler Sherman–Morrison ile O(d²) güncellenir.
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "d", label: "Bağlam boyutu d", min: 2, max: 20, value: 5, int: true },
    { id: "K", label: "Kol sayısı K", min: 2, max: 10, value: 5, int: true },
    { id: "sigma", label: "Gürültü σ", min: 0.05, max: 2, step: 0.05, value: 0.5 },
    { id: "c", label: "LinUCB güven çarpanı c", min: 0, max: 3, step: 0.1, value: 1 },
    { id: "T", label: "Adım sayısı", min: 200, max: 3000, step: 100, value: 1500, int: true },
    { id: "R", label: "Çalıştırma", min: 1, max: 30, value: 10, int: true },
  ],
  seed: true,
  run(q, r) {
    const { d, K, T } = q, s2 = q.sigma * q.sigma, M = 4, reg = range(M).map(() => Array(T).fill(0));
    for (let run = 0; run < q.R; run++) {
      const th = range(K).map(() => range(d).map(() => r.normal() / Math.sqrt(d)));
      const st = range(M).map(() => ({ Ai: range(K).map(() => range(d).map((i) => range(d).map((j) => (i === j ? 1 : 0)))), b: range(K).map(() => Array(d).fill(0)), S: Array(K).fill(0), N: Array(K).fill(0), cum: 0 }));
      for (let t = 0; t < T; t++) {
        const x = range(d).map(() => r.normal()), tr = th.map((w) => w.reduce((a, v, i) => a + v * x[i], 0)), best = Math.max(...tr);
        const z = range(d).map(() => r.normal());
        for (let m = 0; m < M; m++) {
          const S = st[m]; let a = 0;
          if (m === 2) { if (t < K) a = t; else { let bv = -Infinity; for (let k = 0; k < K; k++) { const u = S.S[k] / S.N[k] + Math.sqrt((2 * Math.log(t + 1)) / S.N[k]); if (u > bv) { bv = u; a = k; } } } }
          else {
            let bv = -Infinity;
            for (let k = 0; k < K; k++) {
              const Ai = S.Ai[k], mu = Ai.map((row) => row.reduce((acc, v, j) => acc + v * S.b[k][j], 0)), Ax = Ai.map((row) => row.reduce((acc, v, j) => acc + v * x[j], 0)), v2 = x.reduce((acc, v, i) => acc + v * Ax[i], 0);
              let score;
              if (m === 0) score = mu.reduce((acc, v, i) => acc + v * x[i], 0) + q.c * Math.sqrt(Math.max(v2, 0));
              else if (m === 1) score = mu.reduce((acc, v, i) => acc + v * x[i], 0) + Math.sqrt(Math.max(v2, 0)) * (z.reduce((acc, v, i) => acc + v * x[i], 0) / Math.sqrt(x.reduce((acc, v) => acc + v * v, 0)));
              else score = mu.reduce((acc, v, i) => acc + v * x[i], 0);
              if (score > bv) { bv = score; a = k; }
            }
          }
          const rw = tr[a] + q.sigma * r.normal(); S.cum += best - tr[a]; reg[m][t] += S.cum / q.R;
          if (m === 2) { S.S[a] += rw; S.N[a]++; continue; }
          // Sherman–Morrison: A ← A + x xᵀ / σ²
          const Ai = S.Ai[a], Ax = Ai.map((row) => row.reduce((acc, v, j) => acc + v * x[j], 0)), den = s2 + x.reduce((acc, v, i) => acc + v * Ax[i], 0);
          for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ai[i][j] -= (Ax[i] * Ax[j]) / den;
          for (let i = 0; i < d; i++) S.b[a][i] += (rw * x[i]) / s2;
        }
      }
    }
    const names = ["LinUCB", "doğrusal Thompson", "UCB1 (bağlamsız)", "açgözlü (bağlamlı)"], ts = range(T).map((t) => t + 1);
    const fin = reg.map((a) => a[T - 1]), last = reg.map((a) => a[T - 1] - a[Math.floor(T * 0.8)]);
    const P = [`${T} adımda ortalama birikimli pişmanlık: ${names.map((n, i) => `${n} ${fmt(fin[i], 3)}`).join(", ")}.`];
    P.push(`Son %20'de eklenen: ${names.map((n, i) => `${n} ${fmt(last[i], 3)}`).join(", ")}. Bağlamsız UCB1 her kolun ortalamasını öğrenir; ama gerçek ödüller bağlamla değişir ve ortalamaları sıfıra yakındır: doğrusal pişmanlık yapar.`);
    if (fin[3] < fin[0] && fin[3] < fin[1]) P.push("Bu ayarda bağlamlı açgözlü de iyi: bağlamlar N(0, I)'den bütün yönlerde geldiği için açgözlü seçim bile her kolu yeterince farklı bağlamda dener (Bastani ve ark. 2021).");
    if (q.c === 0) P.push("c = 0: LinUCB, bağlamlı açgözlüyle aynı algoritmadır; iki sayı yalnız farklı gürültü örnekleri yüzünden ayrı.");
    return {
      plots: [{ title: "Birikimli pişmanlık", xlabel: "adım", ylabel: "pişmanlık", wide: true, size: { w: 620, h: 250 }, series: names.map((n, i) => ({ name: n, x: ts.filter((_, j) => j % 5 === 0), y: reg[i].filter((_, j) => j % 5 === 0), c: i })) }],
      stats: names.map((n, i) => ({ label: n, value: fmt(fin[i], 3) })),
      note: P.join(" "),
    };
  },
};
