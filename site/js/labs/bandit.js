// M11.7 — Çok kollu haydut: ε-greedy, UCB1 ve Thompson örneklemesi için birikimli pişmanlık
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "K", label: "Kol sayısı K", min: 2, max: 20, step: 1, value: 5, int: true },
    { id: "gap", label: "En iyi kol farkı Δ", min: 0.01, max: 0.5, step: 0.01, value: 0.1 },
    { id: "eps", label: "ε (ε-greedy)", min: 0, max: 0.5, step: 0.01, value: 0.1 },
    { id: "c", label: "UCB keşif katsayısı c", min: 0, max: 4, step: 0.05, value: 1.41, help: "UCB1: ortalama + c·√(ln t / n_i). Kuramsal değer √2." },
    { id: "T", label: "Ufuk T", min: 100, max: 5000, step: 100, value: 2000, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const mu = range(p.K).map((i) => (i === 0 ? 0.5 + p.gap : 0.5 - 0.02 * (i - 1)));
    const best = Math.max(...mu), R = 30, pts = 50, stepIdx = range(pts).map((i) => Math.round(((i + 1) * p.T) / pts));
    const beta = (a, b) => { const g = (k) => { if (k < 1) return g(k + 1) * rnd() ** (1 / k); const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = rnd.normal(); v = 1 + c * x; } while (v <= 0); v = v ** 3; const u = rnd(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }; const x = g(a), y = g(b); return x / (x + y); };
    const algs = { "ε-greedy": (n, s, t) => (rnd() < p.eps ? Math.floor(rnd() * p.K) : argmax(n.map((ni, i) => (ni ? s[i] / ni : Infinity)))), UCB1: (n, s, t) => argmax(n.map((ni, i) => (ni ? s[i] / ni + p.c * Math.sqrt(Math.log(t) / ni) : Infinity))), Thompson: (n, s) => argmax(n.map((ni, i) => beta(1 + s[i], 1 + ni - s[i]))) };
    function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
    const curves = {}, finals = {};
    for (const [name, pick] of Object.entries(algs)) {
      const acc = new Array(pts).fill(0);
      for (let r = 0; r < R; r++) {
        const n = new Array(p.K).fill(0), s = new Array(p.K).fill(0);
        let reg = 0, j = 0;
        for (let t = 1; t <= p.T; t++) { const a = pick(n, s, t); const x = rnd() < mu[a] ? 1 : 0; n[a]++; s[a] += x; reg += best - mu[a]; if (t === stepIdx[j]) acc[j++] += reg / R; }
      }
      curves[name] = acc; finals[name] = acc[pts - 1];
    }
    const lr = stepIdx.map((t) => (p.K - 1) * (Math.log(t) / p.gap) * 0.25);
    return {
      plots: [{ title: `Birikimli pişmanlık (${R} tekrar ortalaması)`, xlabel: "t", ylabel: "Σ (μ* − μ_a)", yzero: true, series: [...Object.entries(curves).map(([k, v], i) => ({ name: k, x: stepIdx, y: v, c: i })), { name: "∝ (K−1) ln t / Δ (şekil için)", x: stepIdx, y: lr, c: 4, dash: true }] }],
      stats: Object.entries(finals).map(([k, v]) => ({ label: `${k}: T'de pişmanlık`, value: v })).concat([{ label: "Sürekli rastgele seçimin pişmanlığı", value: p.T * (best - mu.reduce((a, b) => a + b, 0) / p.K) }]),
      note: `Sabit ε'lu ε-greedy sonsuza dek ε oranında keşfeder: pişmanlığı doğrusal büyür (≈ εΔT). UCB ve Thompson keşfi belirsizlikle orantılı yapar: pişmanlık ln T ile büyür (Lai–Robbins alt sınırı ∝ Σ Δ_i / KL). Δ küçüldükçe kolları ayırt etmek ≈ 1/Δ² örnek ister.`,
    };
  },
};
