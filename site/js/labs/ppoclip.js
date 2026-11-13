// M11.12 — PPO'nun kırpılmış vekil amacı, K kollu haydutta (tek adımlı RL). Politika softmax(θ). Her yinelemede eski politikadan B örnek
// toplanır, avantaj A = r − ortalama(r). Aynı yığın üzerinde E epoch boyunca gradyan yükselişi (adım α):
// (0) kırpılmamış vekil L = mean[ρA], ρ = π_θ(a)/π_eski(a); (1) PPO: mean[min(ρA, clip(ρ, 1 − ε, 1 + ε)A)]; (2) KL cezası: mean[ρA] − β·KL(π_eski‖π_θ).
// Gerçek amaç J = Σ π(a)μ(a) ve her yinelemenin KL(π_eski‖π_yeni) adımı ölçülür; R tohumlu tekrarın ortalaması.
import { fmt, range } from "./plot.js";

const sm = (t) => { const m = Math.max(...t), e = t.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
const kl = (p, q) => p.reduce((s, v, i) => s + (v > 0 ? v * Math.log(v / q[i]) : 0), 0);
export function train(method, q, r) {
  const K = q.K, mu = range(K).map((a) => 1 - (0.8 * a) / (K - 1)); // kol 0 en iyi (1.0), en kötüsü 0.2
  let th = Array(K).fill(0); const J = [], KL = [];
  for (let it = 0; it < q.I; it++) {
    const pold = sm(th), acts = [], rews = [];
    for (let i = 0; i < q.B; i++) { let u = r(), a = 0; while (a < K - 1 && u >= pold[a]) { u -= pold[a]; a++; } acts.push(a); rews.push(mu[a] + q.noise * r.normal()); }
    const mr = rews.reduce((s, v) => s + v, 0) / q.B, A = rews.map((v) => v - mr);
    for (let e = 0; e < q.E; e++) {
      const p = sm(th), g = Array(K).fill(0);
      for (let i = 0; i < q.B; i++) {
        const a = acts[i], rho = p[a] / pold[a];
        let w = rho * A[i]; // dL/dθ = w · ∇log π(a)
        if (method === 1) { const clipped = (A[i] > 0 && rho > 1 + q.eps) || (A[i] < 0 && rho < 1 - q.eps); if (clipped) w = 0; }
        for (let k = 0; k < K; k++) g[k] += (w * ((k === a ? 1 : 0) - p[k])) / q.B;
      }
      if (method === 2) for (let k = 0; k < K; k++) g[k] -= q.beta * (p[k] - pold[k]); // ∇_θ KL(π_eski‖π_θ) = π_θ − π_eski
      th = th.map((v, k) => v + q.lr * g[k]);
    }
    const pn = sm(th); J.push(pn.reduce((s, v, k) => s + v * mu[k], 0)); KL.push(kl(pold, pn));
  }
  return { J, KL };
}
const MN = ["kırpılmamış vekil", "PPO kırpması", "KL cezası"];
export default {
  params: [
    { id: "K", label: "Kol sayısı", min: 2, max: 20, value: 10, int: true },
    { id: "B", label: "Yığın boyu B", min: 8, max: 256, step: 8, value: 32, int: true },
    { id: "E", label: "Aynı yığında epoch E", min: 1, max: 50, value: 10, int: true },
    { id: "lr", label: "Adım α", min: 0.1, max: 10, step: 0.1, value: 2 },
    { id: "eps", label: "Kırpma ε", min: 0.05, max: 0.5, step: 0.05, value: 0.2 },
    { id: "beta", label: "KL cezası β", min: 0.1, max: 10, step: 0.1, value: 1 },
    { id: "noise", label: "Ödül gürültüsü σ", min: 0, max: 2, step: 0.1, value: 1 },
    { id: "I", label: "Yineleme (yığın sayısı)", min: 5, max: 100, step: 5, value: 40, int: true },
  ],
  seed: true,
  run(q, r) {
    const R = 30, curves = [0, 1, 2].map(() => ({ J: Array(q.I).fill(0), KL: Array(q.I).fill(0), worst: Infinity, fin: [] }));
    for (let k = 0; k < R; k++) for (let m = 0; m < 3; m++) { const o = train(m, q, r); o.J.forEach((v, i) => (curves[m].J[i] += v / R)); o.KL.forEach((v, i) => (curves[m].KL[i] += v / R)); curves[m].fin.push(o.J[q.I - 1]); }
    const xs = range(q.I).map((i) => i + 1), mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const bad = curves.map((c) => c.fin.filter((v) => v < 0.9).length);
    const P = [`${R} tekrarın ortalamasında ${q.I} yinelemeden sonra J (en iyi kol 1.0): ${MN.map((n, i) => `${n} ${fmt(curves[i].J[q.I - 1], 3)}`).join(", ")}. İlk yinelemede ortalama KL(π_eski‖π_yeni): ${MN.map((n, i) => `${fmt(curves[i].KL[0], 3)}`).join(" / ")}.`];
    P.push(`${q.I} yineleme sonunda J < 0.9 kalan (alt-en iyi kola kilitlenen ya da yavaş kalan) tekrar sayısı: ${MN.map((n, i) => `${n} ${bad[i]}`).join(", ")}.`);
    if (q.E === 1) P.push("E = 1: ilk gradyan adımında ρ = 1 olduğu için üç yöntem aynı gradyanı verir (kırpma ancak politika eskisinden uzaklaşınca devreye girer).");
    else P.push(`Aynı yığında ${q.E} epoch: kırpılmamış vekil, örneklemedeki şansı (gürültülü avantajları) defalarca sömürür ve politikayı tek yığına göre aşırı itiyor. Kırpma, oran 1 ± ε'u aşınca gradyanı sıfırlar; KL cezası uzaklaştıkça geri çeker.`);
    return {
      plots: [
        { title: "Gerçek amaç J (ortalama)", xlabel: "yineleme", ylabel: "J", side: true, size: { w: 300, h: 230 }, ylim: [0.2, 1.02], series: [0, 1, 2].map((m) => ({ name: MN[m], x: xs, y: curves[m].J, c: m })) },
        { title: "Yineleme başına KL(π_eski‖π_yeni) (log)", xlabel: "yineleme", ylabel: "KL", side: true, size: { w: 300, h: 230 }, ylog: true, series: [0, 1, 2].map((m) => ({ name: MN[m], x: xs, y: curves[m].KL.map((v) => Math.max(v, 1e-8)), c: m })) },
      ],
      stats: MN.map((n, i) => ({ label: `${n}: son J / J < 0.9 kalan`, value: `${fmt(curves[i].J[q.I - 1], 3)} / ${bad[i]}` })),
      note: P.join(" "),
    };
  },
};
