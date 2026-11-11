// M11.9 — Politika gradyanı kestiricisinin varyansı ve taban çizgisi. Tek durumlu, K eylemli softmax politika π(a) ∝ exp(θ_a)
// (θ = 0'dan başlar: düzgün), eylem a'nın ödülü N(μ_a, σ²), μ_a = m + Δ·a/(K − 1). Tek örnekli kestirici ĝ = (r − b)∇_θ log π(a);
// ∇_θ log π(a) = e_a − π. Her b için beklenti ve kovaryans kapalı biçimde hesaplanır: E[ĝ] taban çizgisinden bağımsızdır (yansızlık),
// toplam varyans tr Cov[ĝ] b'ye göre ikinci derecedendir; en küçük varyanslı b* = E[r‖∇log π‖²]/E[‖∇log π‖²]. Ayrıca N örnekli
// Monte Carlo ile ölçülen varyans.
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "K", label: "Eylem sayısı K", min: 2, max: 10, value: 4, int: true },
    { id: "m", label: "Ödüllerin ortak düzeyi m", min: -10, max: 10, step: 0.5, value: 5 },
    { id: "D", label: "En iyi ile en kötü eylem farkı Δ", min: 0, max: 4, step: 0.1, value: 1 },
    { id: "sigma", label: "Ödül gürültüsü σ", min: 0, max: 3, step: 0.1, value: 1 },
    { id: "b", label: "Taban çizgisi b", min: -10, max: 10, step: 0.5, value: 0 },
  ],
  seed: true,
  run(q, r) {
    const K = q.K, pi = Array(K).fill(1 / K), mu = range(K).map((a) => q.m + (q.D * a) / Math.max(1, K - 1)), s2 = q.sigma ** 2;
    const grad = (a) => range(K).map((j) => (j === a ? 1 : 0) - pi[j]), n2 = (a) => grad(a).reduce((s, v) => s + v * v, 0);
    const Eg = range(K).map((j) => pi.reduce((s, p, a) => s + p * mu[a] * grad(a)[j], 0));
    const tv = (b) => { let t = 0; for (let a = 0; a < K; a++) t += pi[a] * ((mu[a] - b) ** 2 + s2) * n2(a); return t - Eg.reduce((s, v) => s + v * v, 0); };
    const bstar = pi.reduce((s, p, a) => s + p * mu[a] * n2(a), 0) / pi.reduce((s, p, a) => s + p * n2(a), 0);
    const N = 4000; let sum = Array(K).fill(0), sq = 0;
    for (let i = 0; i < N; i++) { const a = Math.floor(r() * K), rw = mu[a] + q.sigma * r.normal(), g = grad(a).map((v) => (rw - q.b) * v); g.forEach((v, j) => (sum[j] += v)); sq += g.reduce((s, v) => s + v * v, 0); }
    const mcMean = sum.map((v) => v / N), mcVar = sq / N - mcMean.reduce((s, v) => s + v * v, 0);
    const bs = range(81).map((i) => -10 + i * 0.25), V = bs.map(tv), vb = tv(q.b), v0 = tv(0), vs = tv(bstar);
    const P = [`Gerçek gradyan E[ĝ] = (${Eg.map((v) => fmt(v, 3)).join(", ")}); b = ${q.b} ile ${N} örnekte ölçülen ortalama (${mcMean.map((v) => fmt(v, 3)).join(", ")}) — taban çizgisi ne olursa olsun beklenti aynı.`];
    P.push(`Toplam varyans: seçili b = ${q.b} için ${fmt(vb, 3)} (Monte Carlo ${fmt(mcVar, 3)}), b = 0 için ${fmt(v0, 3)}, en iyi b* = ${fmt(bstar, 3)} için ${fmt(vs, 3)}. b*, ödüllerin ortalamasına (${fmt(mu.reduce((s, v) => s + v / K, 0), 3)}) çok yakın.`);
    if (Math.abs(q.m) >= 3 && Math.abs(q.b) < 1) P.push(`Ödüller ortak bir düzeyde (m = ${q.m}) toplanmışken b = 0, her örneği bu düzeyle ölçekler: gradyanın yönü değişmez ama varyans ${fmt(v0 / vs, 3)} kat büyür.`);
    if (q.D === 0) P.push("Δ = 0: bütün eylemler eşit, gerçek gradyan sıfır; kestirici yalnız gürültü üretir.");
    return {
      plots: [{ title: "Gradyan kestiricisinin toplam varyansı ve taban çizgisi", xlabel: "taban çizgisi b", ylabel: "toplam varyans", wide: true, size: { w: 620, h: 230 }, series: [{ name: "kuramsal varyans", x: bs, y: V, c: 0 }, { name: "seçili b (Monte Carlo)", x: [q.b], y: [mcVar], kind: "scatter", c: 2 }], marks: [{ x: bstar, label: "b*" }] }],
      stats: [{ label: "Varyans: b / 0 / b*", value: `${fmt(vb, 3)} / ${fmt(v0, 3)} / ${fmt(vs, 3)}` }, { label: "b*", value: fmt(bstar, 3) }, { label: "‖E[ĝ]‖", value: fmt(Math.sqrt(Eg.reduce((s, v) => s + v * v, 0)), 3) }],
      note: P.join(" "),
    };
  },
};
