// M11.9 — Deterministik politika gradyanı, tek boyutlu doğrusal–karesel düzenleyicide (LQR). Sistem s' = s + a (belirlenimci),
// ödül r = −(s² + ρa²), indirim γ; politika a = μ_θ(s) = −k s (θ = k), başlangıç s₀ ~ N(0, 1). Sabit k ile V(s) = −P(k) s², P(k) = (1 + ρk²)/(1 − γ(1 − k)²)
// (|1 − k|√γ < 1 iken). Q(s, a) = −(s² + ρa²) + γ V(s + a); DPG: ∇_k J = Σ_s ρ^π(s) ∇_k μ(s) ∂_a Q(s, a)|_{a=μ(s)}, ρ^π indirimli
// ziyaret dağılımı; kapalı biçimde hesaplanır. "Gürültülü eleştirmen" seçeneğinde ∂_a Q'ya her adımda N(0, η²) hata eklenir.
// Riccati çözümü: P* = 1 + γP*(1 − k*)², k* = γP*/(ρ + γP*).
import { fmt, range } from "./plot.js";

const Pk = (k, g, rho) => { const d = 1 - g * (1 - k) ** 2; return d > 0 ? (1 + rho * k * k) / d : Infinity; };
function riccati(g, rho) { let P = 1; for (let i = 0; i < 2000; i++) { const k = (g * P) / (rho + g * P); P = 1 + rho * k * k + g * P * (1 - k) ** 2; } return { P, k: (g * P) / (rho + g * P) }; }
export default {
  params: [
    { id: "rho", label: "Eylem maliyeti ρ", min: 0.1, max: 5, step: 0.1, value: 1 },
    { id: "g", label: "İndirim γ", min: 0.5, max: 0.99, step: 0.01, value: 0.9 },
    { id: "k0", label: "Başlangıç kazancı k₀", min: 0, max: 1.5, step: 0.05, value: 0.1 },
    { id: "alpha", label: "Adım α", min: 0.005, max: 0.2, step: 0.005, value: 0.05 },
    { id: "eta", label: "Eleştirmen hatası η", min: 0, max: 5, step: 0.25, value: 0 },
    { id: "N", label: "Yineleme", min: 10, max: 500, step: 10, value: 150, int: true },
  ],
  seed: true,
  seedOn: { eta: 2 },
  run(q, r) {
    const { g, rho } = q, R = riccati(g, rho), J = (k) => -Pk(k, g, rho);
    // dQ/da at a = −k s:  ∂_a[−(s² + ρa²) − γP(k)(s + a)²] = −2ρa − 2γP(k)(s + a); ∇_k J = E[−s ∂_a Q] = 2ρ(−k)·(−1)… kapalı biçim (E s² = 1):
    // s_t = (1 − k)^t s₀ olduğundan indirimli ziyaret ağırlığı Σ_t γ^t E[s_t²] = 1/(1 − γ(1 − k)²); DPG gradyanı bu ağırlıkla
    // her durumdaki −s·∂_a Q = s²(−2ρk + 2γP(k)(1 − k)) teriminin toplamıdır. Sonlu farkla J'nin türevine eşit olduğu denetlendi.
    const grad = (k) => { const P = Pk(k, g, rho); if (!Number.isFinite(P)) return NaN; return (-2 * rho * k + 2 * g * P * (1 - k)) / (1 - g * (1 - k) ** 2); };
    let k = q.k0; const ks = [k], Js = [J(k)];
    for (let n = 0; n < q.N; n++) { let gk = grad(k); if (!Number.isFinite(gk)) break; gk += q.eta * r.normal(); k += q.alpha * gk; ks.push(k); Js.push(J(k)); }
    const kgrid = range(121).map((i) => i * 0.0165), Jg = kgrid.map((kk) => { const v = J(kk); return Number.isFinite(v) && v > -60 ? v : NaN; });
    const fin = ks[ks.length - 1], diverged = !Number.isFinite(Js[Js.length - 1]);
    const P = [`Riccati çözümü: k* = ${fmt(R.k, 4)}, J(k*) = −P* = ${fmt(-R.P, 4)}. DPG ${ks.length - 1} adımda k = ${fmt(fin, 4)}'e geldi${diverged ? " ama kazanç kararlılık bölgesinden çıktı (J = −∞)" : `; J = ${fmt(J(fin), 4)}`}.`];
    P.push(`Gradyan ∇_k J = Σ_s ρ^π(s) ∇_k μ(s) · ∂_a Q(s, a)|_(a = μ(s)) (ρ^π indirimli ziyaret dağılımı): politika, eylemi eleştirmenin eğimi yönünde iter. Sürekli eylemde max_a gerekmez; Q'nun a'ya göre türevi yeterlidir.`);
    if (diverged) P.push(`Başlangıç k₀ = ${q.k0} için gradyan ${fmt(grad(q.k0), 3)}: sistem neredeyse kontrolsüzken değer çok negatif ve eğim çok diktir. Büyük bir adım kazancı kararlılık sınırının (k < 1 + 1/√γ = ${fmt(1 + 1 / Math.sqrt(g), 3)}) dışına atıyor. Politika uzayında adım boyunu sınırlamak, güven bölgesi yöntemlerinin (M11.12) çıkış noktasıdır.`);
    if (q.eta > 0) P.push(`Eleştirmen hatası η = ${q.eta}: güncellemeler gürültülü; sabit α ile k, k* etrafında ${fmt(Math.sqrt(q.alpha) * q.eta, 2)} mertebesinde dalgalanır.`);
    if (q.rho >= 3) P.push("Eylem pahalıyken en iyi kazanç küçük: sistem yavaş düzeltilir.");
    return {
      plots: [
        { title: "J(k) ve DPG'nin izi", xlabel: "kazanç k", ylabel: "J(k) = −P(k)", side: true, size: { w: 320, h: 240 }, series: [{ name: "J(k)", x: kgrid, y: Jg, c: 0 }, { name: "DPG adımları", x: ks, y: Js.map((v) => (Number.isFinite(v) && v > -60 ? v : NaN)), kind: "scatter", c: 2 }], marks: [{ x: R.k, label: "k*" }] },
        { title: "Kazancın yinelemelerle değişimi", xlabel: "yineleme", ylabel: "k", side: true, size: { w: 280, h: 240 }, series: [{ name: "k_n", x: range(ks.length), y: ks, c: 2 }, { name: "Riccati k*", x: [0, ks.length - 1], y: [R.k, R.k], c: 4, dash: true }] },
      ],
      stats: [{ label: "k* (Riccati)", value: fmt(R.k, 4) }, { label: "DPG'nin son k'sı", value: fmt(fin, 4) }, { label: "J(k*) / J(son)", value: `${fmt(-R.P, 4)} / ${fmt(J(fin), 4)}` }],
      note: P.join(" "),
    };
  },
};
