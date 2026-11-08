// M2.6 — Kuvvet yinelemesi: v ← Cv/‖Cv‖. Hata her adımda |λ₂/λ₁| oranında küçülür.
import { fmt, range } from "./plot.js";

export function iterate(l1, l2, phi, ang0, T) { // C = R diag(l1, l2) Rᵀ, R açısı phi; başlangıç açısı ang0
  const c = Math.cos(phi), s = Math.sin(phi), C = [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]];
  let v = [Math.cos(ang0), Math.sin(ang0)]; const V = [v], err = [];
  const e1 = [c, s];
  for (let t = 0; t <= T; t++) {
    err.push(Math.sqrt(Math.max(0, 1 - (v[0] * e1[0] + v[1] * e1[1]) ** 2))); // |sin(açı farkı)|
    if (t === T) break;
    const w = [C[0][0] * v[0] + C[0][1] * v[1], C[1][0] * v[0] + C[1][1] * v[1]], nn = Math.hypot(...w) || 1;
    v = [w[0] / nn, w[1] / nn]; V.push(v);
  }
  return { V, err, e1 };
}

export default {
  params: [
    { id: "ratio", label: "Özdeğer oranı λ₂/λ₁", min: -0.99, max: 0.99, step: 0.01, value: 0.7, help: "Negatif λ₂: v her adımda ters yöne de salınır. |λ₂/λ₁| → 1: yakınsama çok yavaş." },
    { id: "ang0", label: "Başlangıç vektörünün açısı (derece)", min: 0, max: 180, step: 1, value: 100 },
    { id: "T", label: "Adım sayısı", min: 1, max: 60, value: 20, int: true },
  ],
  run(q) {
    const phi = (25 * Math.PI) / 180, l1 = 1, l2 = q.ratio, R = iterate(l1, l2, phi, (q.ang0 * Math.PI) / 180, Math.round(q.T));
    const th = range(R.err.length).map((t) => R.err[0] * Math.abs(q.ratio) ** t);
    const pts = R.V.slice(0, 12);
    return {
      plots: [
        { title: "İlk 12 yineleme (birim çember üzerinde) ve baş özvektör", xlabel: "v₁", ylabel: "v₂", size: { w: 360, h: 350 }, xlim: [-1.2, 1.2], ylim: [-1.2, 1.2], series: [
          { name: "birim çember", x: range(121).map((i) => Math.cos((i * Math.PI) / 60)), y: range(121).map((i) => Math.sin((i * Math.PI) / 60)), c: 4, dash: true },
          { name: "baş özvektör ±e₁", x: [-R.e1[0], R.e1[0]], y: [-R.e1[1], R.e1[1]], c: 1 },
          { name: "vₜ (t = 0 … 11)", x: pts.map((v) => v[0]), y: pts.map((v) => v[1]), c: 2, dots: true },
        ] },
        { title: "Açı hatası |sin ∠(vₜ, e₁)|, log ölçek", xlabel: "adım t", ylabel: "hata", ylog: true, series: [
          { name: "ölçülen", x: range(R.err.length), y: R.err.map((e) => Math.max(e, 1e-16)), c: 2, dots: true },
          { name: "kuram ~ |λ₂/λ₁|ᵗ", x: range(th.length), y: th.map((e) => Math.max(e, 1e-16)), c: 0, dash: true },
        ] },
      ],
      stats: [
        { label: "|λ₂/λ₁|", value: Math.abs(q.ratio) },
        { label: "Son hata", value: R.err.at(-1) },
        { label: "10⁻⁶ hata için gereken adım ≈ log(10⁻⁶)/log|λ₂/λ₁|", value: Math.abs(q.ratio) > 0 ? Math.ceil(Math.log(1e-6) / Math.log(Math.abs(q.ratio))) : 1 },
      ],
      note: `v₀'ı özvektörler cinsinden yaz: v₀ = a₁e₁ + a₂e₂. t adım sonra Cᵗv₀ = a₁λ₁ᵗe₁ + a₂λ₂ᵗe₂ = λ₁ᵗ(a₁e₁ + a₂(λ₂/λ₁)ᵗe₂): ikinci terim (λ₂/λ₁)ᵗ ile söner. Hız yalnız özdeğer <b>oranına</b> bağlıdır. ${q.ratio < 0 ? "λ₂ < 0: hata bileşeni her adımda işaret değiştirir, v baş özvektörün iki yanında zikzak yapar. " : ""}Başlangıç vektörü e₁'e tam dik olursa (a₁ = 0) yöntem e₁'i hiç bulamaz — kayan nokta yuvarlaması genelde bunu kurtarır. Büyük seyrek matrislerde (PageRank, en büyük birkaç özdeğer) yalnız matris–vektör çarpımı gerektirdiği için tercih edilir; Lanczos gibi yöntemler aynı fikri hızlandırır.`,
    };
  },
};
