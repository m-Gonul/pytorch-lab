// M4.3 / M4.4 — Zaman boyunca gradyan normu: düz RNN vs sabit unutma kapılı (LSTM hücre yolu)
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "rho", label: "Rekürent ağırlık ölçeği ρ (spektral yarıçap ≈ ρ)", min: 0.2, max: 3, step: 0.05, value: 1 },
    { id: "act", label: "Aktivasyon", options: ["tanh", "doğrusal"], value: 0 },
    { id: "xs", label: "Girdi ölçeği", min: 0, max: 3, step: 0.1, value: 1, help: "Büyük girdi tanh'ı doyurur: türev (1 − h²) küçülür." },
    { id: "f", label: "LSTM unutma kapısı f (karşılaştırma)", min: 0.5, max: 1, step: 0.005, value: 0.95, help: "Hücre yolu c_t = f·c_{t−1} + …: ∂c_T/∂c_t = f^(T−t)." },
    { id: "T", label: "Dizi uzunluğu T", min: 10, max: 200, step: 1, value: 80, int: true },
    { id: "n", label: "Gizli boyut n", min: 4, max: 64, step: 4, value: 32, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const n = p.n, W = Float64Array.from({ length: n * n }, () => (rnd.normal() * p.rho) / Math.sqrt(n));
    const U = Float64Array.from({ length: n }, () => rnd.normal());
    let h = new Float64Array(n);
    const D = [];
    for (let t = 0; t < p.T; t++) {
      const x = p.xs * rnd.normal(), z = new Float64Array(n);
      for (let j = 0; j < n; j++) { let s = U[j] * x; for (let i = 0; i < n; i++) s += W[j * n + i] * h[i]; z[j] = s; }
      h = p.act === 0 ? z.map(Math.tanh) : z;
      D.push(p.act === 0 ? h.map((v) => 1 - v * v) : new Float64Array(n).fill(1));
    }
    // ∂h_T/∂h_t ile rastgele bir yönün geri yayılımı: g_{t-1} = Wᵀ (d_t ⊙ g_t)
    let g = Float64Array.from({ length: n }, () => rnd.normal());
    const g0 = Math.hypot(...g);
    const norms = [1];
    for (let t = p.T - 1; t >= 1; t--) {
      const dz = g.map((v, j) => v * D[t][j]);
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += W[j * n + i] * dz[j]; out[i] = s; }
      g = out;
      norms.push(Math.hypot(...g) / g0);
    }
    const k = range(norms.length), lstm = k.map((d) => p.f ** d);
    const sat = D.flat ? 0 : 0;
    const meanD = D.reduce((a, d) => a + d.reduce((x, y) => x + y, 0) / n, 0) / D.length;
    const half = norms.findIndex((v) => v < 1e-3);
    return {
      plots: [{ title: "Geriye taşınan gradyanın göreli normu", xlabel: "geriye adım T − t", ylabel: "‖∂h_T/∂h_t · v‖", ylog: true, series: [{ name: "düz RNN", x: k, y: norms.map((v) => Math.max(v, 1e-300)) }, { name: `LSTM hücre yolu f^(T−t), f = ${p.f}`, x: k, y: lstm.map((v) => Math.max(v, 1e-300)), c: 2 }], marks: [{ y: 1e-3, label: "10⁻³" }] }],
      stats: [
        { label: "Ortalama tanh türevi E[1 − h²]", value: meanD },
        { label: "Adım başına etkin çarpan ≈ ρ·E[1−h²]", value: p.rho * meanD },
        { label: "Normun 10⁻³ altına düştüğü adım", value: half < 0 ? "düşmedi" : half },
        { label: "LSTM yolunda 10⁻³ adımı ln(10⁻³)/ln f", value: p.f < 1 ? Math.log(1e-3) / Math.log(p.f) : "∞" },
      ],
      note: norms[norms.length - 1] > 1e3 ? "Patlayan gradyan: çarpan > 1. Gradyan kırpma (norm sınırı) bu yüzden RNN eğitiminde standart." : norms[norms.length - 1] < 1e-6 ? `Kaybolan gradyan: ${p.T} adım önceki girdinin etkisi pratikte sıfır — düz RNN uzun bağımlılık öğrenemez. Unutma kapısı f ≈ 1 olan LSTM hücre yolu toplamsal olduğu için çok daha yavaş söner.` : `Bu ayarda gradyan ${fmt(norms[norms.length - 1])} oranında taşınıyor. ρ ≈ 1 ve küçük girdiyle düz RNN "kaosun kıyısında" çalışır; tanh doydukça çarpan düşer.`,
    };
  },
};
