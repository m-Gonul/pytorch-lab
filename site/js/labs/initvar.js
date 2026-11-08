// M3.5 — Derin MLP'de ileri aktivasyon ve geri gradyan ölçeğinin katmanlar boyunca seyri
import { fmt, range } from "./plot.js";
const INITS = ["N(0, 1)", "N(0, 0.01²)", "Xavier/Glorot: Var = 2/(n_in+n_out)", "He/Kaiming: Var = 2/n_in", "LeCun: Var = 1/n_in"];
const ACTS = ["tanh", "ReLU", "SELU", "doğrusal (aktivasyon yok)"];
const SELU_A = 1.6732632423543772, SELU_L = 1.0507009873554805;
const act = (k, z) => k === 0 ? Math.tanh(z) : k === 1 ? Math.max(0, z) : k === 2 ? SELU_L * (z > 0 ? z : SELU_A * (Math.exp(z) - 1)) : z;
const dact = (k, z) => k === 0 ? 1 - Math.tanh(z) ** 2 : k === 1 ? (z > 0 ? 1 : 0) : k === 2 ? SELU_L * (z > 0 ? 1 : SELU_A * Math.exp(z)) : 1;
export default {
  params: [
    { id: "init", label: "Başlatma", options: INITS, value: 3 },
    { id: "act", label: "Aktivasyon", options: ACTS, value: 1 },
    { id: "L", label: "Derinlik (katman)", min: 2, max: 40, step: 1, value: 20, int: true },
    { id: "n", label: "Genişlik n", min: 8, max: 128, step: 8, value: 64, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const n = p.n, B = 48, var_ = [1, 1e-4, 1 / n, 2 / n, 1 / n][p.init];
    const sd = Math.sqrt(var_);
    const Ws = [], Zs = [], Hs = [];
    let h = Array.from({ length: B }, () => Float64Array.from({ length: n }, () => rnd.normal()));
    Hs.push(h);
    for (let l = 0; l < p.L; l++) {
      const W = Float64Array.from({ length: n * n }, () => rnd.normal() * sd);
      Ws.push(W);
      const Z = h.map((row) => { const z = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += row[i] * W[i * n + j]; z[j] = s; } return z; });
      Zs.push(Z);
      h = Z.map((z) => z.map((v) => act(p.act, v)));
      Hs.push(h);
    }
    // geri: L = toplam(h_L · r), r rastgele yön
    let g = Array.from({ length: B }, () => Float64Array.from({ length: n }, () => rnd.normal()));
    const gstd = new Array(p.L + 1);
    const std = (M) => { let s2 = 0, c = 0; for (const r of M) for (const v of r) { s2 += v * v; c++; } return Math.sqrt(s2 / c); }; // RMS: ikinci moment (ReLU çıktısının ortalaması 0 değil)
    gstd[p.L] = std(g);
    for (let l = p.L - 1; l >= 0; l--) {
      const W = Ws[l], Z = Zs[l];
      g = g.map((gr, b) => { const dz = gr.map((v, j) => v * dact(p.act, Z[b][j])); const out = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += W[i * n + j] * dz[j]; out[i] = s; } return out; });
      gstd[l] = std(g);
    }
    const astd = Hs.map(std);
    const dead = p.act === 1 ? Hs[p.L].reduce((a, r) => a + r.filter((v) => v === 0).length, 0) / (B * n) : null;
    const ratio = astd[p.L] / astd[0];
    const fl = (v) => Math.max(v, 1e-300);
    return {
      plots: [{ title: "Katman başına RMS (√E[x²])", xlabel: "katman l", ylabel: "RMS", ylog: true, series: [{ name: "aktivasyon h_l", x: range(p.L + 1), y: astd.map(fl), dots: true }, { name: "gradyan ∂L/∂h_l", x: range(p.L + 1), y: gstd.map(fl), dots: true, c: 2 }], marks: [{ y: 1, label: "1" }] }],
      stats: [
        { label: "RMS(h_L) / RMS(h_0)", value: ratio },
        { label: "RMS(∂/∂h_0) / RMS(∂/∂h_L)", value: gstd[0] / gstd[p.L] },
        { label: "n·Var(w) (ReLU için korunum 2, tanh/doğrusal için 1)", value: n * var_ },
        ...(dead != null ? [{ label: "son katmanda sıfır çıkan ReLU oranı", value: dead }] : []),
      ],
      note: ratio < 1e-3 ? "Sinyal sönüyor: her katman ikinci momenti < 1 çarpanla küçültüyor; derin katmanlar sabite yaklaşır, gradyan da kaybolur." : ratio > 1e3 ? "Sinyal patlıyor: katman başına çarpan > 1; tanh'ta bu doyma (türev ≈ 0) olarak geri döner, gradyan yine kaybolur." : ratio < 0.3 ? `Yavaş sönme (${fmt(ratio)}×): katman başına çarpan 1'in biraz altında — 20 katmanda birikir. Koşul Var(w)·n·E[φ'(z)²] ≈ 1; tanh doğrusal bölgeden çıkınca E[φ'²] < 1 olur.` : ratio > 3 ? `Yavaş büyüme (${fmt(ratio)}×): katman başına çarpan 1'in biraz üstünde.` : `İkinci moment kabaca korunuyor (${fmt(ratio)}×). Koşul: Var(w)·n·E[φ'(z)²] ≈ 1 — ReLU yarısını sıfırladığı için He 2/n, tanh sıfır çevresinde ≈ doğrusal olduğu için Xavier/LeCun 1/n. Tam 1 çıkmaması sonlu genişlik etkisi: her katmanın çarpanı rastgele, log'u ≈ 4/n varyanslı; derinlik/genişlik oranı büyüdükçe dağılım genişler (n'yi büyüt ya da tohumu değiştir).`,
    };
  },
};
