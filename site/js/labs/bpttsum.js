// M4.3 — BPTT: paylaşılan ağırlığın gradyanı zaman adımlarının katkılarının toplamıdır. Skaler RNN h_t = tanh(w·h_{t−1} + u·x_t),
// kayıp yalnız son adımda: L = ½(h_T − y)². Her adımın ∂L/∂w katkısı ve kesilmiş (truncated) BPTT'nin k adımlık penceresi.
import { fmt, range } from "./plot.js";

export function bptt(w, u, x, y) {
  const T = x.length, h = [0];
  for (let t = 0; t < T; t++) h.push(Math.tanh(w * h[t] + u * x[t]));
  let dh = h[T] - y; const cw = new Array(T).fill(0), cu = new Array(T).fill(0);
  for (let t = T; t >= 1; t--) { const dz = dh * (1 - h[t] ** 2); cw[t - 1] = dz * h[t - 1]; cu[t - 1] = dz * x[t - 1]; dh = dz * w; }
  return { h, cw, cu, loss: 0.5 * (h[T] - y) ** 2 };
}

export default {
  params: [
    { id: "w", label: "Rekürent ağırlık w", min: -2, max: 2, step: 0.01, value: 0.9 },
    { id: "u", label: "Girdi ağırlığı u", min: -2, max: 2, step: 0.01, value: 0.5 },
    { id: "T", label: "Dizi uzunluğu T", min: 5, max: 120, value: 40, int: true },
    { id: "k", label: "Kesilmiş BPTT penceresi k (son k adım)", min: 1, max: 120, value: 10, int: true },
    { id: "y", label: "Hedef y (son adım)", min: -1, max: 1, step: 0.05, value: 0.8 },
  ],
  seed: true,
  run(q, r) {
    const T = Math.round(q.T), k = Math.min(Math.round(q.k), T), x = range(T).map(() => r.normal()), R = bptt(q.w, q.u, x, q.y);
    const full = R.cw.reduce((a, b) => a + b, 0), trunc = R.cw.slice(T - k).reduce((a, b) => a + b, 0), ts = range(T).map((t) => t + 1);
    // sayısal kontrol: merkezi fark
    const e = 1e-6, num = (bptt(q.w + e, q.u, x, q.y).loss - bptt(q.w - e, q.u, x, q.y).loss) / (2 * e);
    const abs = R.cw.map((v) => Math.max(Math.abs(v), 1e-30));
    return {
      plots: [
        { title: "Her zaman adımının ∂L/∂w'ye katkısı |δ_t · h_{t−1}| (log)", xlabel: "zaman adımı t", ylabel: "|katkı|", ylog: true, series: [{ name: "kesilen (pencere dışı)", x: ts.slice(0, T - k), y: abs.slice(0, T - k), kind: "bar", c: 4 }, { name: `son ${k} adım (kesilmiş BPTT)`, x: ts.slice(T - k), y: abs.slice(T - k), kind: "bar", c: 0 }], ylim: [Math.max(1e-30, Math.min(...abs) / 3), Math.max(...abs) * 3] },
        { title: "Durum h_t", xlabel: "zaman adımı t", ylabel: "h_t", ylim: [-1.05, 1.05], series: [{ name: "h_t", x: [0, ...ts], y: R.h, c: 0, dots: T <= 60 }, { name: "hedef y", x: [T], y: [q.y], kind: "scatter", c: 2, r: 6 }] },
      ],
      stats: [
        { label: "Tam BPTT: ∂L/∂w = Σ_t katkı", value: full },
        { label: "Sayısal kontrol (merkezi fark)", value: num },
        { label: `Kesilmiş (son ${k} adım)`, value: trunc },
        { label: "Kesilmişin göreli hatası", value: Math.abs(full) > 1e-15 ? `%${fmt((100 * Math.abs(trunc - full)) / Math.abs(full), 3)}` : "—" },
      ],
      note: `w her adımda aynıdır; dolayısıyla ∂L/∂w, açılmış ağdaki her kopyanın katkısının <b>toplamıdır</b>: Σ_t δ_t·h_{t−1}, δ_t = ∂L/∂z_t. Geriye giderken δ her adımda w·(1 − h_t²) ile çarpılır: |w|·E[1 − h²] < 1 ise uzak adımların katkısı üstel olarak küçülür ve kesilmiş BPTT neredeyse doğrudur (ama uzun bağımlılıkları <b>hiç</b> öğrenemez); ≈ 1 ise uzak katkılar yakınlar kadar büyüktür, kesmek gradyanı ciddi biçimde saptırır (yanlı kestirim). Dikkat: w büyük olsa bile büyük girdiler tanh'ı doyurur ve (1 − h²) çarpanı küçülür; uzun bellek için durum doymamalı (küçük u, w ≈ 1).`,
    };
  },
};
