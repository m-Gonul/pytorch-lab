// M4.4 — Uzun mesafeli gradyan: rastgele başlatılmış düz RNN, LSTM ve GRU'da ‖∂h_T/∂x_{T−k}‖, gecikme k'ya göre.
// LSTM'in hücre yolu c_t = f⊙c_{t−1} + i⊙g: doğrudan çarpan yalnız f (unutma kapısı bias'ı b_f ile ayarlanır).
import { fmt, range } from "./plot.js";

const sig = (z) => 1 / (1 + Math.exp(-z));
const mat = (r, a, b, s) => range(a).map(() => range(b).map(() => s * r.normal()));
const mv = (M, v) => M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
const mtv = (M, g) => M[0].map((_, j) => M.reduce((s, row, i) => s + row[j] * g[i], 0));
const add = (...vs) => vs[0].map((_, i) => vs.reduce((s, v) => s + v[i], 0));

// her model: ileri geçişte ara değerleri sakla, sonra son durumdan rastgele bir yönde geriye yay ve her adımda ‖∂/∂x_t‖'yi ölç
export function lagGrad(kind, q, r, probe = null) {
  const n = Math.round(q.n), d = 4, T = Math.round(q.T), sx = 1 / Math.sqrt(d), sh = q.ws / Math.sqrt(n);
  const X = range(T).map(() => range(d).map(() => q.xs * r.normal()));
  const G = kind === 0 ? 1 : kind === 1 ? 4 : 3, Wx = range(G).map(() => mat(r, n, d, sx)), Wh = range(G).map(() => mat(r, n, n, sh)), dir = range(n).map(() => r.normal() / Math.sqrt(n));
  if (probe) X[probe.t][probe.j] += probe.eps;
  const S = []; let h = new Array(n).fill(0), c = new Array(n).fill(0);
  for (let t = 0; t < T; t++) {
    const x = X[t], pre = (k, bias = 0) => add(mv(Wx[k], x), mv(Wh[k], h)).map((v) => v + bias);
    if (kind === 0) { const z = pre(0), hn = z.map(Math.tanh); S.push({ h, z, hn }); h = hn; }
    else if (kind === 1) { const i = pre(0).map(sig), f = pre(1, q.bf).map(sig), g = pre(2).map(Math.tanh), o = pre(3).map(sig), cn = c.map((cv, k) => f[k] * cv + i[k] * g[k]), tc = cn.map(Math.tanh), hn = o.map((ov, k) => ov * tc[k]); S.push({ h, c, i, f, g, o, cn, tc }); h = hn; c = cn; }
    else { const z = pre(0, q.bf).map(sig), rr = pre(1).map(sig), rh = rr.map((v, k) => v * h[k]), nn = add(mv(Wx[2], x), mv(Wh[2], rh)).map(Math.tanh), hn = z.map((zv, k) => zv * h[k] + (1 - zv) * nn[k]); S.push({ h, z, rr, rh, nn }); h = hn; }
  }
  if (probe) return h.reduce((a, v, k) => a + v * dir[k], 0);
  let dh = [...dir], dc = new Array(n).fill(0); const gx = new Array(T);
  for (let t = T - 1; t >= 0; t--) {
    const s = S[t];
    if (kind === 0) { const dz = dh.map((v, k) => v * (1 - s.hn[k] ** 2)); gx[t] = mtv(Wx[0], dz); dh = mtv(Wh[0], dz); }
    else if (kind === 1) {
      const dtc = dh.map((v, k) => v * s.o[k]), dcn = dc.map((v, k) => v + dtc[k] * (1 - s.tc[k] ** 2));
      const dpo = dh.map((v, k) => v * s.tc[k] * s.o[k] * (1 - s.o[k])), dpi = dcn.map((v, k) => v * s.g[k] * s.i[k] * (1 - s.i[k])), dpf = dcn.map((v, k) => v * s.c[k] * s.f[k] * (1 - s.f[k])), dpg = dcn.map((v, k) => v * s.i[k] * (1 - s.g[k] ** 2));
      const P = [dpi, dpf, dpg, dpo]; gx[t] = add(...P.map((p, k) => mtv(Wx[k], p))); dh = add(...P.map((p, k) => mtv(Wh[k], p))); dc = dcn.map((v, k) => v * s.f[k]);
    } else {
      const dz = dh.map((v, k) => v * (s.h[k] - s.nn[k]) * s.z[k] * (1 - s.z[k])), dnn = dh.map((v, k) => v * (1 - s.z[k]) * (1 - s.nn[k] ** 2));
      const drh = mtv(Wh[2], dnn), dr = drh.map((v, k) => v * s.h[k] * s.rr[k] * (1 - s.rr[k]));
      gx[t] = add(mtv(Wx[0], dz), mtv(Wx[1], dr), mtv(Wx[2], dnn));
      dh = add(dh.map((v, k) => v * s.z[k]), mtv(Wh[0], dz), mtv(Wh[1], dr), drh.map((v, k) => v * s.rr[k]));
    }
  }
  return probe === null && q.raw ? gx : gx.map((g) => Math.sqrt(g.reduce((a, b) => a + b * b, 0)));
}

export default {
  params: [
    { id: "T", label: "Dizi uzunluğu T", min: 20, max: 200, value: 100, int: true },
    { id: "n", label: "Gizli boyut", min: 8, max: 48, value: 24, int: true },
    { id: "bf", label: "Unutma kapısı bias'ı b_f (LSTM) / güncelleme kapısı bias'ı b_z (GRU)", min: -3, max: 5, step: 0.1, value: 1 },
    { id: "ws", label: "Rekürent ağırlık ölçeği", min: 0.3, max: 2, step: 0.05, value: 1 },
    { id: "xs", label: "Girdi ölçeği", min: 0.1, max: 3, step: 0.1, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const T = Math.round(q.T), names = ["düz RNN (tanh)", "LSTM", "GRU"], G = [0, 1, 2].map((k) => lagGrad(k, q, r)), lags = range(T).map((t) => T - 1 - t);
    const at = (g, k) => g[T - 1 - Math.min(k, T - 1)] / Math.max(g[T - 1], 1e-300), k50 = Math.min(50, T - 1);
    const fMean = sig(q.bf);
    return {
      plots: [{ title: "‖∂h_T/∂x_{T−k}‖ / ‖∂h_T/∂x_T‖, gecikmeye göre (log)", xlabel: "gecikme k (adım)", ylabel: "göreli gradyan", ylog: true,
        series: G.map((g, k) => ({ name: names[k], x: lags, y: g.map((v) => Math.max(v / Math.max(g[T - 1], 1e-300), 1e-30)), c: [2, 0, 3][k] })), marks: [{ x: k50, label: `k = ${k50}` }] }],
      stats: [
        { label: `k = ${k50} adım öncesine ulaşan gradyan: RNN`, value: at(G[0], k50) },
        { label: "… LSTM", value: at(G[1], k50) },
        { label: "… GRU", value: at(G[2], k50) },
        { label: "σ(b_f): girdisiz durumda unutma kapısının değeri", value: fMean },
      ],
      note: `Düz RNN'de gradyan her adımda W_hᵀ·diag(1 − h²) ile çarpılır: tipik olarak üstel söner. LSTM'in hücre yolunda doğrudan çarpan yalnız unutma kapısıdır: ∂c_t/∂c_{t−1} = diag(f_t) (+ kapıların h üzerinden dolaylı yolları). b_f = ${fmt(q.bf, 2)} ile f ≈ σ(b_f) = ${fmt(fMean, 3)}: hücre yolu ${fMean > 0.9 ? "uzun süre korunur" : "hızla unutur"}${fMean < 0.99 ? ` (k adımda yaklaşık ${fmt(fMean, 3)}ᵏ)` : ""}. b_f'yi 0'dan 3'e çıkar: LSTM'in uzun mesafe gradyanı belirgin biçimde artar — Keras'ın unutma bias'ını 1'le başlatmasının (ve "chrono" başlatmanın, x2) nedeni. GRU'da h_t = z⊙h_{t−1} + (1 − z)⊙n: büyük b_z, z ≈ 1 yapıp durumu kopyalar.`,
    };
  },
};
