// M4.3 — Tek rekürent nöron, zamanda açılmış: h_t = φ(w_x·x_t + w_h·h_{t−1} + b). Dürtü/basamak/sinüs girdisine yanıt;
// doğrusal durumda dürtü yanıtı w_x·w_hᵗ (IIR süzgeç), sabit pencereli tam bağlı model ise FIR süzgeçtir.
import { fmt, range } from "./plot.js";

export const ACTS = [(z) => Math.tanh(z), (z) => Math.max(0, z), (z) => z];
export function simulate(q, x) {
  const h = []; let prev = 0;
  for (let t = 0; t < x.length; t++) { const v = ACTS[q.act](q.wx * x[t] + q.wh * prev + q.b); h.push(v); prev = v; }
  return h;
}
export function input(kind, T, r) {
  return range(T).map((t) => (kind === 0 ? (t === 5 ? 1 : 0) : kind === 1 ? (t >= 5 ? 1 : 0) : kind === 2 ? Math.sin((2 * Math.PI * t) / 12) : r.normal()));
}

export default {
  params: [
    { id: "inp", label: "Girdi dizisi x_t", options: ["dürtü (t = 5'te 1)", "basamak (t ≥ 5'te 1)", "sinüs (periyot 12)", "beyaz gürültü"], value: 0 },
    { id: "act", label: "Aktivasyon φ", options: ["tanh", "ReLU", "doğrusal (yok)"], value: 0 },
    { id: "wh", label: "Rekürent ağırlık w_h", min: -1.5, max: 1.5, step: 0.01, value: 0.8 },
    { id: "wx", label: "Girdi ağırlığı w_x", min: -3, max: 3, step: 0.05, value: 1 },
    { id: "b", label: "Bias b", min: -1, max: 1, step: 0.05, value: 0 },
    { id: "T", label: "Dizi uzunluğu T", min: 20, max: 200, value: 60, int: true },
  ],
  seed: true,
  seedOn: { inp: 3 },
  run(q, r) {
    const T = Math.round(q.T), x = input(q.inp, T, r), h = simulate(q, x), ts = range(T);
    const peak = Math.max(...h.map(Math.abs)), after = h.slice(6), idx = after.findIndex((v, i) => after.slice(i).every((w) => Math.abs(w) < 0.01 * Math.max(peak, 1e-12)));
    const mem = q.inp === 0 && peak > 0 ? (idx >= 0 ? idx + 1 : `> ${T - 6}`) : "—";
    const lin = q.act === 2, exploded = !h.every(Number.isFinite) || peak > 1e6;
    return {
      plots: [
        { title: "Girdi x_t ve durum h_t", xlabel: "zaman adımı t", ylabel: "değer", series: [{ name: "x_t", x: ts, y: x, kind: q.inp === 0 ? "bar" : undefined, c: 4, dash: q.inp !== 0 }, { name: "h_t", x: ts, y: h.map((v) => (Number.isFinite(v) ? Math.max(-1e6, Math.min(1e6, v)) : 1e6)), c: 0, dots: T <= 80 }] },
        { title: "|h_t| (log): sönme ya da patlama", xlabel: "zaman adımı t", ylabel: "|h_t|", ylog: true, series: [{ name: "|h_t|", x: ts, y: h.map((v) => Math.max(Math.min(Math.abs(v), 1e12), 1e-12)), c: 0 }, ...(lin && q.inp === 0 ? [{ name: "w_x·|w_h|^(t−5) (doğrusal kuram)", x: ts.slice(5), y: ts.slice(5).map((t) => Math.max(Math.abs(q.wx) * Math.abs(q.wh) ** (t - 5), 1e-12)), c: 2, dash: true }] : [])] },
      ],
      stats: [
        { label: "En büyük |h_t|", value: exploded ? "patladı (> 10⁶)" : peak },
        { label: "Dürtünün izi %1'in altına kaç adımda iner?", value: mem },
        { label: "Parametre sayısı (w_x, w_h, b)", value: 3 },
      ],
      note: `Aynı üç parametre her adımda yeniden kullanılır; h_t, x_0 … x_t'nin <b>tümüne</b> bağlıdır (bellek). ${lin ? `Aktivasyon yokken h_t = Σ w_x·w_hᵏ·x_{t−k} + …: dürtü yanıtı geometrik dizidir — sonsuz dürtü yanıtlı (IIR) bir süzgeç. |w_h| ${Math.abs(q.wh) < 1 ? `< 1: kararlı, iz ≈ ln(0.01)/ln|w_h| ≈ ${fmt(Math.log(0.01) / Math.log(Math.abs(q.wh) || 1e-9), 3)} adımda %1'e iner.` : Math.abs(q.wh) === 1 ? "= 1: sınırda, iz sönmez." : "> 1: kararsız, durum üstel büyür."}` : q.act === 1 ? `ReLU doymaz: w_h > 1 iken pozitif durum her adımda büyür${q.wh > 1 ? " — burada patlıyor" : ""}. RNN'lerde ReLU yerine tanh'ın varsayılan olmasının nedeni bu.` : `tanh durumu (−1, 1) aralığında tutar: w_h büyük olsa bile patlama olmaz, ama doyunca türev ≈ 0 olur (gradyan söner, M4.3 deneyi l1).`} Tam bağlı bir model yalnız sabit bir pencereyi (son K girdi) görür: sonlu dürtü yanıtlı (FIR) süzgeç.${q.wh < 0 ? " w_h < 0: durumun işareti her adımda değişir (salınım)." : ""}`,
    };
  },
};
