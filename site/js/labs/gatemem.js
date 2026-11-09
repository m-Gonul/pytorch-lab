// M4.4 — Kapılar elle: skaler bir LSTM hücresinde f, i, o kapılarını zaman içinde açıp kapatarak "yaz → tut → oku" belleği kurmak.
// c_t = f_t·c_{t−1} + i_t·g_t,  h_t = o_t·tanh(c_t). Bağlı (CIFG/GRU benzeri) seçenekte i_t = 1 − f_t.
import { fmt, range } from "./plot.js";

export function simulate(q) {
  const T = 60, w0 = 5, w1 = w0 + Math.round(q.wlen), rd = Math.min(T - 3, w1 + Math.round(q.hold));
  const c = [0], h = [0], F = [], I = [], O = [], X = [];
  for (let t = 1; t <= T; t++) {
    const writing = t >= w0 && t < w1, reading = t >= rd && t < rd + 3;
    const x = writing ? q.x : q.noise * Math.sin(1.7 * t) * Math.cos(0.9 * t * t);
    const f = writing ? (q.coupled ? 1 - q.iw : q.fw) : q.fh, i = q.coupled ? 1 - f : writing ? q.iw : q.ih, o = reading ? 1 : q.oh;
    const g = Math.tanh(2 * x), cn = f * c[t - 1] + i * g;
    c.push(cn); h.push(o * Math.tanh(cn)); F.push(f); I.push(i); O.push(o); X.push(x);
  }
  return { T, c, h, F, I, O, X, w0, w1, rd };
}

export default {
  params: [
    { id: "x", label: "Yazılacak değer x (yazma penceresinde)", min: -1, max: 1, step: 0.05, value: 0.6 },
    { id: "wlen", label: "Yazma penceresi (adım)", min: 1, max: 10, value: 3, int: true },
    { id: "hold", label: "Tutma süresi (yazma bitiminden okumaya, adım)", min: 1, max: 45, value: 30, int: true },
    { id: "fh", label: "Tutarken unutma kapısı f", min: 0.5, max: 1, step: 0.005, value: 0.98 },
    { id: "ih", label: "Tutarken giriş kapısı i", min: 0, max: 1, step: 0.01, value: 0.05, on: { coupled: [0] } },
    { id: "oh", label: "Okuma dışında çıkış kapısı o", min: 0, max: 1, step: 0.01, value: 0.1 },
    { id: "fw", label: "Yazarken unutma kapısı f", min: 0, max: 1, step: 0.01, value: 0.1, on: { coupled: [0] } },
    { id: "iw", label: "Yazarken giriş kapısı i", min: 0, max: 1, step: 0.01, value: 0.9 },
    { id: "noise", label: "Tutma sırasındaki girdi gürültüsü (genlik)", min: 0, max: 1, step: 0.05, value: 0.5 },
    { id: "coupled", label: "Bağlı kapılar (i = 1 − f, GRU/CIFG gibi)", options: ["hayır (LSTM)", "evet"], value: 0 },
  ],
  run(q) {
    const S = simulate(q), ts = range(S.T + 1), target = Math.tanh(2 * q.x), readC = S.c[S.rd], half = q.fh < 1 ? Math.log(0.5) / Math.log(q.fh) : Infinity;
    const iHold = q.coupled ? 1 - q.fh : q.ih;
    return {
      plots: [
        { title: "Hücre durumu c_t ve çıktı h_t", xlabel: "zaman adımı t", ylabel: "değer", ylim: [-1.1, 1.1], series: [{ name: "c_t (uzun süreli)", x: ts, y: S.c, c: 0 }, { name: "h_t (kısa süreli / çıktı)", x: ts, y: S.h, c: 2, dots: true }, { name: "girdi x_t", x: ts.slice(1), y: S.X, c: 4, dash: true }], marks: [{ x: S.w0, label: "yaz" }, { x: S.rd, label: "oku" }] },
        { title: "Kapı değerleri", xlabel: "zaman adımı t", ylabel: "kapı", ylim: [-0.05, 1.05], series: [{ name: "f (unut)", x: ts.slice(1), y: S.F, c: 0 }, { name: "i (yaz)", x: ts.slice(1), y: S.I, c: 1 }, { name: "o (oku)", x: ts.slice(1), y: S.O, c: 2 }] },
      ],
      stats: [
        { label: "Yazılmak istenen tanh(2x)", value: target },
        { label: "Okuma anında c_t", value: readC },
        { label: "Okumada h_t = tanh(c_t)", value: S.h[S.rd] },
        { label: "Belleğin yarılanma süresi ln ½ / ln f", value: Number.isFinite(half) ? `${fmt(half, 3)} adım` : "∞ (f = 1)" },
      ],
      note: `Yazma penceresinde i ≈ 1, f ≈ 0: eski içerik silinir, yeni değer yazılır. Tutarken f ≈ 1 ve i ≈ 0: c_t neredeyse hiç değişmez (her adımda yalnız f ile çarpılır — gradyan da geriye bu çarpanla akar). ${iHold > 0.02 ? `Tutarken i = ${fmt(iHold, 2)}: gürültü belleğe sızar. ` : ""}${q.coupled ? "Bağlı kapılarda tutma (f ≈ 1) otomatik olarak yazmayı kapatır (i = 1 − f ≈ 0): 'yazmadan önce sil' ilkesi — GRU'nun güncelleme kapısı z tam olarak budur. " : ""}Çıkış kapısı okuma dışında kapalıyken (o ≈ 0) bellek çıktıyı etkilemez; okuma anında o = 1 ve h_t = tanh(c_t). Kapılar burada elle ayarlanıyor; gerçek LSTM'de her biri girdiden ve h_{t−1}'den öğrenilen bir sigmoid katmanıdır.`,
    };
  },
};
