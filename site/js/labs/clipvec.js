// M3.7 — Gradyan kırpma: değere göre (her bileşen [−c, c]'ye) vs norma göre (‖g‖ > c ise g·c/‖g‖). Yön ve büyüklük ne olur?
import { fmt, linspace } from "./plot.js";

export const clipValue = (g, c) => g.map((v) => Math.max(-c, Math.min(c, v)));
export const clipNorm = (g, c) => { const n = Math.hypot(...g); return n > c ? g.map((v) => (v * c) / n) : [...g]; };
const ang = (a, b) => { const d = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b) || 1); return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI; };

export default {
  params: [
    { id: "g1", label: "g₁ (birinci bileşen)", min: -200, max: 200, step: 0.1, value: 0.9 },
    { id: "g2", label: "g₂ (ikinci bileşen)", min: -200, max: 200, step: 0.1, value: 100 },
    { id: "c", label: "Eşik c", min: 0.1, max: 100, value: 1, log: true, hardMin: 1e-6, why: "Eşik pozitif olmalı." },
  ],
  validate(q) { return q.g1 === 0 && q.g2 === 0 ? [{ level: "error", msg: "g = 0: kırpılacak bir gradyan yok, yön tanımsız." }] : []; },
  run(q) {
    const g = [q.g1, q.g2], c = q.c, v = clipValue(g, c), n = clipNorm(g, c), G = Math.hypot(...g), lim = 1.7 * c;
    const t = linspace(0, 2 * Math.PI, 121);
    const arrow = (w, name, col) => ({ name, x: [0, w[0]], y: [0, w[1]], c: col, dots: true });
    return {
      plots: [
        {
          title: "Kırpma bölgeleri ve sonuç vektörleri (eksenler c'ye göre ölçekli)", xlabel: "g₁", ylabel: "g₂", size: { w: 420, h: 400 }, xlim: [-lim, lim], ylim: [-lim, lim],
          series: [
            { name: "değere göre: kare [−c, c]²", x: [-c, c, c, -c, -c], y: [-c, -c, c, c, -c], c: 1, dash: true },
            { name: "norma göre: daire ‖g‖ = c", x: t.map((a) => c * Math.cos(a)), y: t.map((a) => c * Math.sin(a)), c: 2, dash: true },
            arrow(g, G > lim ? "orijinal g (pencerenin dışına uzanıyor)" : "orijinal g", 5),
            arrow(v, "clip_by_value(g)", 1),
            arrow(n, "clip_by_norm(g)", 2),
          ],
        },
        {
          title: "Büyüklükler", xlabel: "", ylabel: "‖·‖", ylog: true, xlim: [-0.6, 2.6], ylim: [Math.min(G, Math.hypot(...v), Math.hypot(...n)) / 3, G * 3],
          xticks: [[0, "orijinal"], [1, "değere göre"], [2, "norma göre"]],
          series: [{ name: "norm", x: [0, 1, 2], y: [G, Math.hypot(...v), Math.hypot(...n)], kind: "bar", c: 0 }],
        },
      ],
      stats: [
        { label: "‖g‖", value: G },
        { label: "clip_by_value(g)", value: `(${fmt(v[0], 4)}, ${fmt(v[1], 4)})` },
        { label: "clip_by_norm(g)", value: `(${fmt(n[0], 4)}, ${fmt(n[1], 4)})` },
        { label: "Yön sapması: değere göre", value: `${fmt(ang(g, v), 3)}°` },
        { label: "Yön sapması: norma göre", value: `${fmt(ang(g, n), 3)}°` },
      ],
      note: `${G <= c ? `‖g‖ = ${fmt(G, 3)} ≤ c: norma göre kırpma hiçbir şey yapmaz. ` : `Norma göre kırpma g'yi c/‖g‖ = ${fmt(c / G, 3)} ile ölçekler: yön tam korunur, yalnız boy c'ye iner. `}${Math.abs(q.g1) > c || Math.abs(q.g2) > c ? `Değere göre kırpma, |gᵢ| > c olan bileşenleri ayrı ayrı c'ye keser: büyük bileşenler küçük olanlarla eşitlenir ve yön ${fmt(ang(g, v), 3)}° döner. ` : "Tüm bileşenler [−c, c] içinde: değere göre kırpma da bir şey yapmaz. "}Pratikte (PyTorch <code>clip_grad_norm_</code>) norm <b>tüm parametreler birlikte</b> hesaplanır (global norm): her tensör aynı çarpanla küçülür, güncellemenin yönü bütünüyle korunur.`,
    };
  },
};
