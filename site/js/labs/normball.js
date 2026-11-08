// M0.1 — Lp normlarının birim "çemberleri" ve bir noktanın farklı normları
import { fmt, linspace } from "./plot.js";
const lp = (v, p) => (p === Infinity ? Math.max(...v.map(Math.abs)) : v.reduce((s, x) => s + Math.abs(x) ** p, 0) ** (1 / p));
export default {
  params: [
    { id: "p", label: "p (Lp normu)", min: 0.3, max: 20, log: true, value: 2, help: "p = 1: Manhattan, p = 2: Öklid, büyük p: L∞'a (en büyük bileşen) yaklaşır. p < 1 norm değildir." },
    { id: "vx", label: "Nokta v: x", min: -2, max: 2, step: 0.05, value: 1 },
    { id: "vy", label: "Nokta v: y", min: -2, max: 2, step: 0.05, value: 0.6 },
  ],
  validate: (p) => (p.p < 1 ? [{ level: "warn", msg: `p = ${fmt(p.p)} < 1: "birim top" dışbükey değil (yıldız biçimli) ve üçgen eşitsizliği bozulur — bu bir norm değildir. Seyreklik cezalarında (L0'a yaklaşım) yine de kullanılır.` }] : []),
  run(p) {
    const t = linspace(0, 2 * Math.PI, 361);
    const ball = (q) => t.map((a) => { const v = [Math.cos(a), Math.sin(a)], n = lp(v, q); return [v[0] / n, v[1] / n]; });
    const B = ball(p.p), B1 = ball(1), B2 = ball(2), Bi = ball(Infinity);
    const v = [p.vx, p.vy];
    const tri = lp([1, 0], p.p) + lp([0, 1], p.p) >= lp([1, 1], p.p) - 1e-12;
    return {
      plots: [{ title: "Birim top {x : ‖x‖_p = 1} ve v noktası", xlabel: "x₁", ylabel: "x₂", xlim: [-2.1, 2.1], ylim: [-2.1, 2.1], series: [
        { name: "L1", x: B1.map((q) => q[0]), y: B1.map((q) => q[1]), c: 4, dash: true },
        { name: "L2", x: B2.map((q) => q[0]), y: B2.map((q) => q[1]), c: 1, dash: true },
        { name: "L∞", x: Bi.map((q) => q[0]), y: Bi.map((q) => q[1]), c: 3, dash: true },
        { name: `L${fmt(p.p, 3)}`, x: B.map((q) => q[0]), y: B.map((q) => q[1]), c: 0 },
        { name: "v", x: [v[0]], y: [v[1]], kind: "scatter", r: 5, c: 5 }] }],
      stats: [
        { label: "‖v‖₁ = |v₁| + |v₂|", value: lp(v, 1) }, { label: "‖v‖₂ = √(v₁² + v₂²)", value: lp(v, 2) },
        { label: "‖v‖∞ = max |vᵢ|", value: lp(v, Infinity) }, { label: `‖v‖_p (p = ${fmt(p.p, 3)})`, value: lp(v, p.p) },
        { label: "Üçgen eşitsizliği e₁, e₂ için sağlanıyor mu?", value: tri ? "evet" : "hayır" },
      ],
      note: `Her zaman ‖v‖∞ ≤ ‖v‖₂ ≤ ‖v‖₁ (burada ${fmt(lp(v, Infinity), 3)} ≤ ${fmt(lp(v, 2), 3)} ≤ ${fmt(lp(v, 1), 3)}): p büyüdükçe norm küçülür, birim top büyür ve kareye yaklaşır. L1 topunun köşeleri eksenlerde — L1 cezasının seyrek çözüm vermesinin geometrik nedeni (M1.3).`,
    };
  },
};
