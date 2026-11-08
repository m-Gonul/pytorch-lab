// M0.3 — Gradyan: en dik çıkış yönü, eş-yükselti eğrisine dik; yönlü türev D_u f = ∇f · u = ‖∇f‖ cos φ.
import { fmt, linspace } from "./plot.js";

export const FUNS = [
  { name: "eliptik çanak ½x² + 2y²", f: (x, y) => 0.5 * x * x + 2 * y * y, g: (x, y) => [x, 4 * y] },
  { name: "eyer x² − y²", f: (x, y) => x * x - y * y, g: (x, y) => [2 * x, -2 * y] },
  { name: "muz vadisi (1−x)² + 5(y−x²)²", f: (x, y) => (1 - x) ** 2 + 5 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)] },
  { name: "dalgalı sin x + cos y", f: (x, y) => Math.sin(x) + Math.cos(y), g: (x, y) => [Math.cos(x), -Math.sin(y)] },
];
const R = 2.5;

export default {
  params: [
    { id: "fn", label: "Fonksiyon f(x, y)", options: FUNS.map((F) => F.name), value: 0 },
    { id: "px", label: "Nokta x", min: -2, max: 2, step: 0.05, value: 1.2 },
    { id: "py", label: "Nokta y", min: -2, max: 2, step: 0.05, value: 0.6 },
    { id: "th", label: "Yön açısı θ (derece)", min: 0, max: 360, step: 1, value: 20, unit: "°", help: "Birim yön vektörü u = (cos θ, sin θ). Yönlü türev, bu yönde birim adım başına f'nin değişimidir." },
  ],
  validate(p) {
    const g = FUNS[p.fn].g(p.px, p.py);
    return Math.hypot(...g) < 1e-9 ? [{ level: "warn", msg: "Bu noktada ∇f = 0 (durağan nokta): her yönde yönlü türev 0, 'en dik yön' tanımsız." }] : [];
  },
  run(p) {
    const F = FUNS[p.fn], x = p.px, y = p.py, fp = F.f(x, y), g = F.g(x, y), gn = Math.hypot(...g);
    const t = (p.th * Math.PI) / 180, u = [Math.cos(t), Math.sin(t)];
    const Du = g[0] * u[0] + g[1] * u[1];
    const hh = 1e-5, Dnum = (F.f(x + hh * u[0], y + hh * u[1]) - F.f(x - hh * u[0], y - hh * u[1])) / (2 * hh);
    // ısı haritası
    const nx = 60, ny = 60, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) z.push(F.f(-R + ((i + 0.5) * 2 * R) / nx, -R + ((j + 0.5) * 2 * R) / ny));
    // p'den geçen eş-yükselti eğrisi: ince ızgarada işaret değişimi
    const N = 220, cx = [], cy = [], xs = linspace(-R, R, N);
    const V = xs.map((yy) => xs.map((xx) => F.f(xx, yy) - fp));
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
      const a = V[j][i], b = V[j][i + 1], c = V[j + 1][i];
      if (a * b <= 0 && a !== b) { cx.push(xs[i] + ((xs[i + 1] - xs[i]) * a) / (a - b)); cy.push(xs[j]); }
      else if (a * c <= 0 && a !== c) { cx.push(xs[i]); cy.push(xs[j] + ((xs[j + 1] - xs[j]) * a) / (a - c)); }
    }
    const L = 0.9, arrow = (d, len) => {
      const n = Math.hypot(...d) || 1, e = [d[0] / n, d[1] / n], tip = [x + len * e[0], y + len * e[1]], w = 0.16;
      const l = [tip[0] - w * (e[0] * 0.9 - e[1] * 0.5), tip[1] - w * (e[1] * 0.9 + e[0] * 0.5)], r = [tip[0] - w * (e[0] * 0.9 + e[1] * 0.5), tip[1] - w * (e[1] * 0.9 - e[0] * 0.5)];
      return { x: [x, tip[0], NaN, l[0], tip[0], r[0]], y: [y, tip[1], NaN, l[1], tip[1], r[1]] };
    };
    const ga = gn > 1e-9 ? arrow(g, L) : { x: [x], y: [y] }, ua = arrow(u, L * 0.8);
    const ths = linspace(0, 360, 181), dd = ths.map((a) => g[0] * Math.cos((a * Math.PI) / 180) + g[1] * Math.sin((a * Math.PI) / 180));
    const gAng = ((Math.atan2(g[1], g[0]) * 180) / Math.PI + 360) % 360;
    const phi = gn > 1e-9 ? (Math.acos(Math.max(-1, Math.min(1, Du / gn))) * 180) / Math.PI : NaN;
    return {
      plots: [
        { title: `${FUNS[p.fn].name} — koyu = küçük f`, xlabel: "x", ylabel: "y", size: { w: 380, h: 370 }, xlim: [-R, R], ylim: [-R, R],
          field: { x0: -R, x1: R, y0: -R, y1: R, nx, ny, z, log: p.fn !== 3 },
          series: [
            { name: "p'den geçen eş-yükselti eğrisi f = f(p)", x: cx, y: cy, kind: "scatter", r: 0.9, c: 4 },
            { name: "∇f yönü (en dik çıkış)", x: ga.x, y: ga.y, c: 5 },
            { name: "u = (cos θ, sin θ)", x: ua.x, y: ua.y, c: 0 },
            { name: "p", x: [x], y: [y], kind: "scatter", r: 4.5, c: 3 },
          ] },
        { title: "Yönlü türev D_u f = ∇f · u, θ'ya göre", xlabel: "θ (derece)", ylabel: "D_u f", series: [
            { name: "∇f · (cos θ, sin θ)", x: ths, y: dd, c: 0 },
            { name: "seçili θ", x: [p.th], y: [Du], kind: "scatter", r: 4.5, c: 3 },
          ], marks: [{ y: 0, label: "" }, ...(gn > 1e-9 ? [{ x: gAng, label: "∇f yönü" }] : [])] },
      ],
      stats: [
        { label: "f(p)", value: fp },
        { label: "∇f(p) = (∂f/∂x, ∂f/∂y)", value: `(${fmt(g[0])}, ${fmt(g[1])})` },
        { label: "‖∇f‖ = en büyük yönlü türev", value: gn },
        { label: "D_u f = ∇f · u (formül)", value: Du },
        { label: "D_u f sayısal (u boyunca merkezi fark)", value: Dnum },
        { label: "u ile ∇f arasındaki açı φ", value: Number.isFinite(phi) ? `${fmt(phi, 3)}°` : "tanımsız" },
      ],
      note: `Yönlü türev ‖∇f‖·cos φ'dir: u, ∇f ile aynı yöndeyken (φ = 0) en büyük (= ‖∇f‖), zıt yöndeyken en küçük (gradyan inişinin yönü), dikken (φ = 90°) <b>sıfırdır</b> — o yön eş-yükselti eğrisine teğettir. Bu yüzden ∇f her zaman eş-yükselti eğrisine diktir. Sağ grafik bu yüzden saf bir kosinüs: tepe noktası ∇f'nin açısında.${p.fn === 2 ? " Muz vadisinde gradyan çoğu yerde vadinin dibine değil, dik duvara doğru bakar: en dik iniş yönü minimuma giden yön değildir (M1.2)." : ""}`,
    };
  },
};
