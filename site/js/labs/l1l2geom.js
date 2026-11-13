// M1.3 — Neden L1 seyreklik üretir? Kısıtlı biçim: min f(w) s.t. ‖w‖_p ≤ t. Kayıp elipsleri kısıt bölgesine ilk nerede değer?
import { fmt, linspace } from "./plot.js";

// L1 topuna izdüşüm (Duchi ve ark. 2008), 2-B
export function projL1(v, t) {
  if (Math.abs(v[0]) + Math.abs(v[1]) <= t) return [...v];
  const u = v.map(Math.abs).sort((a, b) => b - a);
  let cs = 0, th = 0;
  for (let i = 0; i < u.length; i++) { cs += u[i]; const c = (cs - t) / (i + 1); if (u[i] > c) th = c; }
  return v.map((x) => Math.sign(x) * Math.max(Math.abs(x) - th, 0));
}
export const projL2 = (v, t) => { const n = Math.hypot(...v); return n <= t ? [...v] : v.map((x) => (x * t) / n); };
// elastic net: g(w) = α|w₁|+α|w₂|+(1−α)(w₁²+w₂²)/2; penalized formun çözümü ISTA ile,
// kısıtlı biçim için g(w(λ)) = t olacak λ dikine bölme (bisection) ile bulunur.
export const soft2 = (z, t) => { const v = Math.sign(z) * Math.max(Math.abs(z) - t, 0); return v || 0; };
const Lof = (H) => Math.max(H[0][0] + Math.abs(H[0][1]), H[1][1] + Math.abs(H[0][1]));
export function istaEN(H, ws, lam, alpha, iters = 400) {
  const eta = 1 / Lof(H);
  let w = [0, 0];
  for (let k = 0; k < iters; k++) {
    const d = [w[0] - ws[0], w[1] - ws[1]], gr = [H[0][0] * d[0] + H[0][1] * d[1], H[0][1] * d[0] + H[1][1] * d[1]];
    const v = [w[0] - eta * gr[0], w[1] - eta * gr[1]];
    const l1 = eta * lam * alpha, l2 = eta * lam * (1 - alpha);
    const nw = [soft2(v[0], l1) / (1 + l2), soft2(v[1], l1) / (1 + l2)];
    if (Math.hypot(nw[0] - w[0], nw[1] - w[1]) < 1e-13) { w = nw; break; }
    w = nw;
  }
  return w;
}
export const gEN = (w, a) => a * (Math.abs(w[0]) + Math.abs(w[1])) + (1 - a) * 0.5 * (w[0] * w[0] + w[1] * w[1]);
export function solveEN(H, ws, t, alpha) {
  let lo = 0, hi = 20;
  for (let i = 0; i < 6 && gEN(istaEN(H, ws, hi, alpha), alpha) > t; i++) hi *= 4;
  for (let i = 0; i < 50; i++) {
    const mid = 0.5 * (lo + hi);
    if (gEN(istaEN(H, ws, mid, alpha), alpha) > t) lo = mid; else hi = mid;
  }
  return istaEN(H, ws, hi, alpha, 1600);
}
export function solve(H, ws, t, p, alpha) { // izdüşümlü gradyan inişi, f = ½ (w − w*)ᵀ H (w − w*)
  if (p === 2) return solveEN(H, ws, t, alpha ?? 0.5);
  const L = Math.max(H[0][0] + Math.abs(H[0][1]), H[1][1] + Math.abs(H[0][1])), eta = 1 / L;
  let w = [0, 0];
  for (let k = 0; k < 4000; k++) {
    const d = [w[0] - ws[0], w[1] - ws[1]], g = [H[0][0] * d[0] + H[0][1] * d[1], H[0][1] * d[0] + H[1][1] * d[1]];
    const z = [w[0] - eta * g[0], w[1] - eta * g[1]], nw = p === 0 ? projL1(z, t) : projL2(z, t);
    if (Math.hypot(nw[0] - w[0], nw[1] - w[1]) < 1e-12) { w = nw; break; }
    w = nw;
  }
  return w;
}

export default {
  params: [
    { id: "p", label: "Kısıt (ceza)", options: ["L1: |w₁| + |w₂| ≤ t (lasso)", "L2: w₁² + w₂² ≤ t² (ridge)", "Elastic net: α|w|₁ + (1−α)/2·‖w‖₂² ≤ t (karma)"], value: 0 },
    { id: "alpha", label: "α (L1 payı; elastic net modunda)", min: 0, max: 1, step: 0.05, value: 0.5, on: { p: [2] } },
    { id: "wx", label: "Cezasız optimum w₁*", min: -3, max: 3, step: 0.05, value: 2 },
    { id: "wy", label: "Cezasız optimum w₂*", min: -3, max: 3, step: 0.05, value: 1 },
    { id: "t", label: "Bütçe t (büyük t = küçük λ)", min: 0.1, max: 4, step: 0.05, value: 1 },
    { id: "kap", label: "Elips uzunluğu (koşul sayısı)", min: 1, max: 20, value: 4, log: true },
    { id: "ang", label: "Elips açısı (derece)", min: 0, max: 180, step: 1, value: 20 },
  ],
  run(q) {
    const th = (q.ang * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th), l1 = 1, l2 = 1 / q.kap;
    const H = [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]];
    const ws = [q.wx, q.wy], f = (a, b) => { const d = [a - ws[0], b - ws[1]]; return 0.5 * (H[0][0] * d[0] * d[0] + 2 * H[0][1] * d[0] * d[1] + H[1][1] * d[1] * d[1]); };
    const w = solve(H, ws, q.t, q.p, q.alpha), fw = f(...w);
    const R = 3.5, N = 60, z = [];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) z.push(f(-R + ((i + 0.5) * 2 * R) / N, -R + ((j + 0.5) * 2 * R) / N));
    // çözümden geçen eş-yükselti elipsi: f = f(w)
    const E = [], ang = linspace(0, 2 * Math.PI, 161);
    const ev = [[c, s], [-s, c]], rr = [Math.sqrt((2 * fw) / l1), Math.sqrt((2 * fw) / l2)];
    for (const a of ang) { const u = rr[0] * Math.cos(a), v = rr[1] * Math.sin(a); E.push([ws[0] + u * ev[0][0] + v * ev[1][0], ws[1] + u * ev[0][1] + v * ev[1][1]]); }
    // kısıt bölgesi: L1 elmas, L2 çember, elastic net ikisinin arasında yuvarlak köşeli şekil
    const rEN = (a) => { const s1 = Math.abs(Math.cos(a)) + Math.abs(Math.sin(a)); return q.alpha > 0.999 ? q.t / s1 : (-q.alpha * s1 + Math.sqrt(q.alpha * q.alpha * s1 * s1 + 2 * (1 - q.alpha) * q.t)) / (1 - q.alpha); };
    const ballName = q.p === 0 ? "L1 topu (elmas)" : q.p === 1 ? "L2 topu (çember)" : `Elastic net bölgesi (α = ${fmt(q.alpha, 2)})`;
    const ball = q.p === 0 ? [[q.t, 0], [0, q.t], [-q.t, 0], [0, -q.t], [q.t, 0]] : ang.map((a) => [(q.p === 2 ? rEN(a) : q.t) * Math.cos(a), (q.p === 2 ? rEN(a) : q.t) * Math.sin(a)]);
    const inside = q.p === 0 ? Math.abs(ws[0]) + Math.abs(ws[1]) <= q.t : q.p === 1 ? Math.hypot(...ws) <= q.t : gEN(ws, q.alpha) <= q.t;
    const zero = w.map((v) => Math.abs(v) < 1e-6);
    return {
      plots: [{ title: "Kayıp (koyu = küçük), kısıt bölgesi ve çözüm", xlabel: "w₁", ylabel: "w₂", size: { w: 400, h: 390 }, xlim: [-R, R], ylim: [-R, R],
        field: { x0: -R, x1: R, y0: -R, y1: R, nx: N, ny: N, z, log: true },
        series: [
          { name: ballName, x: ball.map((b) => b[0]), y: ball.map((b) => b[1]), c: 0 },
          { name: "çözümden geçen kayıp eğrisi", x: E.map((e) => e[0]), y: E.map((e) => e[1]), c: 2, dash: true },
          { name: "cezasız optimum w*", x: [ws[0]], y: [ws[1]], kind: "scatter", c: 4, r: 5 },
          { name: "kısıtlı çözüm", x: [w[0]], y: [w[1]], kind: "scatter", c: 1, r: 6 },
        ], marks: [{ x: 0, label: "" }, { y: 0, label: "" }] }],
      stats: [
        { label: "Kısıtlı çözüm (w₁, w₂)", value: `(${fmt(w[0], 4)}, ${fmt(w[1], 4)})` },
        { label: "Tam sıfır olan koordinat", value: zero[0] && zero[1] ? "ikisi de" : zero[0] ? "w₁ = 0" : zero[1] ? "w₂ = 0" : "yok" },
        { label: "w* kısıtın içinde mi?", value: inside ? "evet — ceza etkisiz" : "hayır — çözüm sınırda" },
        { label: "Kayıp artışı f(ŵ) − f(w*)", value: fw },
      ],
      note: `Kısıtlı problem, cezalı biçimin (kayıp + λ·ceza) eşdeğeridir: her λ'ya bir t karşılık gelir (Lagrange çarpanı). Çözüm, w*'tan büyüyen kayıp elipslerinin kısıt bölgesine <b>ilk değdiği</b> noktadır. ${q.p === 0 ? "Elmasın <b>köşeleri</b> eksenlerdedir ve dışarı doğru sivridir: elips çoğu konumda önce bir köşeye değer, orada bir koordinat tam 0'dır. Elmasın kenarına değerse iki koordinat da sıfırdan farklıdır — w*'ın konumunu ve elipsin açısını değiştirerek bunu dene." : q.p === 1 ? "Çemberin köşesi yoktur: değme noktası neredeyse hiçbir zaman tam eksen üzerinde değildir. Ridge katsayıları küçültür ama sıfırlamaz." : `Elastic net bölgesi elmas ile çember arasında: α = ${fmt(q.alpha, 2)}. Kuadratik terim köşeleri yuvarlar ama α > 0 olduğu sürece eksenler üzerindeki "yakalama alanı" korunur — çözüm hâlâ tam sıfır koordinat üretebilir. α'yı 0'a çek: bölge çembere dönüşür (saf ridge), 1'e: elmasa (saf lasso).`}`,
    };
  },
};
