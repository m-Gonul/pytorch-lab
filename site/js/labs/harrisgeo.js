// M9.7 — Harris matrisinin geometrisi. 15 × 15'lik yapay bir yamada (düz, kenar, köşe, tek parlak nokta) [−1, 0, 1]/2 türevleriyle
// (f_x, f_y) saçılımı, M = Σ w·[f_x², f_x f_y; f_x f_y, f_y²] (kutu ya da Gauss ağırlık), özdeğerler ve ağırlıklı std elipsi;
// (λ1, λ2) düzleminde R = det M − k·tr²M = 0 eğrileri ve yamanın noktası.
import { fmt, range } from "./plot.js";

const N = 15, C = 7;
const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const step = (d) => 0.5 * (1 + erf(d / (0.6 * Math.SQRT2)));
export function patch(type, th, sn, r) {
  const co = Math.cos(th), si = Math.sin(th), f = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const u = (x - C) * co + (y - C) * si, v = -(x - C) * si + (y - C) * co;
    let val = 0.3;
    if (type === 1) val = 0.3 + 0.5 * step(u);
    else if (type === 2) val = 0.3 + 0.5 * step(u) * step(v);
    else if (type === 3) val = x === C && y === C ? 0.9 : 0.3;
    f[y * N + x] = val + sn * r.normal();
  }
  return f;
}
export function harris(f, win, sw) {
  const pts = [], w = [];
  let a = 0, b = 0, c = 0, S = 0;
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const fx = (f[y * N + x + 1] - f[y * N + x - 1]) / 2, fy = (f[(y + 1) * N + x] - f[(y - 1) * N + x]) / 2;
    const ww = win === 0 ? (Math.abs(x - C) <= 3 && Math.abs(y - C) <= 3 ? 1 : 0) : Math.exp(-((x - C) ** 2 + (y - C) ** 2) / (2 * sw * sw));
    if (ww < 1e-3) continue;
    pts.push([fx, fy]); w.push(ww); a += ww * fx * fx; b += ww * fx * fy; c += ww * fy * fy; S += ww;
  }
  const tr = a + c, det = a * c - b * b, d = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det)), l1 = tr / 2 + d, l2 = tr / 2 - d;
  const ang = 0.5 * Math.atan2(2 * b, a - c);
  return { a, b, c, tr, det, l1, l2, ang, pts, w, S };
}
const TYPES = ["düz bölge", "kenar", "köşe (90°)", "tek parlak nokta"];
export default {
  params: [
    { id: "type", label: "Yama", options: TYPES, value: 2 },
    { id: "th", label: "Yön θ (°)", min: 0, max: 90, step: 1, value: 0, on: { type: [1, 2] } },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.02 },
    { id: "win", label: "Ağırlık penceresi", options: ["kutu 7 × 7", "Gauss"], value: 1 },
    { id: "sw", label: "Gauss pencere σ", min: 0.8, max: 3, step: 0.1, value: 1.5, on: { win: [1] } },
    { id: "k", label: "k", min: 0.01, max: 0.25, step: 0.005, value: 0.04 },
  ],
  seed: true,
  run(q, r) {
    const f = patch(q.type, (q.th * Math.PI) / 180, q.noise, r), H = harris(f, q.win, q.sw), R = H.det - q.k * H.tr * H.tr;
    const s1 = Math.sqrt(H.l1 / H.S), s2 = Math.sqrt(Math.max(H.l2, 0) / H.S), ca = Math.cos(H.ang), sa = Math.sin(H.ang);
    const el = range(73).map((i) => { const t = (i * 2 * Math.PI) / 72, u = 2 * s1 * Math.cos(t), v = 2 * s2 * Math.sin(t); return [u * ca - v * sa, u * sa + v * ca]; });
    const ratio = H.l2 / Math.max(H.l1, 1e-12), cls = H.tr < 0.05 ? "düz (tr M küçük)" : R > 0 ? "köşe (R > 0)" : "kenar (R < 0)";
    const L = Math.max(H.l1 * 1.3, 0.05), xs = range(121).map((i) => (L * i) / 120);
    const disc = (1 - 2 * q.k) ** 2 - 4 * q.k * q.k, lo = xs.map((x) => (x * (1 - 2 * q.k - Math.sqrt(disc))) / (2 * q.k)), hi = xs.map((x) => (x * (1 - 2 * q.k + Math.sqrt(disc))) / (2 * q.k));
    const P = [`λ₁ = ${fmt(H.l1, 3)}, λ₂ = ${fmt(H.l2, 3)} (oran ${fmt(ratio, 3)}); det M = ${fmt(H.det, 3)}, tr M = ${fmt(H.tr, 3)}, R = ${fmt(R, 3)} → <b>${cls}</b>.`];
    P.push(["Düz bölgede türevler yalnız gürültüden gelir: küçük ve yaklaşık dairesel bir bulut, iki küçük özdeğer.", "Kenarda türevler tek bir yönde (kenar normali) yayılır: bir büyük, bir küçük özdeğer; olağan k değerlerinde R negatif. θ değişince elips döner, özdeğerler yaklaşık aynı kalır (0° ile 45° arasındaki küçük fark ayrık türevin yön hatasıdır): dönmeye dayanıklılık.", "Köşede iki kenarın türevleri iki ayrı yönde kümelenir: iki büyük özdeğer, R pozitif.", "Tek parlak nokta da her yönde değişim üretir: iki büyük özdeğer ve pozitif R. Harris bunu köşeden ayıramaz (slaytın 'ne yazık ki'si)."][q.type]);
    if (q.type === 1 && R > 0) P.push(`<b>Dikkat:</b> bu k ile kenar bile R > 0 veriyor; k çok küçükse kenarlar köşe sayılır.`);
    P.push(`k = ${q.k}: R = 0 sınırı λ₂/λ₁ = ${fmt((1 - 2 * q.k - Math.sqrt(disc)) / (2 * q.k), 3)} doğrusudur; oranı bunun üstünde olan yamalar köşe sayılır. k küçüldükçe sınır yatıklaşır ve daha çok yama köşe sayılır; k → 0.25'te sınır λ₂ = λ₁'e dayanır.`);
    return {
      plots: [
        { title: "Yamadaki (f_x, f_y) saçılımı ve 2-std elipsi", xlabel: "f_x", ylabel: "f_y", series: [{ name: "türevler", x: H.pts.map((p) => p[0]), y: H.pts.map((p) => p[1]), kind: "scatter", c: 0, r: 2.5 }, { name: "elips (√λ yönleri)", x: el.map((p) => p[0]), y: el.map((p) => p[1]), c: 1 }], xlim: [-0.3, 0.3], ylim: [-0.3, 0.3] },
        { title: "Özdeğer düzlemi ve R = 0 sınırları", xlabel: "λ₁", ylabel: "λ₂", series: [{ name: "R = 0", x: xs, y: lo, c: 4, dash: true }, { name: "", x: xs, y: hi, c: 4, dash: true }, { name: "λ₁ = λ₂", x: [0, L], y: [0, L], c: 2 }, { name: "bu yama", x: [H.l1, H.l2], y: [H.l2, H.l1], kind: "scatter", c: 1, r: 5 }], xlim: [0, L], ylim: [0, L] },
      ],
      stats: [{ label: "λ₁, λ₂", value: `${fmt(H.l1, 3)}, ${fmt(H.l2, 3)}` }, { label: "R = det − k·tr²", value: R }, { label: "Shi–Tomasi min(λ₁, λ₂)", value: H.l2 }, { label: "Karar", value: cls }],
      note: P.join(" "),
    };
  },
};
