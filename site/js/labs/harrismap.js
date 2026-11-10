// M9.7 — Harris ve Shi–Tomasi bir test deseninde. Desen (96 × 96): kare, eşkenar dörtgen, üçgen (11 gerçek köşe), disk (köşe yok)
// ve üç tek parlak nokta; θ kadar döndürülebilir (gerçek köşeler de döner). [−1, 0, 1]/2 türevler, Gauss ağırlıklı M, yanıt
// R = det − k·tr² ya da min(λ1, λ2); görüntü üstündeki noktalar plot.js kuralıyla (y yukarı) çizilir; eşik max yanıtın oranı; 3 × 3 yerel maksimum. Ölçüt: 2.5 piksel içinde eşleşme.
import { fmt, range } from "./plot.js";

const W = 96, CC = 47.5;
const POLY = [
  { v: [[18, 18], [40, 18], [40, 40], [18, 40]], val: 200 },
  { v: [[72, 14], [84, 26], [72, 38], [60, 26]], val: 160 },
  { v: [[18, 76], [40, 76], [29, 56]], val: 230 },
];
const DOTS = [[50, 50], [84, 56], [54, 84]];
const inPoly = (p, v) => { let c = false; for (let i = 0, j = v.length - 1; i < v.length; j = i++) { const [xi, yi] = v[i], [xj, yj] = v[j]; if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c; } return c; };
const rot = (p, t) => [CC + (p[0] - CC) * Math.cos(t) - (p[1] - CC) * Math.sin(t), CC + (p[0] - CC) * Math.sin(t) + (p[1] - CC) * Math.cos(t)];
export function scene(t, sn, r) {
  const f = new Float64Array(W * W), ss = 3;
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let a = 0; a < ss; a++) for (let b = 0; b < ss; b++) {
      const p = rot([x + (a + 0.5) / ss - 0.5, y + (b + 0.5) / ss - 0.5], -t);
      let v = 50;
      if ((p[0] - 70) ** 2 + (p[1] - 70) ** 2 <= 12 * 12) v = 180;
      for (const P of POLY) if (inPoly(p, P.v)) v = P.val;
      acc += v;
    }
    f[y * W + x] = acc / (ss * ss) + sn * r.normal();
  }
  for (const d of DOTS) { const [x, y] = rot(d, t).map(Math.round); f[y * W + x] = 255; }
  return { f, gt: POLY.flatMap((P) => P.v.map((p) => rot(p, t))), dots: DOTS.map((d) => rot(d, t)) };
}
export function response(f, sw, k, mode) {
  const fx = new Float64Array(W * W), fy = new Float64Array(W * W);
  for (let y = 1; y < W - 1; y++) for (let x = 1; x < W - 1; x++) { fx[y * W + x] = (f[y * W + x + 1] - f[y * W + x - 1]) / 2; fy[y * W + x] = (f[(y + 1) * W + x] - f[(y - 1) * W + x]) / 2; }
  const R = Math.ceil(3 * sw), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / sw) ** 2));
  const blur = (a) => { const t = new Float64Array(W * W), o = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let s = 0; for (let j = 0; j <= 2 * R; j++) { const X = x + j - R; if (X >= 0 && X < W) s += w[j] * a[y * W + X]; } t[y * W + x] = s; } for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let s = 0; for (let j = 0; j <= 2 * R; j++) { const Y = y + j - R; if (Y >= 0 && Y < W) s += w[j] * t[Y * W + x]; } o[y * W + x] = s; } return o; };
  const A = blur(fx.map((v) => v * v)), B = blur(fx.map((v, i) => v * fy[i])), Cm = blur(fy.map((v) => v * v));
  return A.map((a, i) => { const c = Cm[i], b = B[i], tr = a + c, det = a * c - b * b; return mode === 0 ? det - k * tr * tr : tr / 2 - Math.sqrt(Math.max(0, (tr / 2) ** 2 - det)); });
}
export function detect(Rm, frac) {
  const mx = Math.max(...Rm), T = frac * mx, out = [];
  for (let y = 2; y < W - 2; y++) for (let x = 2; x < W - 2; x++) { const v = Rm[y * W + x]; if (v <= T) continue; let ok = true; for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && Rm[(y + dy) * W + x + dx] > v) { ok = false; break; } if (ok) out.push([x, y]); }
  return out;
}
const near = (a, b, d) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= d * d;
export default {
  params: [
    { id: "mode", label: "Yanıt", options: ["Harris: R = det M − k·tr²M", "Shi–Tomasi: min(λ₁, λ₂)"], value: 0 },
    { id: "k", label: "k", min: 0.01, max: 0.25, step: 0.005, value: 0.04, on: { mode: [0] } },
    { id: "T", label: "Eşik T (max yanıtın oranı)", min: 0.001, max: 0.5, value: 0.01, log: true },
    { id: "sw", label: "Gauss pencere σ", min: 0.7, max: 4, step: 0.1, value: 1.5 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 30, step: 1, value: 5 },
    { id: "th", label: "Sahne dönmesi θ (°)", min: 0, max: 90, step: 1, value: 0 },
  ],
  seed: true,
  run(q, r) {
    const { f, gt, dots } = scene((q.th * Math.PI) / 180, q.noise, r), Rm = response(f, q.sw, q.k, q.mode), D = detect(Rm, q.T);
    const tol = 2.5, hitGT = gt.filter((g) => D.some((d) => near(d, g, tol))).length, hitDot = dots.filter((g) => D.some((d) => near(d, g, tol))).length;
    const fp = D.filter((d) => !gt.some((g) => near(d, g, tol)) && !dots.some((g) => near(d, g, tol))).length;
    const P = [`${D.length} tespit: ${hitGT}/${gt.length} gerçek köşe bulundu, ${hitDot}/3 tek nokta da "köşe" sayıldı, ${fp} sahte tespit (kenar, disk ya da gürültü).`];
    if (q.mode === 0) P.push(`k = ${q.k}${q.k > 0.2 ? ": üst sınıra (0.25) yakın; yalnız λ₁ ≈ λ₂ olan yamalar pozitif kalır, gerçek köşeler bile kaybolabilir" : q.k < 0.03 ? ": küçük k kenarlara da pozitif R verdirir; sahte tespitler artar" : ""}.`);
    else P.push("Shi–Tomasi doğrudan küçük özdeğeri eşikler: kenarda min(λ) küçük olduğu için k'ye gerek yoktur; OpenCV'nin goodFeaturesToTrack'i bunu kullanır.");
    if (q.th > 0) P.push(`Sahne ${q.th}° döndü; Harris matrisinin özdeğerleri dönmeyle (ayrık türevin küçük yön hatası dışında) değişmediği için köşeler hâlâ bulunuyor; eğik kenarların merdiven biçimli örneklenmesi ise birkaç sahte tepe ekleyebilir.`);
    if (q.sw >= 2.5) P.push(`Pencere σ = ${q.sw} büyük: yanıtın tepesi köşeden şeklin içine kayar (gürültüsüz sahnede σ = 3'te sivri üçgen köşelerinde 3.6–4 piksel); 2.5 piksellik toleransla bu kaymış tepeler 'bulunamadı' ve 'sahte' sayılıyor. Ölçek büyüdükçe konum belirsizliği artar.`);
    const mx = Math.max(...Rm), img = Array.from(f, (v) => Math.max(0, Math.min(1, v / 255)));
    return {
      plots: [
        { title: "Girdi ve tespitler (turuncu), gerçek köşeler (yeşil)", img: { w: W, h: W, data: img }, side: true, size: { w: 300, h: 290 }, series: [{ name: "tespit", x: D.map((d) => d[0] + 0.5), y: D.map((d) => W - d[1] - 0.5), kind: "scatter", c: 2, r: 4 }, { name: "gerçek köşe", x: gt.map((g) => g[0] + 0.5), y: gt.map((g) => W - g[1] - 0.5), kind: "scatter", c: 1, r: 2 }] },
        { title: q.mode === 0 ? "R haritası (pozitif kısım)" : "min(λ₁, λ₂) haritası", img: { w: W, h: W, data: Array.from(Rm, (v) => Math.max(0, Math.min(1, Math.sqrt(Math.max(v, 0) / mx)))) }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "Gerçek köşe bulundu", value: `${hitGT} / ${gt.length}` }, { label: "Tek nokta tespiti", value: `${hitDot} / 3` }, { label: "Sahte tespit", value: fp }, { label: "Toplam tespit", value: D.length }],
      note: P.join(" "),
    };
  },
};
