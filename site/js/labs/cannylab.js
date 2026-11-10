// M9.6 — Canny, Marr–Hildreth ve "Sobel + tek eşik" aynı test deseninde. Desen: disk (her yön), dikdörtgen, eşkenar dörtgen
// (45° kenarlar); mercek bulanıklığı σ = 1 ve Gauss gürültüsü. Gerçek kenarlar bulanıklaştırılmamış desenden (sağ ya da alt komşusu
// farklı pikseller). Ölçütler 1 piksel toleransla kesinlik/duyarlılık, F1, 8-bağlı bileşen sayısı (parçalanma).
import { fmt, range } from "./plot.js";

export const W = 96;
export function pattern() {
  const f = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let v = 60;
    if ((x - 32) ** 2 + (y - 32) ** 2 <= 22 * 22) v = 190;
    if (x >= 60 && x < 88 && y >= 10 && y < 42) v = 130;
    if (Math.abs(x - 66) + Math.abs(y - 70) <= 17) v = 220;
    if (x >= 10 && x < 44 && y >= 64 && y < 86) v = 120;
    f[y * W + x] = v;
  }
  return f;
}
export function truth(f) { const t = new Uint8Array(W * W); for (let y = 0; y < W - 1; y++) for (let x = 0; x < W - 1; x++) if (f[y * W + x] !== f[y * W + x + 1] || f[y * W + x] !== f[(y + 1) * W + x]) t[y * W + x] = 1; return t; }
const at = (a, x, y) => a[Math.min(W - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))];
export function gblur(a, s) {
  if (s <= 0) return Float64Array.from(a);
  const R = Math.ceil(3 * s), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / s) ** 2)), S = w.reduce((p, q) => p + q, 0), t = new Float64Array(W * W), o = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(a, x + j - R, y); t[y * W + x] = v / S; }
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(t, x, y + j - R); o[y * W + x] = v / S; }
  return o;
}
export function sobel(a) {
  const gx = new Float64Array(W * W), gy = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    gx[y * W + x] = (at(a, x + 1, y - 1) + 2 * at(a, x + 1, y) + at(a, x + 1, y + 1) - at(a, x - 1, y - 1) - 2 * at(a, x - 1, y) - at(a, x - 1, y + 1)) / 8;
    gy[y * W + x] = (at(a, x - 1, y + 1) + 2 * at(a, x, y + 1) + at(a, x + 1, y + 1) - at(a, x - 1, y - 1) - 2 * at(a, x, y - 1) - at(a, x + 1, y - 1)) / 8;
  }
  return { gx, gy, M: gx.map((v, i) => Math.hypot(v, gy[i])) };
}
export function nms(M, gx, gy) {
  const o = new Float64Array(W * W);
  for (let y = 1; y < W - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x; let a = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI; if (a < 0) a += 180;
    const [dx, dy] = a < 22.5 || a >= 157.5 ? [1, 0] : a < 67.5 ? [1, 1] : a < 112.5 ? [0, 1] : [-1, 1];
    if (M[i] >= M[i + dy * W + dx] && M[i] >= M[i - dy * W - dx]) o[i] = M[i];
  }
  return o;
}
export function hyst(N, lo, hi) {
  const e = new Uint8Array(W * W), st = [];
  for (let i = 0; i < W * W; i++) if (N[i] >= hi) { e[i] = 1; st.push(i); }
  while (st.length) { const i = st.pop(), x = i % W, y = (i / W) | 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= W) continue; const j = Y * W + X; if (!e[j] && N[j] >= lo) { e[j] = 1; st.push(j); } } }
  return e;
}
export function zerocross(L, thr) {
  const e = new Uint8Array(W * W);
  for (let y = 1; y < W - 1; y++) for (let x = 1; x < W - 1; x++) for (const [dx, dy] of [[1, 0], [0, 1], [1, 1], [1, -1]]) { const a = L[(y - dy) * W + x - dx], b = L[(y + dy) * W + x + dx]; if (a * b < 0 && Math.abs(a - b) > thr) { e[y * W + x] = 1; break; } }
  return e;
}
export function score(e, t) {
  const near = (m, i) => { const x = i % W, y = (i / W) | 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < W && Y < W && m[Y * W + X]) return true; } return false; };
  let nd = 0, tp = 0, nt = 0, hit = 0;
  for (let i = 0; i < W * W; i++) { if (e[i]) { nd++; if (near(t, i)) tp++; } if (t[i]) { nt++; if (near(e, i)) hit++; } }
  const P = nd ? tp / nd : 0, R = nt ? hit / nt : 0, lab = new Int32Array(W * W); let comp = 0;
  for (let i = 0; i < W * W; i++) if (e[i] && !lab[i]) { comp++; const st = [i]; lab[i] = comp; while (st.length) { const j = st.pop(), x = j % W, y = (j / W) | 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= W) continue; const k = Y * W + X; if (e[k] && !lab[k]) { lab[k] = comp; st.push(k); } } } }
  return { P, R, F: P + R > 0 ? (2 * P * R) / (P + R) : 0, n: nd, comp, nt };
}
export function noisy(r, sn) { const c = pattern(), b = gblur(c, 1); return { clean: c, g: b.map((v) => v + sn * r.normal()) }; }

const STAGES = ["düzleştirilmiş görüntü", "gradyan büyüklüğü", "NMS sonrası", "güçlü (beyaz) ve zayıf (gri) pikseller", "son kenarlar"];
export default {
  params: [
    { id: "m", label: "Yöntem", options: ["Canny", "Marr–Hildreth (LoG + sıfır geçişi)", "Sobel büyüklüğü + tek eşik"], value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 40, step: 1, value: 10 },
    { id: "s", label: "Gauss σ", min: 0.5, max: 4, step: 0.1, value: 1.4 },
    { id: "th", label: "Yüksek eşik T_H (büyüklük, gri seviye/piksel)", min: 1, max: 60, step: 0.5, value: 12, on: { m: [0, 2] } },
    { id: "ratio", label: "T_H / T_L", min: 1, max: 5, step: 0.1, value: 2.5, on: { m: [0] } },
    { id: "zt", label: "Sıfır geçişi eşiği (max |LoG|'un %'si)", min: 0, max: 30, step: 1, value: 4, on: { m: [1] } },
    { id: "stage", label: "Gösterilen aşama", options: STAGES, value: 4, on: { m: [0] } },
  ],
  seed: true,
  run(q, r) {
    const { clean, g } = noisy(r, q.noise), t = truth(clean), fs = gblur(g, q.s);
    let e, show, P = [];
    if (q.m === 1) {
      const L = new Float64Array(W * W);
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) L[y * W + x] = at(fs, x + 1, y) + at(fs, x - 1, y) + at(fs, x, y + 1) + at(fs, x, y - 1) - 4 * at(fs, x, y);
      const mx = Math.max(...L.map(Math.abs)); e = zerocross(L, (q.zt / 100) * mx); show = e;
      P.push(`LoG: σ = ${q.s} ile Gauss, sonra Laplasyen; sıfır geçişi için karşılıklı komşuların işareti farklı ve farkları max |LoG|'un %${q.zt}'inden büyük olmalı. ${q.zt === 0 ? "Eşik 0: gürültünün her işaret değişimi bir 'kenar' olur; spagetti benzeri kapalı eğriler." : ""}`);
    } else {
      const { gx, gy, M } = sobel(fs);
      if (q.m === 2) { e = Uint8Array.from(M, (v) => (v >= q.th ? 1 : 0)); show = e; P.push(`Tek eşik T = ${q.th}: kenarlar kalın kalır (NMS yok) ve eşik düşükse gürültü, yüksekse kopukluk üretir.`); }
      else {
        const N = nms(M, gx, gy), lo = q.th / q.ratio; e = hyst(N, lo, q.th);
        show = [fs.map((v) => v / 255), M.map((v) => Math.min(1, v / 40)), N.map((v) => Math.min(1, v / 40)), N.map((v) => (v >= q.th ? 1 : v >= lo ? 0.45 : 0)), e][q.stage];
        const strong = N.filter((v) => v >= q.th).length, weak = N.filter((v) => v >= lo && v < q.th).length;
        P.push(`Canny: σ = ${q.s}, T_H = ${q.th}, T_L = ${fmt(lo, 3)}. NMS sonrası ${strong} güçlü, ${weak} zayıf piksel; histerezis güçlülere 8-bağlı zayıfları ekliyor.`);
      }
    }
    const sc = score(e, t);
    P.push(`1 piksel toleransla kesinlik ${fmt(sc.P, 3)}, duyarlılık ${fmt(sc.R, 3)}, F1 ${fmt(sc.F, 3)}; ${sc.n} kenar pikseli (gerçek ${sc.nt}), ${sc.comp} bağlı bileşen (desende 4 nesne var).`);
    if (sc.n > 1.6 * sc.nt) P.push("Kenar pikseli sayısı gerçeğin çok üstünde: kenarlar kalın ya da gürültü kenar sanılıyor.");
    return {
      plots: [
        { title: "Gürültülü girdi", img: { w: W, h: W, data: Array.from(g, (v) => Math.max(0, Math.min(1, v / 255))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: q.m === 0 ? STAGES[q.stage] : "kenar haritası", img: { w: W, h: W, data: Array.from(q.m === 0 && q.stage < 4 ? show : e) }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "Kesinlik / duyarlılık", value: `${fmt(sc.P, 3)} / ${fmt(sc.R, 3)}` }, { label: "F1", value: sc.F }, { label: "Kenar pikseli (gerçek)", value: `${sc.n} (${sc.nt})` }, { label: "8-bağlı bileşen", value: sc.comp }],
      note: P.join(" "),
    };
  },
};
