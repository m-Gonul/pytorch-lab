// M9.15 — MOSSE (Bolme ve ark. 2010) ile sabit şablonlu NCC izleyicinin karşılaştırması. 96 × 96, 60 karelik yapay video: dokulu
// zemin üzerinde dokulu bir hedef (14 × 14) eğri bir yolda hareket eder; isteğe bağlı kısmi örtme (kare 25–32), aydınlatma kayması ve
// hedef görünüşünün yavaşça değişmesi (doku döner). MOSSE: 32 × 32 pencere, log + normalize + kosinüs penceresi, istenen çıktı
// G = Gauss tepe (σ = 2), A ← ηG·F̄ + (1 − η)A, B ← ηF·F̄ + (1 − η)B, H̄ = A/(B + λ); güven ölçüsü PSR.
import { fmt, range } from "./plot.js";
import { fft2 } from "./circconv.js";

const W = 96, N = 32, FR = 60;
export function video(r, q) {
  const frames = [], pos = [];
  for (let t = 0; t < FR; t++) {
    const cx = 30 + 36 * (t / FR) + 8 * Math.sin(t / 6) * q.speed, cy = 48 + 22 * Math.sin((t / FR) * Math.PI * 1.5) * q.speed, rot = q.morph * t * 0.03;
    const f = new Float64Array(W * W);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      let v = 0.45 + 0.1 * Math.sin(x / 2.7 + y / 4.1) + 0.06 * Math.cos(x / 1.3 - y / 2.2);
      const u = x - cx, w = y - cy;
      if (Math.abs(u) <= 7 && Math.abs(w) <= 7) { const a = u * Math.cos(rot) - w * Math.sin(rot), b = u * Math.sin(rot) + w * Math.cos(rot); v = 0.5 + 0.35 * Math.sign(Math.sin(a / 1.6)) * Math.sign(Math.cos(b / 2.1)); }
      if (q.occ && t >= 25 && t < 33 && x >= cx - 2 && x <= cx + 9 && y >= cy - 9 && y <= cy + 9) v = 0.2;
      f[y * W + x] = (v + q.noise * r.normal()) * (1 + q.light * (t / FR));
    }
    frames.push(f); pos.push([cx, cy]);
  }
  return { frames, pos };
}
const cosw = range(N * N).map((i) => Math.sin((Math.PI * ((i % N) + 0.5)) / N) * Math.sin((Math.PI * (((i / N) | 0) + 0.5)) / N));
function crop(f, cx, cy) { const o = new Float64Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const X = Math.min(W - 1, Math.max(0, Math.round(cx) - N / 2 + x)), Y = Math.min(W - 1, Math.max(0, Math.round(cy) - N / 2 + y)); o[y * N + x] = f[Y * W + X]; } return o; }
// Başlatma için küçük afin bozulmalar (Bolme ve ark.): dönme ve ölçekle bilineer örneklenmiş yama.
function cropAff(f, cx, cy, th, sc) { const o = new Float64Array(N * N), c = Math.cos(th), s = Math.sin(th); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const dx = x - N / 2, dy = y - N / 2, u = Math.min(W - 1.001, Math.max(0, cx + sc * (c * dx - s * dy))), v = Math.min(W - 1.001, Math.max(0, cy + sc * (s * dx + c * dy))), x0 = Math.floor(u), y0 = Math.floor(v), a = u - x0, b = v - y0; o[y * N + x] = (1 - a) * (1 - b) * f[y0 * W + x0] + a * (1 - b) * f[y0 * W + x0 + 1] + (1 - a) * b * f[(y0 + 1) * W + x0] + a * b * f[(y0 + 1) * W + x0 + 1]; } return o; }
function prep(p) { const l = p.map((v) => Math.log(Math.max(v, 1e-3) + 1)), m = l.reduce((a, b) => a + b, 0) / l.length, s = Math.sqrt(l.reduce((a, b) => a + (b - m) ** 2, 0) / l.length) + 1e-6; return l.map((v, i) => ((v - m) / s) * cosw[i]); }
export function mosse(frames, start, eta, lam, init = true, gate = 0) {
  const G = fft2(Float64Array.from(range(N * N), (i) => Math.exp(-(((i % N) - N / 2) ** 2 + (((i / N) | 0) - N / 2) ** 2) / (2 * 4))), new Float64Array(N * N), N);
  let [cx, cy] = start;
  let Ar = new Float64Array(N * N), Ai = new Float64Array(N * N), B = new Float64Array(N * N);
  const upd = (F, e) => { for (let i = 0; i < N * N; i++) { const fr = F.re[i], fi = F.im[i]; Ar[i] = e * (G.re[i] * fr + G.im[i] * fi) + (1 - e) * Ar[i]; Ai[i] = e * (G.im[i] * fr - G.re[i] * fi) + (1 - e) * Ai[i]; B[i] = e * (fr * fr + fi * fi) + (1 - e) * B[i]; } };
  const nInit = init ? 8 : 1;
  for (let k = 0; k < nInit; k++) { const th = nInit > 1 ? -0.18 + (0.36 * k) / (nInit - 1) : 0, sc = nInit > 1 ? 0.92 + 0.16 * ((k * 5) % 8) / 7 : 1; const F = fft2(prep(cropAff(frames[0], cx, cy, th, sc)), new Float64Array(N * N), N); upd(F, 1 / (k + 1)); }
  const track = [[cx, cy]], psr = [NaN];
  for (let t = 1; t < frames.length; t++) {
    const F = fft2(prep(crop(frames[t], cx, cy)), new Float64Array(N * N), N), Rr = new Float64Array(N * N), Ri = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) { const hr = Ar[i] / (B[i] + lam), hi = Ai[i] / (B[i] + lam); Rr[i] = hr * F.re[i] - hi * F.im[i]; Ri[i] = hr * F.im[i] + hi * F.re[i]; }
    const R = fft2(Rr, Ri, N, true).re; let k = 0; for (let i = 1; i < N * N; i++) if (R[i] > R[k]) k = i;
    const px = k % N, py = (k / N) | 0; const side = []; for (let i = 0; i < N * N; i++) { const dx = (i % N) - px, dy = ((i / N) | 0) - py; if (Math.abs(dx) > 5 || Math.abs(dy) > 5) side.push(R[i]); }
    const sm = side.reduce((a, b) => a + b, 0) / side.length, ss = Math.sqrt(side.reduce((a, b) => a + (b - sm) ** 2, 0) / side.length) + 1e-12;
    psr.push((R[k] - sm) / ss);
    cx += px - N / 2; cy += py - N / 2; track.push([cx, cy]);
    if (eta > 0 && !(gate > 0 && psr[psr.length - 1] < gate)) upd(fft2(prep(crop(frames[t], cx, cy)), new Float64Array(N * N), N), eta);
  }
  return { track, psr };
}
export function nccTrack(frames, start) {
  const h = 7, T = 2 * h + 1, [sx, sy] = start.map(Math.round), tpl = []; for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) tpl.push(frames[0][(sy + j) * W + sx + i]);
  const tm = tpl.reduce((a, b) => a + b, 0) / tpl.length, tz = tpl.map((v) => v - tm), tn = Math.sqrt(tz.reduce((a, v) => a + v * v, 0));
  let cx = sx, cy = sy; const track = [[cx, cy]];
  for (let t = 1; t < frames.length; t++) {
    let best = -2, bx = cx, by = cy;
    for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
      const X = cx + dx, Y = cy + dy; if (X - h < 0 || Y - h < 0 || X + h >= W || Y + h >= W) continue;
      let m = 0; for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) m += frames[t][(Y + j) * W + X + i]; m /= T * T;
      let s = 0, v2 = 0, k = 0; for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) { const d = frames[t][(Y + j) * W + X + i] - m; s += tz[k++] * d; v2 += d * d; }
      const g = s / (tn * Math.sqrt(v2) + 1e-12); if (g > best) { best = g; bx = X; by = Y; }
    }
    cx = bx; cy = by; track.push([cx, cy]);
  }
  return { track };
}
export default {
  params: [
    { id: "eta", label: "MOSSE öğrenme oranı η", min: 0, max: 0.4, step: 0.005, value: 0.125 },
    { id: "lam", label: "Düzenleme λ (|F|² ≈ 10³ ölçeğinde)", min: 0.01, max: 10000, value: 100, log: true },
    { id: "gate", label: "PSR eşiği: altında güncelleme yapma (0 = kapalı)", min: 0, max: 10, step: 0.5, value: 0 },
    { id: "speed", label: "Hareket genliği", min: 0.3, max: 1.5, step: 0.05, value: 1 },
    { id: "occ", label: "Kısmi örtme (kare 25–32)", options: ["yok", "var"], value: 1 },
    { id: "morph", label: "Görünüş değişimi (dönme hızı)", min: 0, max: 1, step: 0.05, value: 0.5 },
    { id: "light", label: "Aydınlatma kayması", min: -0.5, max: 1, step: 0.05, value: 0.4 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.02 },
  ],
  seed: true,
  run(q, r) {
    const { frames, pos } = video(r, q), M = mosse(frames, pos[0], q.eta, q.lam, true, q.gate), C = nccTrack(frames, pos[0]);
    const err = (tr) => tr.map((p, t) => Math.hypot(p[0] - pos[t][0], p[1] - pos[t][1]));
    const eM = err(M.track), eC = err(C.track), mean = (a) => a.reduce((s, v) => s + v, 0) / a.length, lost = (a) => a.filter((v) => v > 8).length;
    const psrOcc = q.occ ? Math.min(...M.psr.slice(25, 33)) : NaN, psrFree = mean(M.psr.slice(5, 22));
    const P = [`Ortalama merkez hatası MOSSE ${fmt(mean(eM), 3)}, sabit şablonlu NCC ${fmt(mean(eC), 3)} piksel; 8 pikselden fazla sapılan kare: ${lost(eM)} ve ${lost(eC)} (60 karede).`];
    P.push(q.eta === 0 ? "η = 0: MOSSE ilk karedeki süzgeçle kalır, görünüş değişimine uyum sağlamaz." : `η = ${q.eta}: süzgeç her karede yeni görünüşe doğru güncellenir (üstel ortalama); ${q.eta > 0.25 ? "çok büyük η örtmede örtücüyü de öğrenir ve kayabilir." : "orta η görünüş değişimini izler ama birkaç karelik örtmeyi tolere eder."}`);
    P.push(`PSR (tepe/yan lob oranı) örtmesiz karelerde ortalama ${fmt(psrFree, 3)}${q.occ ? `, örtme sırasında en düşük ${fmt(psrOcc, 3)}: Bolme ve ark. PSR < 7 civarını başarısızlık işareti sayar ve güncellemeyi durdurmayı önerir` : ""}.${q.gate > 0 ? ` PSR eşiği ${q.gate}: güven düşükken süzgeç güncellenmiyor, örtücü öğrenilmiyor. Ama bu bir ödünleşimdir: görünüş hızla değişirken PSR de düşer ve güncelleme durunca süzgeç hedefin gerisinde kalabilir.` : ""}`);
    const last = frames[FR - 1];
    return {
      plots: [
        { title: "Son kare ve izler", img: { w: W, h: W, data: Array.from(last, (v) => Math.max(0, Math.min(1, v / (1 + Math.max(0, q.light))))) }, side: true, size: { w: 300, h: 290 }, series: [{ name: "gerçek", x: pos.map((p) => p[0] + 0.5), y: pos.map((p) => W - p[1] - 0.5), c: 2 }, { name: "MOSSE", x: M.track.map((p) => p[0] + 0.5), y: M.track.map((p) => W - p[1] - 0.5), c: 0 }, { name: "NCC", x: C.track.map((p) => p[0] + 0.5), y: C.track.map((p) => W - p[1] - 0.5), c: 1, dash: true }] },
        { title: "Merkez hatası ve PSR", xlabel: "kare", ylabel: "piksel / PSR", side: true, size: { w: 300, h: 290 }, series: [{ name: "MOSSE hata", x: range(FR), y: eM, c: 0 }, { name: "NCC hata", x: range(FR), y: eC, c: 1, dash: true }, { name: "PSR / 2", x: range(FR), y: M.psr.map((v) => v / 2), c: 3 }] },
      ],
      stats: [{ label: "Ortalama hata MOSSE / NCC", value: `${fmt(mean(eM), 3)} / ${fmt(mean(eC), 3)} px` }, { label: "Kaybedilen kare (> 8 px)", value: `${lost(eM)} / ${lost(eC)}` }, { label: "PSR: serbest / örtmede en düşük", value: `${fmt(psrFree, 3)} / ${q.occ ? fmt(psrOcc, 3) : "—"}` }],
      note: P.join(" "),
    };
  },
};
