// M9.13 — Wang ve ark. (2004) "eşit MSE" deneyi: 64 × 64, 8 bit bir sahneye beş bozulma uygulanır ve her birinin gücü, MSE tam
// hedef değere eşit olacak biçimde (analitik ya da ikiye bölmeyle) ayarlanır: kontrast germe, ortalama kaydırma, JPEG benzeri 8 × 8
// DCT nicemleme, Gauss bulanıklık, Gauss gürültüsü. SSIM: 11 × 11 Gauss pencere (σ = 1.5) ya da 9 × 9 kutu, K1 = 0.01, K2 = 0.03.
import { fmt, range } from "./plot.js";

const N = 64;
export function scene() {
  const f = new Float64Array(N * N);
  let s = 12345; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const t = Float64Array.from(range(N * N), () => rnd() - 0.5), tex = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let a = 0, w = 0; for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) { const X = Math.min(N - 1, Math.max(0, x + k)), Y = Math.min(N - 1, Math.max(0, y + j)), g = Math.exp(-(j * j + k * k) / 2); a += g * t[Y * N + X]; w += g; } tex[y * N + x] = a / w; }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 110 + 30 * Math.sin(x / 9) * Math.cos(y / 13);
    if ((x - 20) ** 2 + (y - 22) ** 2 <= 140) v = 175;
    if (x >= 36 && x < 58 && y >= 34 && y < 56) v = 70 + 120 * tex[y * N + x] + 0.8 * (x - 36);
    if (y >= 46 && x < 30 && (x + y) % 6 < 3) v = 150;
    f[y * N + x] = Math.max(30, Math.min(220, v + 40 * tex[y * N + x]));
  }
  return f;
}
const mse = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
function gblur(a, s) { if (s <= 0) return Float64Array.from(a); const R = Math.ceil(3 * s), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / s) ** 2)), S = w.reduce((p, q) => p + q, 0), t = new Float64Array(N * N), o = new Float64Array(N * N); const at = (b, x, y) => b[Math.min(N - 1, Math.max(0, y)) * N + Math.min(N - 1, Math.max(0, x))]; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(a, x + j - R, y); t[y * N + x] = v / S; } for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(t, x, y + j - R); o[y * N + x] = v / S; } return o; }
const C8 = range(8).map((u) => range(8).map((x) => (u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * Math.cos(((2 * x + 1) * u * Math.PI) / 16)));
function jpeg(a, q) {
  const o = new Float64Array(N * N);
  for (let by = 0; by < N; by += 8) for (let bx = 0; bx < N; bx += 8) {
    const B = range(8).map((y) => range(8).map((x) => a[(by + y) * N + bx + x] - 128));
    const D = range(8).map((u) => range(8).map((v) => { let s = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s += C8[u][y] * C8[v][x] * B[y][x]; return Math.round(s / (q * (1 + u + v))) * q * (1 + u + v); }));
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { let s = 0; for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) s += C8[u][y] * C8[v][x] * D[u][v]; o[(by + y) * N + bx + x] = s + 128; }
  }
  return o;
}
const bisect = (fn, lo, hi, target) => { for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (fn(m) < target) lo = m; else hi = m; } return (lo + hi) / 2; };
export function distort(f, kind, M) {
  const mu = f.reduce((s, v) => s + v, 0) / f.length, vr = f.reduce((s, v) => s + (v - mu) ** 2, 0) / f.length;
  if (kind === 0) { const a = 1 + Math.sqrt(M / vr); return f.map((v) => mu + a * (v - mu)); }
  if (kind === 1) return f.map((v) => v + Math.sqrt(M));
  if (kind === 2) { const q = bisect((q) => mse(f, jpeg(f, q)), 0.1, 200, M); return jpeg(f, q); }
  if (kind === 3) { const s = bisect((s) => mse(f, gblur(f, s)), 0.05, 20, M); return gblur(f, s); }
  let st = 99; const nz = Float64Array.from(range(N * N), () => { const u = ((st = (st * 16807) % 2147483647) / 2147483647) || 1e-9, v = (st = (st * 16807) % 2147483647) / 2147483647; return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); });
  const m0 = nz.reduce((s, v) => s + v * v, 0) / nz.length; return f.map((v, i) => v + nz[i] * Math.sqrt(M / m0));
}
export function ssimMap(a, b, win, K1 = 0.01, K2 = 0.03, L = 255) {
  const C1 = (K1 * L) ** 2, C2 = (K2 * L) ** 2, R = win === 0 ? 5 : 4, w = [];
  for (let j = -R; j <= R; j++) for (let k = -R; k <= R; k++) w.push([j, k, win === 0 ? Math.exp(-(j * j + k * k) / (2 * 1.5 * 1.5)) : 1]);
  const ws = w.reduce((s, e) => s + e[2], 0), out = [];
  for (let y = R; y < N - R; y++) for (let x = R; x < N - R; x++) {
    let mx = 0, my = 0; for (const [j, k, g] of w) { mx += g * a[(y + j) * N + x + k]; my += g * b[(y + j) * N + x + k]; } mx /= ws; my /= ws;
    let vx = 0, vy = 0, cxy = 0; for (const [j, k, g] of w) { const dx = a[(y + j) * N + x + k] - mx, dy = b[(y + j) * N + x + k] - my; vx += g * dx * dx; vy += g * dy * dy; cxy += g * dx * dy; } vx /= ws; vy /= ws; cxy /= ws;
    out.push({ x, y, s: ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2)) });
  }
  return { R, out };
}
const KINDS = ["kontrast germe", "ortalama kaydırma", "JPEG benzeri (8 × 8 DCT)", "Gauss bulanıklık", "Gauss gürültüsü"];
export default {
  params: [
    { id: "M", label: "Hedef MSE", min: 20, max: 600, step: 10, value: 210 },
    { id: "kind", label: "Gösterilen bozulma", options: KINDS, value: 2 },
    { id: "win", label: "SSIM penceresi", options: ["11 × 11 Gauss (σ = 1.5)", "9 × 9 kutu"], value: 0 },
  ],
  run(q) {
    const f = scene(), all = range(5).map((k) => distort(f, k, q.M)), mses = all.map((g) => mse(f, g)), ss = all.map((g) => { const m = ssimMap(f, g, q.win); return m.out.reduce((s, e) => s + e.s, 0) / m.out.length; });
    const g = all[q.kind], m = ssimMap(f, g, q.win), W2 = N - 2 * m.R, map = new Float64Array(W2 * W2); m.out.forEach((e) => (map[(e.y - m.R) * W2 + e.x - m.R] = e.s));
    const psnr = 10 * Math.log10((255 * 255) / q.M), clip = Array.from(g).filter((v) => v < 0 || v > 255).length;
    const P = [`Beş bozulmanın da MSE'si ${fmt(q.M, 3)} (PSNR ${fmt(psnr, 3)} dB); SSIM sırasıyla ${ss.map((v) => fmt(v, 3)).join(", ")}.`];
    P.push("Ortalama kaydırma yalnız parlaklık terimini, kontrast germe yalnız kontrast terimini etkiler: yapı korunur, SSIM yüksek kalır. JPEG ve bulanıklık yapıyı (yerel ilintiyi) bozar, gürültü yapıyı rastgele bozar: aynı MSE'de SSIM çok daha düşüktür. Wang ve ark.'ın 'Einstein' figüründe kaydırma 0.99, germe 0.92, gürültü 0.77, bulanıklık 0.71, JPEG 0.69'du: üstteki iki bozulma bu görüntüde de en yüksek SSIM'i alıyor; alttaki üçü birbirine yakın ve sıraları görüntüye ve pencereye bağlı.");
    if (clip) P.push(`<b>Not:</b> ${clip} piksel 0–255 dışına taştı (kırpılmadan ölçüldü).`);
    return {
      plots: [
        { title: `${KINDS[q.kind]}, MSE = ${fmt(mses[q.kind], 4)}`, img: { w: N, h: N, data: Array.from(g, (v) => Math.max(0, Math.min(1, v / 255))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Yerel SSIM haritası (beyaz = 1)", img: { w: W2, h: W2, data: Array.from(map, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Aynı MSE'de ortalama SSIM", xlabel: "bozulma", ylabel: "SSIM", wide: true, size: { w: 620, h: 170 }, series: [{ name: "SSIM", x: range(5), y: ss, kind: "bar", c: 0 }], xticks: range(5).map((i) => [i, ["germe", "kaydırma", "JPEG", "bulanık", "gürültü"][i]]), ylim: [0, 1] },
      ],
      stats: [{ label: "PSNR (hepsinde)", value: `${fmt(psnr, 4)} dB` }, ...KINDS.map((k, i) => ({ label: `SSIM: ${k}`, value: ss[i] }))],
      note: P.join(" "),
    };
  },
};
