// M9.9 — Bir rengin bütün modellerdeki karşılığı ve HSI'da düzenleme. RGB (0–255) → HSI (Gonzalez formülleri), HSV (OpenCV
// tanımı), CMY/CMYK, YCbCr (OpenCV YCrCb katsayıları), CIE L*a*b* (sRGB → doğrusal → XYZ, D65). HSI'da ton döndürme, doygunluk ve
// yoğunluk ölçekleme, sonra sektör formülleriyle RGB'ye dönüş; kırpılırsa bildirir. Yoğunluk eksenine dik düzlemde (α, β) izdüşüm.
import { fmt, range } from "./plot.js";

const D = Math.PI / 180;
export function rgb2hsi(r, g, b) {
  // Payda yalnız R = G = B'de 0'dır; Gonzalez'in +ε'u bu durumda θ = arccos(0) = 90° verir (burada açıkça).
  const num = 0.5 * (r - g + (r - b)), den = Math.sqrt((r - g) ** 2 + (r - b) * (g - b));
  const th = den < 1e-12 ? 90 : Math.acos(Math.max(-1, Math.min(1, num / den))) / D, H = b <= g ? th : 360 - th, sum = r + g + b;
  return { H, S: sum > 0 ? 1 - (3 * Math.min(r, g, b)) / sum : 0, I: sum / 3 };
}
export function hsi2rgb(H, S, I) {
  H = ((H % 360) + 360) % 360;
  const sec = (h) => [I * (1 - S), I * (1 + (S * Math.cos(h * D)) / Math.cos((60 - h) * D))];
  let r, g, b;
  if (H < 120) { const [x, y] = sec(H); b = x; r = y; g = 3 * I - (r + b); }
  else if (H < 240) { const [x, y] = sec(H - 120); r = x; g = y; b = 3 * I - (r + g); }
  else { const [x, y] = sec(H - 240); g = x; b = y; r = 3 * I - (g + b); }
  return [r, g, b];
}
export function rgb2hsv(r, g, b) { const V = Math.max(r, g, b), m = Math.min(r, g, b), d = V - m; let H = 0; if (d > 0) H = V === r ? 60 * ((g - b) / d) : V === g ? 60 * ((b - r) / d) + 120 : 60 * ((r - g) / d) + 240; return { H: (H + 360) % 360, S: V > 0 ? d / V : 0, V }; }
export function rgb2lab(r, g, b) {
  const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4), [R, G, B] = [r, g, b].map(lin);
  const X = 0.4124 * R + 0.3576 * G + 0.1805 * B, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = 0.0193 * R + 0.1192 * G + 0.9505 * B;
  const h = (q) => (q > 0.008856 ? Math.cbrt(q) : 7.787 * q + 16 / 116), fx = h(X / 0.95047), fy = h(Y / 1), fz = h(Z / 1.08883);
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}
const ab = (r, g, b) => [r - (g + b) / 2, (Math.sqrt(3) / 2) * (g - b)];
export default {
  params: [
    { id: "R", label: "R", min: 0, max: 255, value: 204, int: true },
    { id: "G", label: "G", min: 0, max: 255, value: 102, int: true },
    { id: "B", label: "B", min: 0, max: 255, value: 51, int: true },
    { id: "dh", label: "HSI'da ton döndürme ΔH (°)", min: -180, max: 180, step: 5, value: 120 },
    { id: "sk", label: "Doygunluk çarpanı", min: 0, max: 2, step: 0.05, value: 1 },
    { id: "ik", label: "Yoğunluk çarpanı", min: 0, max: 2, step: 0.05, value: 1 },
  ],
  run(q) {
    const [r, g, b] = [q.R, q.G, q.B].map((v) => v / 255), hsi = rgb2hsi(r, g, b), hsv = rgb2hsv(r, g, b), lab = rgb2lab(r, g, b);
    const C = 1 - r, M = 1 - g, Yc = 1 - b, K = Math.min(C, M, Yc), cm = K >= 1 ? [0, 0, 0] : [C, M, Yc].map((v) => (v - K) / (1 - K));
    const Y = 0.299 * q.R + 0.587 * q.G + 0.114 * q.B, Cr = 0.713 * (q.R - Y) + 128, Cb = 0.564 * (q.B - Y) + 128;
    const raw = hsi2rgb(hsi.H + q.dh, Math.min(1, hsi.S * q.sk), hsi.I * q.ik), out = raw.map((v) => Math.max(0, Math.min(1, v))), clipped = raw.some((v) => v < -1e-6 || v > 1 + 1e-6);
    const back = hsi2rgb(hsi.H, hsi.S, hsi.I), rt = Math.max(...back.map((v, i) => Math.abs(v - [r, g, b][i]))) * 255;
    const gray = q.R === q.G && q.G === q.B;
    const P = [`HSI: H = ${fmt(hsi.H, 4)}°, S = ${fmt(hsi.S, 4)}, I = ${fmt(hsi.I, 4)}; gidiş-dönüş hatası ${fmt(rt, 2)} gri seviye.`];
    if (gray) P.push("<b>Gri renk:</b> doygunluk 0 ve ton tanımsız. Gonzalez formülü pay 0, payda ε olduğu için θ = arccos 0 = 90° verir: bu değer anlamsızdır ve ton döndürme rengi değiştirmez.");
    P.push(`HSV'de ton aynı yönde ama farklı bir sayı olabilir (HSV ${fmt(hsv.H, 4)}°): HSI tonu yoğunluk eksenine dik düzlemdeki açıdır, HSV tonu altıgen koninin kenar parçalarından doğrusal hesaplanır; ikisi ana ve ara renklerde (0°, 60°, …) çakışır.`);
    P.push(`Düzenleme sonrası RGB = (${out.map((v) => Math.round(255 * v)).join(", ")})${clipped ? ": <b>RGB küpünün dışına taştı ve kırpıldı</b> (yüksek doygunluk ya da yoğunlukta HSI'nın her noktası küpte değildir)" : ""}.`);
    const hex = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]].map((c) => ab(...c));
    const p0 = ab(r, g, b), p1 = ab(...out);
    return {
      plots: [
        { title: "Özgün | düzenlenmiş", img: { w: 2, h: 1, rgb: true, data: [r, g, b, ...out] }, side: true, size: { w: 300, h: 170 }, series: [] },
        { title: "Yoğunluk eksenine dik düzlem (α, β)", xlabel: "α = R − (G + B)/2", ylabel: "β", side: true, size: { w: 300, h: 290 }, series: [{ name: "altıgen (S = 1 sınırı, I'ya göre ölçeklenir)", x: hex.map((p) => p[0]), y: hex.map((p) => p[1]), c: 4 }, { name: "özgün", x: [0, p0[0]], y: [0, p0[1]], c: 1 }, { name: "düzenlenmiş", x: [0, p1[0]], y: [0, p1[1]], c: 0, dash: true }], xlim: [-1.1, 1.1], ylim: [-1, 1] },
      ],
      stats: [
        { label: "HSI (H°, S, I)", value: `${fmt(hsi.H, 4)}, ${fmt(hsi.S, 4)}, ${fmt(hsi.I, 4)}` },
        { label: "HSV (H°, S, V)", value: `${fmt(hsv.H, 4)}, ${fmt(hsv.S, 4)}, ${fmt(hsv.V, 4)}` },
        { label: "CMY / CMYK (C′, M′, Y′, K)", value: `${[C, M, Yc].map((v) => fmt(v, 3)).join(", ")} / ${cm.map((v) => fmt(v, 3)).join(", ")}, ${fmt(K, 3)}` },
        { label: "YCbCr (Y, Cb, Cr)", value: `${fmt(Y, 4)}, ${fmt(Cb, 4)}, ${fmt(Cr, 4)}` },
        { label: "CIE L*a*b*", value: `${fmt(lab.L, 4)}, ${fmt(lab.a, 4)}, ${fmt(lab.b, 4)}` },
        { label: "Düzenlenmiş RGB", value: out.map((v) => Math.round(255 * v)).join(", ") },
      ],
      note: P.join(" "),
    };
  },
};
