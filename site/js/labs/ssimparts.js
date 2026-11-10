// M9.13 — SSIM'in üç bileşeni tek bir yamada (slaytlardaki tanımlar, yansız σ, K1 = 0.01, K2 = 0.03, L = 255, C3 = C2/2).
// y = μ_x + a·(B_σ x − μ_x) + d + gürültü: d parlaklığı, a kontrastı (negatifse yapıyı da ters çevirir), bulanıklık ve gürültü
// yapıyı değiştirir. Çıkış: l, c, s, SSIM = l·c·s, MSE, PSNR ve iki yamanın grafiği.
import { fmt, range } from "./plot.js";

const N = 48;
const base = range(N).map((i) => 120 + 40 * Math.sin(i / 3) + (i > 24 && i < 30 ? 45 : 0) + 15 * Math.cos(i / 1.3));
export function parts(x, y, K1 = 0.01, K2 = 0.03, L = 255) {
  const n = x.length, C1 = (K1 * L) ** 2, C2 = (K2 * L) ** 2, C3 = C2 / 2;
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  const vx = x.reduce((a, v) => a + (v - mx) ** 2, 0) / (n - 1), vy = y.reduce((a, v) => a + (v - my) ** 2, 0) / (n - 1), cxy = x.reduce((a, v, i) => a + (v - mx) * (y[i] - my), 0) / (n - 1);
  const sx = Math.sqrt(vx), sy = Math.sqrt(vy);
  const l = (2 * mx * my + C1) / (mx * mx + my * my + C1), c = (2 * sx * sy + C2) / (vx + vy + C2), s = (cxy + C3) / (sx * sy + C3);
  return { l, c, s, ssim: l * c * s, closed: ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2)) };
}
export default {
  params: [
    { id: "d", label: "Parlaklık kayması d", min: -80, max: 80, step: 1, value: 0 },
    { id: "a", label: "Kontrast çarpanı a (negatif: ters yapı)", min: -1.5, max: 2.5, step: 0.05, value: 1 },
    { id: "b", label: "Bulanıklık σ (örnek)", min: 0, max: 4, step: 0.1, value: 0 },
    { id: "n", label: "Gürültü σ", min: 0, max: 60, step: 1, value: 15 },
  ],
  seed: true,
  run(q, r) {
    const x = base, mx = x.reduce((a, b) => a + b, 0) / N;
    const bl = q.b <= 0 ? x.slice() : x.map((_, i) => { const R = Math.ceil(3 * q.b); let s = 0, w = 0; for (let k = -R; k <= R; k++) { const j = Math.min(N - 1, Math.max(0, i + k)), g = Math.exp(-0.5 * (k / q.b) ** 2); s += g * x[j]; w += g; } return s / w; });
    const y = bl.map((v) => mx + q.a * (v - mx) + q.d + q.n * r.normal());
    const P = parts(x, y), mse = x.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / N, psnr = mse < 1e-9 ? Infinity : 10 * Math.log10((255 * 255) / mse);
    const msg = [`l = ${fmt(P.l, 4)}, c = ${fmt(P.c, 4)}, s = ${fmt(P.s, 4)} → SSIM = ${fmt(P.ssim, 4)} (C₃ = C₂/2 ile kapalı biçim: ${fmt(P.closed, 4)}). MSE ${fmt(mse, 4)}, PSNR ${fmt(psnr, 4)} dB.`];
    if (q.a < 0) msg.push("a < 0: yama ters çevrildi; yapı terimi s negatif olur ve SSIM de negatife düşebilir. MSE ise yalnız büyüklüğü görür.");
    if (q.d !== 0 && q.b === 0 && q.n === 0 && q.a === 1) msg.push("Yalnız parlaklık değişti: c = s = 1, SSIM yalnız l ile azalır; MSE = d² olduğu hâlde yapı kusursuz korunuyor.");
    msg.push("Üç bileşen görece bağımsızdır: d yalnız l'yi, a'nın büyüklüğü yalnız c'yi değiştirir; gürültü ve bulanıklık hem c'yi hem s'yi etkiler.");
    return {
      plots: [
        { title: "Referans x ve bozulmuş y", xlabel: "i", ylabel: "değer", wide: true, size: { w: 620, h: 190 }, series: [{ name: "x", x: range(N), y: x, c: 4 }, { name: "y", x: range(N), y, c: 0 }] },
        { title: "SSIM bileşenleri", xlabel: "", ylabel: "değer", wide: true, size: { w: 620, h: 160 }, series: [{ name: "değer", x: range(4), y: [P.l, P.c, P.s, P.ssim], kind: "bar", c: 1 }], xticks: [[0, "l (parlaklık)"], [1, "c (kontrast)"], [2, "s (yapı)"], [3, "SSIM"]], ylim: [-1, 1] },
      ],
      stats: [{ label: "l, c, s", value: `${fmt(P.l, 4)}, ${fmt(P.c, 4)}, ${fmt(P.s, 4)}` }, { label: "SSIM", value: P.ssim }, { label: "MSE / PSNR", value: `${fmt(mse, 4)} / ${fmt(psnr, 4)} dB` }],
      note: msg.join(" "),
    };
  },
};
