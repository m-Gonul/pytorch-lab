// M9.10 — Frekans uzayında süzgeçleme, Gonzalez'in hattıyla: 64 × 64 görüntüyü (isteğe bağlı) 128 × 128'e sıfırla doldur,
// merkezle, DFT al, H(u, v) ile çarp, ters DFT, kırp. İdeal, Butterworth (n) ve Gauss; alçak ya da yüksek geçiren.
// Ölçütler: korunan güç oranı α, aşma (halkalanma) ve dolgusuz durumda sarmalanmanın kenara etkisi.
import { fmt, range } from "./plot.js";
import { fft2 } from "./circconv.js";

const M = 64;
export function scene() {
  const f = new Float64Array(M * M);
  for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) {
    let v = 0.15;
    if (x >= 8 && x < 30 && y >= 8 && y < 30) v = 0.85;
    if ((x - 46) ** 2 + (y - 18) ** 2 <= 81) v = 0.6;
    if (y >= 38 && y < 58 && x >= 8 && x < 56) v = 0.4 + 0.3 * (Math.floor(x / 3) % 2);
    if (x >= 60) v = 0.95;
    f[y * M + x] = v;
  }
  return f;
}
export function H(type, hp, D, D0, n) {
  let h = type === 0 ? (D <= D0 ? 1 : 0) : type === 1 ? 1 / (1 + (D / D0) ** (2 * n)) : Math.exp(-(D * D) / (2 * D0 * D0));
  return hp ? 1 - h : h;
}
const TYPES = ["ideal", "Butterworth", "Gauss"];
export default {
  params: [
    { id: "type", label: "Süzgeç", options: TYPES, value: 0 },
    { id: "hp", label: "Tür", options: ["alçak geçiren", "yüksek geçiren"], value: 0 },
    { id: "D0", label: "Kesim D₀ (dolgulu ızgarada frekans örneği)", min: 2, max: 60, step: 1, value: 20 },
    { id: "n", label: "Butterworth derecesi n", min: 1, max: 10, value: 2, int: true, on: { type: [1] } },
    { id: "pad", label: "Dolgu", options: ["yok (64 × 64)", "sıfır dolgu (128 × 128)"], value: 1 },
  ],
  run(q) {
    const f = scene(), P = q.pad ? 2 * M : M, D0 = q.pad ? q.D0 : q.D0 / 2;
    const re = new Float64Array(P * P);
    for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) re[y * P + x] = f[y * M + x] * ((x + y) % 2 ? -1 : 1);
    const F = fft2(re, new Float64Array(P * P), P), c = P / 2;
    let pin = 0, ptot = 0;
    const Hm = new Float64Array(P * P);
    for (let v = 0; v < P; v++) for (let u = 0; u < P; u++) { const D = Math.hypot(u - c, v - c), h = H(q.type, q.hp, D, D0, q.n), i = v * P + u, pw = F.re[i] ** 2 + F.im[i] ** 2; Hm[i] = h; ptot += pw; pin += pw * h * h; F.re[i] *= h; F.im[i] *= h; }
    const G = fft2(F.re, F.im, P, true), g = new Float64Array(M * M);
    for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) g[y * M + x] = G.re[y * P + x] * ((x + y) % 2 ? -1 : 1);
    // Halkalanma: büyük karenin içinde kenardan 2–6 piksel uzaktaki bantta en büyük sapma (AP'de), ya da düz arka plandaki salınım (YG'de).
    let over = 0; for (let y = 10; y < 28; y++) for (let x = 10; x < 28; x++) if (Math.min(x - 8, 29 - x, y - 8, 29 - y) >= 2) over = Math.max(over, Math.abs(g[y * M + x] - (q.hp ? 0 : 0.85)));
    // Sol kenar (x = 0, 1; zemin 0.15): dolgusuzda sağdaki parlak şerit sarmalanıp aydınlatır, sıfır dolguda sıfırlar karartır.
    let edge = 0; for (let y = 32; y < 38; y++) for (let x = 0; x < 2; x++) edge += (g[y * M + x] - (q.hp ? 0 : 0.15)) / 12;
    const disp = (a) => { if (!q.hp) return Array.from(a, (v) => Math.max(0, Math.min(1, v))); const m = Math.max(...a.map(Math.abs)) || 1; return Array.from(a, (v) => 0.5 + 0.5 * v / m); };
    const spec = new Float64Array(P * P), Fo = fft2(re, new Float64Array(P * P), P); let mx = 0;
    for (let i = 0; i < P * P; i++) { spec[i] = Math.log(1 + Math.hypot(Fo.re[i], Fo.im[i])); mx = Math.max(mx, spec[i]); }
    const P2 = [`Korunan güç oranı α = ${fmt(100 * pin / ptot, 3)}%. Karenin içinde (kenardan ≥ 2 piksel) en büyük sapma ${fmt(over, 3)}; sol kenarda ortalama sapma ${fmt(edge, 3)} (pozitif: aydınlanma, negatif: kararma).`];
    P2.push(q.type === 0 ? "İdeal süzgecin frekanstaki keskin kesimi uzayda sinc biçimli, yavaş sönen bir dürtü yanıtıdır: kenarların iki yanında halkalar (Gibbs) belirir." : q.type === 1 ? `Butterworth n = ${q.n}: n küçükken geçiş yumuşak, halkalanma yok denecek kadar az; n büyüdükçe ideale yaklaşır ve halkalanma geri gelir.` : "Gauss süzgecin dürtü yanıtı da Gauss'tur: negatif lob yok, halkalanma olmaz; aynı D₀'da ideal süzgeçten daha çok bulanıklaştırır.");
    if (!q.pad) P2.push("Dolgu yok: DFT görüntüyü periyodik sayar; sağdaki parlak şerit sol kenara sızar ve sol kenar aydınlanır (sarmalanma). Sıfır dolgu bunu önler ama kenar ötesini 0 sayar: sol kenar bu kez biraz kararır; Gonzalez bunu dolgunun bilinen yan etkisi olarak not eder (yansıtmalı dolgu azaltır).");
    if (q.hp) P2.push("Yüksek geçiren çıkış 0 ortalamalı (DC kesildi); gösterim için 0.5 gri merkezli ölçeklendi.");
    return {
      plots: [
        { title: `Çıkış: ${TYPES[q.type]} ${q.hp ? "YG" : "AG"}, D₀ = ${q.D0}`, img: { w: M, h: M, data: disp(g) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `log(1 + |F|) ve süzgeç (${P} × ${P}, merkezli)`, img: { w: P, h: P, data: Array.from(spec, (v, i) => (v / mx) * (0.35 + 0.65 * Hm[i])) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Süzgecin kesiti H(D)", xlabel: "D (merkezden uzaklık)", ylabel: "H", wide: true, size: { w: 620, h: 150 }, series: TYPES.map((t, k) => ({ name: t, x: range(P / 2 + 1), y: range(P / 2 + 1).map((D) => H(k, q.hp, D, D0, q.n)), c: k === q.type ? 0 : 4, dash: k !== q.type })), marks: [{ x: D0, label: "D₀" }] },
      ],
      stats: [{ label: "Korunan güç α", value: `${fmt(100 * pin / ptot, 3)}%` }, { label: "Karede en büyük sapma (halka)", value: over }, { label: "Sol kenarda ortalama sapma", value: edge }],
      note: P2.join(" "),
    };
  },
};
