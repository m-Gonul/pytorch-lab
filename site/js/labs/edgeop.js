// M9.6 — Gradyan operatörleri: Roberts, Prewitt, Sobel, Kirsch (8 pusula çekirdeğinin en büyüğü). Büyüklük L2 ya da |g_x| + |g_y|.
// İki ölçüm: (1) aynı test deseninde eşiklenmiş kenar haritası ve F1 (cannylab'ın deseni ve ölçütü); (2) yön yanıtı: θ açılı,
// σ = 0.8 ile yumuşatılmış bir basamakta dört alt-piksel fazın ortalaması, 0°'ye göre normalize.
import { fmt, range } from "./plot.js";
import { W, gblur, noisy, truth, score } from "./cannylab.js";

const SOB = [1, 2, 1], PRE = [1, 1, 1];
const RING = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]];
// Kirsch: halkada ardışık üç 5, kalan beş −3; 8 dönüşü.
const KIR = range(8).map((k) => RING.map((_, i) => ((i - k + 8) % 8 < 3 ? 5 : -3)));
export function grad(a, n, op, L1) {
  const at = (x, y) => a[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))], M = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let gx = 0, gy = 0;
    if (op === 0) { gx = at(x + 1, y + 1) - at(x, y); gy = at(x, y + 1) - at(x + 1, y); }
    else if (op === 1 || op === 2) { const w = op === 2 ? SOB : PRE, s = op === 2 ? 4 : 3; for (let j = -1; j <= 1; j++) { gx += w[j + 1] * (at(x + 1, y + j) - at(x - 1, y + j)); gy += w[j + 1] * (at(x + j, y + 1) - at(x + j, y - 1)); } gx /= 2 * s; gy /= 2 * s; }
    else { let best = 0; for (const K of KIR) { let v = 0; RING.forEach(([dx, dy], i) => { v += K[i] * at(x + dx, y + dy); }); best = Math.max(best, v); } M[y * n + x] = best / 15; continue; }
    M[y * n + x] = L1 ? Math.abs(gx) + Math.abs(gy) : Math.hypot(gx, gy);
  }
  return M;
}
const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
// θ normalli, σ = 0.8 ile yumuşatılmış ideal basamak; dört alt-piksel faz kaymasında merkez yakınındaki en büyük yanıtın ortalaması.
export function orient(op, L1) {
  const n = 21, c = 10;
  return range(37).map((k) => {
    const th = (k * 5 * Math.PI) / 180, co = Math.cos(th), si = Math.sin(th);
    let acc = 0;
    for (const ph of [0, 0.25, 0.5, 0.75]) {
      const a = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) a[y * n + x] = 0.5 * (1 + erf(((x - c) * co + (y - c) * si - ph) / (0.8 * Math.SQRT2)));
      const M = grad(a, n, op, L1); let m = 0;
      for (let y = c - 3; y <= c + 3; y++) for (let x = c - 3; x <= c + 3; x++) m = Math.max(m, M[y * n + x]);
      acc += m / 4;
    }
    return acc;
  });
}
const NAMES = ["Roberts (2 × 2)", "Prewitt", "Sobel", "Kirsch (8 yön, en büyük)"];
export default {
  params: [
    { id: "op", label: "Operatör", options: NAMES, value: 2 },
    { id: "L1", label: "Büyüklük", options: ["√(g_x² + g_y²)", "|g_x| + |g_y|"], value: 0, on: { op: [0, 1, 2] } },
    { id: "s", label: "Önce Gauss σ (0 = yok)", min: 0, max: 3, step: 0.1, value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 40, step: 1, value: 10 },
    { id: "t", label: "Eşik (max büyüklüğün %'si)", min: 1, max: 80, step: 1, value: 20 },
  ],
  seed: true,
  run(q, r) {
    const L1 = q.op < 3 && q.L1 === 1, { clean, g } = noisy(r, q.noise), fs = gblur(g, q.s), M = grad(fs, W, q.op, L1), mx = Math.max(...M);
    const e = Uint8Array.from(M, (v) => (v >= (q.t / 100) * mx ? 1 : 0)), sc = score(e, truth(clean));
    const o = orient(q.op, L1), o0 = o[0], on = o.map((v) => v / o0), hi = Math.max(...on), lo = Math.min(...on), ang = range(37).map((k) => k * 5);
    const P = [`${NAMES[q.op]}${q.op < 3 ? (L1 ? ", |g_x| + |g_y|" : ", L2") : ""}: yön yanıtı 0°'ye göre ${fmt(lo, 3)}–${fmt(hi, 3)} arasında (en çok ${ang[on.indexOf(hi)]}°'de).`];
    if (L1) P.push(q.op === 0 ? "Roberts'ın bileşenleri köşegen yönlerdedir; bu yüzden L1 yaklaşımı tersine eksen yönlü kenarları abartır: 45°'deki yanıt 0°'dekinin ≈ 1/√2'si." : "L1 yaklaşımı köşegen kenarları abartır: 45°'de iki bileşen birlikte büyük olduğu için yanıt ≈ √2 kat.");
    else if (q.op === 0) P.push("Roberts 2 × 2 köşegen farklarla çalışır; yarım piksel kaymış bir noktada ölçer ve çok küçük olduğu için gürültüye en duyarlı olanıdır.");
    else if (q.op === 3) P.push("Kirsch tek bir büyüklük değil, 8 pusula çekirdeğinin en büyük yanıtını verir; hangi çekirdeğin kazandığı yönü 45°'lik adımlarla söyler. Yanıt yöne göre %10'a kadar dalgalanır.");
    else P.push(q.op === 2 ? "Sobel'in merkez ağırlığı 2, yön yanıtını Prewitt'ten daha düzgün yapar ve biraz daha yumuşatır." : "Prewitt ağırlıksızdır: köşegenlerde Sobel'den biraz daha fazla sapar.");
    P.push(`Eşiklenmiş haritada kesinlik ${fmt(sc.P, 3)}, duyarlılık ${fmt(sc.R, 3)}, F1 ${fmt(sc.F, 3)}; ${sc.n} piksel (gerçek ${sc.nt}): eşiklenmiş gradyan kenarları kalın ve kopuktur, ince ve bağlı kenar için NMS ve histerezis gerekir (l2).`);
    return {
      plots: [
        { title: "Gradyan büyüklüğü", img: { w: W, h: W, data: Array.from(M, (v) => Math.min(1, v / (0.6 * mx))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Eşiklenmiş kenar haritası", img: { w: W, h: W, data: Array.from(e) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Yön yanıtı (ideal basamak, 0°'ye göre)", xlabel: "kenar normalinin açısı θ (°)", ylabel: "göreli yanıt", wide: true, size: { w: 620, h: 180 }, series: [{ name: NAMES[q.op], x: ang, y: on, c: 0 }, { name: "izotropik", x: [0, 180], y: [1, 1], c: 4, dash: true }], ylim: [0.6, 1.6], marks: [{ x: 45, label: "45°" }, { x: 90, label: "90°" }, { x: 135, label: "135°" }] },
      ],
      stats: [{ label: "Yön yanıtı (min–max)", value: `${fmt(lo, 3)}–${fmt(hi, 3)}` }, { label: "F1 (1 px tolerans)", value: sc.F }, { label: "Kenar pikseli (gerçek)", value: `${sc.n} (${sc.nt})` }, { label: "Kesinlik / duyarlılık", value: `${fmt(sc.P, 3)} / ${fmt(sc.R, 3)}` }],
      note: P.join(" "),
    };
  },
};
