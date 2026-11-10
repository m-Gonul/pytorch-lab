// M9.17 — Video stabilizasyonunda istenmeyen hareketin belirlenmesi: 1-B kamera yörüngesi (yatay konum, piksel) 300 kare. Yörünge =
// kasıtlı hareket (durgun → kaydırma (pan) → durgun) + düşük frekanslı sallanma (yürüme, periyot 40 kare) + yüksek frekanslı titreme
// (el, AR(1) gürültü). Yumuşatma yöntemleri: Gauss süzgeci (çevrimdışı, simetrik ya da çevrimiçi, yalnız geçmiş kareler), Kalman süzgeci
// (sabit hız modeli, çevrimiçi), L2 eniyileme (çevrimdışı: Σ(p − c)² + λ Σ(Δ²p)², kapalı çözüm). Ölçütler: kalan titreme (ikinci farkın
// std'si), gereken kırpma (|p − c|'nin en büyüğü: her kenardan kesilecek piksel), kaydırmadaki gecikme.
import { fmt, range } from "./plot.js";

const T = 300;
function path(r, jit, bounce) {
  const pan = (t) => (t < 100 ? 0 : t < 180 ? (300 * (t - 100)) / 80 : 300);
  let e = 0; const c = [], intent = [];
  for (let t = 0; t < T; t++) { e = 0.6 * e + jit * r.normal(); intent.push(pan(t)); c.push(pan(t) + bounce * Math.sin((2 * Math.PI * t) / 40) + e); }
  return { c, intent };
}
function gauss(c, s, causal) {
  const R = Math.ceil(3 * s), w = range(2 * R + 1).map((k) => Math.exp(-((k - R) ** 2) / (2 * s * s)));
  return c.map((_, t) => { let a = 0, z = 0; for (let k = -R; k <= (causal ? 0 : R); k++) { const i = Math.max(0, Math.min(T - 1, t + k)); a += w[k + R] * c[i]; z += w[k + R]; } return a / z; });
}
function kalman(c, q, rr) {
  let x = [c[0], 0], P = [[100, 0], [0, 100]]; const out = [];
  for (const z of c) {
    x = [x[0] + x[1], x[1]]; P = [[P[0][0] + P[0][1] + P[1][0] + P[1][1] + q / 4, P[0][1] + P[1][1] + q / 2], [P[1][0] + P[1][1] + q / 2, P[1][1] + q]];
    const S = P[0][0] + rr, K = [P[0][0] / S, P[1][0] / S], y = z - x[0];
    x = [x[0] + K[0] * y, x[1] + K[1] * y]; P = [[(1 - K[0]) * P[0][0], (1 - K[0]) * P[0][1]], [P[1][0] - K[1] * P[0][0], P[1][1] - K[1] * P[0][1]]];
    out.push(x[0]);
  }
  return out;
}
// (I + λ DᵀD) p = c, D ikinci fark; beş köşegenli sistem, Gauss eleme (bant).
function l2opt(c, lam) {
  const A = range(T).map(() => new Float64Array(T)), b = Float64Array.from(c);
  for (let i = 0; i < T; i++) A[i][i] = 1;
  for (let k = 1; k < T - 1; k++) { const idx = [k - 1, k, k + 1], co = [1, -2, 1]; for (let a = 0; a < 3; a++) for (let d = 0; d < 3; d++) A[idx[a]][idx[d]] += lam * co[a] * co[d]; }
  for (let i = 0; i < T; i++) { const p = A[i][i]; for (let j = i + 1; j < Math.min(T, i + 3); j++) { const f = A[j][i] / p; if (!f) continue; for (let k = i; k < Math.min(T, i + 3); k++) A[j][k] -= f * A[i][k]; b[j] -= f * b[i]; } }
  const x = new Float64Array(T); for (let i = T - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < Math.min(T, i + 3); k++) s -= A[i][k] * x[k]; x[i] = s / A[i][i]; }
  return Array.from(x);
}
const jitter = (p) => { const d = range(T - 2).map((t) => p[t + 2] - 2 * p[t + 1] + p[t]), m = d.reduce((a, b) => a + b, 0) / d.length; return Math.sqrt(d.reduce((a, b) => a + (b - m) ** 2, 0) / d.length); };
export default {
  params: [
    { id: "meth", label: "Yumuşatma", options: ["Gauss süzgeci", "Kalman (sabit hız)", "L2 eniyileme (Δ² cezası)"], value: 0 },
    { id: "sigma", label: "Gauss σ (kare)", min: 1, max: 60, value: 10, int: true, on: { meth: [0] } },
    { id: "causal", label: "Gauss: kareler", options: ["çevrimdışı (geçmiş + gelecek)", "çevrimiçi (yalnız geçmiş)"], value: 0, on: { meth: [0] } },
    { id: "q", label: "Kalman ivme varyansı q", min: 1e-4, max: 10, value: 0.01, log: true, on: { meth: [1] } },
    { id: "lam", label: "L2 eniyileme λ", min: 1, max: 1e6, value: 1000, log: true, on: { meth: [2] } },
    { id: "jit", label: "El titremesi (piksel)", min: 0, max: 10, step: 0.5, value: 4 },
    { id: "bounce", label: "Yürüme sallanması genliği (piksel)", min: 0, max: 30, value: 10, int: true },
  ],
  seed: true,
  run(q, r) {
    const { c, intent } = path(r, q.jit, q.bounce);
    const p = q.meth === 0 ? gauss(c, q.sigma, q.causal === 1) : q.meth === 1 ? kalman(c, q.q, 16) : l2opt(c, q.lam);
    const dev = c.map((v, t) => p[t] - v), crop = Math.max(...dev.map(Math.abs)), jIn = jitter(c), jOut = jitter(p);
    // Gecikme: kaydırma bölgesinde (t = 110..170) düzeltilmiş yolun kasıtlı yoldan ortalama geride kalması (piksel → kare: hız 3.75 px/kare)
    const lagPx = range(60).map((k) => intent[110 + k] - p[110 + k]).reduce((a, b) => a + b, 0) / 60, lag = lagPx / 3.75;
    const bres = Math.sqrt(range(T).map((t) => (p[t] - intent[t]) ** 2).reduce((a, b) => a + b, 0) / T);
    const names = ["Gauss", "Kalman", "L2 eniyileme"];
    const P = [`${names[q.meth]}: kalan titreme (ikinci fark std'si) ${fmt(jOut, 3)} piksel/kare² (girdi ${fmt(jIn, 3)}). Gereken kırpma her kenardan ${fmt(crop, 3)} piksel (640 piksellik genişliğin ${fmt((200 * crop) / 640, 3)}%'i). Kasıtlı yoldan RMS sapma ${fmt(bres, 3)} piksel; kaydırmada ortalama gecikme ${fmt(lag, 3)} kare (negatif: önde).`];
    if (q.meth === 0 && q.causal === 1) P.push("Çevrimiçi Gauss yalnız geçmişi gördüğü için kaydırmayı geriden izliyor: gecikme ve kırpma büyüyor.");
    if (q.meth === 0 && q.causal === 0 && q.sigma >= 30) P.push("Büyük σ sallanmayı da siliyor ama kaydırmanın başı ve sonu yuvarlanıyor; bu bölgelerde kırpma artıyor.");
    if (q.meth === 1) P.push("Kalman sabit hız varsayar ve yalnız geçmişi görür: q küçükse yol düzleşir ama kaydırmayı geriden izler ve hız değişimlerinde taşar; q büyükse ölçülen yola yapışır.");
    return {
      plots: [
        { title: "Kamera yörüngesi: ölçülen, yumuşatılmış ve kasıtlı", xlabel: "kare", ylabel: "x (piksel)", wide: true, size: { w: 620, h: 220 }, series: [{ name: "ölçülen", x: range(T), y: c, c: 4 }, { name: "yumuşatılmış", x: range(T), y: p, c: 0 }, { name: "kasıtlı", x: range(T), y: intent, c: 2, dash: true }] },
        { title: "Düzeltme p − c (kırpma bunu karşılamalı)", xlabel: "kare", ylabel: "piksel", wide: true, size: { w: 620, h: 150 }, series: [{ name: "p − c", x: range(T), y: dev, c: 1 }], marks: [] },
      ],
      stats: [{ label: "Kalan titreme / girdi", value: `${fmt(jOut, 3)} / ${fmt(jIn, 3)}` }, { label: "Gereken kırpma (her kenar)", value: `${fmt(crop, 3)} piksel` }, { label: "Kaydırmada gecikme", value: `${fmt(lag, 3)} kare` }, { label: "Kasıtlı yoldan RMS sapma", value: `${fmt(bres, 3)} piksel` }],
      note: P.join(" "),
    };
  },
};
