// M11.12 — 2-B güven bölgesi yöntemleri. Model m(p) = f + gᵀp + ½pᵀHp, bölge ‖p‖ ≤ δ. Alt problem çözümleri: Cauchy noktası (gradyan
// yönünde modelin en küçüğü, bölgeyle kırpılmış), dogleg (H pozitif tanımlıyken Cauchy → Newton kırık çizgisi, değilse Cauchy), tam çözüm
// (H + λI ⪰ 0 ve ‖p(λ)‖ = δ koşulunu sağlayan λ ikiye bölmeyle; belirsiz H'de de global en küçük). Oran ρ = gerçek azalma / öngörülen
// azalma: ρ < 0.25 → δ ← δ/4; ρ > 0.75 ve sınırdaysa δ ← 2δ (en çok 2); ρ > 0.1 ise adım kabul edilir. Karşılaştırma: sabit adımlı
// gradyan inişi ve korumasız Newton (x ← x − H⁻¹g). Fonksiyonlar: Rosenbrock (1 − x)² + 10(y − x²)², eyerli x² − y² + y⁴/4 (minimum
// (0, ±√2)), ve iki çukurlu (x² − 1)² + y².
import { fmt, range } from "./plot.js";

const FN = [
  { n: "Rosenbrock", f: ([x, y]) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, g: ([x, y]) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], H: ([x, y]) => [[2 - 40 * (y - x * x) + 80 * x * x, -40 * x], [-40 * x, 20]], box: [-2, 2, -1, 3], fmin: 0 },
  { n: "eyer noktalı: x² − y² + y⁴/4", f: ([x, y]) => x * x - y * y + y ** 4 / 4, g: ([x, y]) => [2 * x, -2 * y + y ** 3], H: ([x, y]) => [[2, 0], [0, -2 + 3 * y * y]], box: [-2, 2, -2.5, 2.5], fmin: -1 },
  { n: "iki çukur: (x² − 1)² + y²", f: ([x, y]) => (x * x - 1) ** 2 + y * y, g: ([x, y]) => [4 * x * (x * x - 1), 2 * y], H: ([x, y]) => [[12 * x * x - 4, 0], [0, 2]], box: [-2, 2, -1.5, 1.5], fmin: 0 },
];
const STARTS = [[-1.2, 1], [0.6, 0.02], [0.05, 1.2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1], nrm = (a) => Math.hypot(a[0], a[1]);
const Hv = (H, v) => [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]];
const solve2 = (H, b) => { const d = H[0][0] * H[1][1] - H[0][1] * H[1][0]; return [(H[1][1] * b[0] - H[0][1] * b[1]) / d, (-H[1][0] * b[0] + H[0][0] * b[1]) / d]; };
const eig2 = (H) => { const a = H[0][0], b = H[0][1], c = H[1][1], t = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b); return [t - d, t + d]; };
function cauchy(g, H, d) { const gHg = dot(g, Hv(H, g)), gn = nrm(g); const tau = gHg <= 0 ? 1 : Math.min(1, gn ** 3 / (d * gHg)); return [(-tau * d * g[0]) / gn, (-tau * d * g[1]) / gn]; }
function dogleg(g, H, d) { const [l1] = eig2(H); if (l1 <= 0) return cauchy(g, H, d); const pN = solve2(H, [-g[0], -g[1]]); if (nrm(pN) <= d) return pN; const gHg = dot(g, Hv(H, g)), pU = [(-dot(g, g) / gHg) * g[0], (-dot(g, g) / gHg) * g[1]]; if (nrm(pU) >= d) return [(d * pU[0]) / nrm(pU), (d * pU[1]) / nrm(pU)]; const D = [pN[0] - pU[0], pN[1] - pU[1]], a = dot(D, D), b = 2 * dot(pU, D), c = dot(pU, pU) - d * d, t = (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a); return [pU[0] + t * D[0], pU[1] + t * D[1]]; }
function exact(g, H, d) {
  const [l1] = eig2(H), p = (lam) => solve2([[H[0][0] + lam, H[0][1]], [H[1][0], H[1][1] + lam]], [-g[0], -g[1]]);
  if (l1 > 0) { const pn = p(0); if (nrm(pn) <= d) return pn; }
  let lo = Math.max(0, -l1) + 1e-12, hi = lo + 1; while (nrm(p(hi)) > d) hi *= 2;
  if (nrm(p(lo)) < d) { // "zor durum": gradyanın en küçük özdeğer yönünde bileşeni yok; o yönde bölge sınırına kadar git
    const pl = p(lo), a = H[0][0], b = H[0][1], c = H[1][1], v = Math.abs(b) > 1e-12 ? [l1 - c, b] : a <= c ? [1, 0] : [0, 1], vn = nrm(v), u = [v[0] / vn, v[1] / vn], t = Math.sqrt(Math.max(0, d * d - dot(pl, pl)));
    return [pl[0] + t * u[0], pl[1] + t * u[1]];
  }
  for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (nrm(p(m)) > d) lo = m; else hi = m; }
  return p(hi);
}
export function run(fn, method, x0, d0, N, eta) {
  const F = FN[fn]; let x = x0.slice(), d = d0; const path = [x.slice()], ds = [], rhos = [], fs = [F.f(x)]; let rej = 0;
  for (let k = 0; k < N; k++) {
    const g = F.g(x), H = F.H(x); if (nrm(g) < 1e-9) break;
    if (method === 0) { x = [x[0] - eta * g[0], x[1] - eta * g[1]]; }
    else if (method === 1) { const s = solve2(H, [-g[0], -g[1]]); x = [x[0] + s[0], x[1] + s[1]]; }
    else {
      const p = method === 2 ? cauchy(g, H, d) : method === 3 ? dogleg(g, H, d) : exact(g, H, d);
      const pred = -(dot(g, p) + 0.5 * dot(p, Hv(H, p))), xn = [x[0] + p[0], x[1] + p[1]], act = F.f(x) - F.f(xn), rho = pred > 0 ? act / pred : -1;
      if (pred < 1e-13) break; // öngörülen azalma kayan nokta düzeyinde: yakınsadı
      rhos.push(rho); ds.push(d);
      if (rho < 0.25) d /= 4; else if (rho > 0.75 && Math.abs(nrm(p) - d) < 1e-9) d = Math.min(2 * d, 2);
      if (rho > 0.1) x = xn; else rej++;
    }
    if (!x.every(Number.isFinite) || Math.abs(x[0]) > 1e6) break;
    path.push(x.slice()); fs.push(F.f(x));
  }
  return { path, ds, rhos, fs, rej };
}
const MN = ["gradyan inişi (sabit adım)", "Newton (korumasız)", "güven bölgesi: Cauchy noktası", "güven bölgesi: dogleg", "güven bölgesi: tam alt problem"];
export default {
  params: [
    { id: "fn", label: "Fonksiyon", options: FN.map((f) => f.n), value: 0 },
    { id: "m", label: "Yöntem", options: MN, value: 3 },
    { id: "st", label: "Başlangıç", options: ["(−1.2, 1)", "(0.6, 0.02)", "(0.05, 1.2)"], value: 0 },
    { id: "d0", label: "İlk yarıçap δ₀", min: 0.05, max: 2, step: 0.05, value: 0.5, on: { m: [2, 3, 4] } },
    { id: "eta", label: "Adım η", min: 0.001, max: 0.1, step: 0.001, value: 0.02, on: { m: [0] } },
    { id: "N", label: "Yineleme", min: 5, max: 300, step: 5, value: 60, int: true },
  ],
  run(q) {
    const F = FN[q.fn], o = run(q.fn, q.m, STARTS[q.st], q.d0, q.N, q.eta), last = o.path[o.path.length - 1], fl = F.f(last);
    const [a, b, c, e] = F.box, nx = 70, ny = 70, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) z.push(F.f([a + ((i + 0.5) * (b - a)) / nx, c + ((j + 0.5) * (e - c)) / ny]));
    const kHit = o.fs.findIndex((v) => v - F.fmin < 1e-6);
    const P = [`${MN[q.m]}, ${F.n}: ${o.path.length - 1} adım sonra (${fmt(last[0], 4)}, ${fmt(last[1], 4)}), f = ${fmt(fl, 4)}; en küçük değer ${fmt(F.fmin, 3)}. ${kHit >= 0 ? `f − f* < 10⁻⁶ ilk kez ${kHit}. adımda.` : "Bu adım sayısında 10⁻⁶ yakınlığa ulaşılmadı."}`];
    if (q.m >= 2) P.push(`Reddedilen adım: ${o.rej}. Oran ρ (gerçek/öngörülen azalma) ortalaması ${fmt(o.rhos.reduce((x, y) => x + y, 0) / Math.max(1, o.rhos.length), 3)}; son yarıçap ${fmt(o.ds[o.ds.length - 1] ?? q.d0, 3)}.`);
    const [l1] = eig2(F.H(STARTS[q.st]));
    if (l1 < 0) P.push(`Başlangıçta Hessian belirsiz (en küçük özdeğer ${fmt(l1, 3)}): Newton adımı bir eyer noktasına ya da en büyüğe çekilebilir; güven bölgesinin tam çözümü negatif eğrilik yönünü kullanır.`);
    if (q.m === 1 && fl > F.fmin + 1e-3) P.push("Korumasız Newton bu başlangıçta en küçüğe gitmiyor: model kötüyken tam adım atıyor ya da eyer noktasına yakınsıyor.");
    return {
      plots: [
        { title: `${F.n}: yol (koyu bölge = küçük f)`, xlabel: "x", ylabel: "y", side: true, size: { w: 320, h: 300 }, xlim: [a, b], ylim: [c, e], field: { x0: a, x1: b, y0: c, y1: e, nx, ny, z, log: q.fn === 0 },
          series: [{ name: "yol", x: o.path.map((p) => p[0]), y: o.path.map((p) => p[1]), c: 2, dots: true }] },
        { title: "f − f* (log)", xlabel: "yineleme", ylabel: "f − f*", side: true, size: { w: 290, h: 300 }, ylog: true, series: [{ name: "f − f*", x: range(o.fs.length), y: o.fs.map((v) => Math.max(v - F.fmin, 1e-14)), c: 0 }, ...(q.m >= 2 ? [{ name: "δ", x: range(o.ds.length), y: o.ds, c: 4, dash: true }] : [])] },
      ],
      stats: [{ label: "Son f", value: fmt(fl, 4) }, { label: "10⁻⁶'ya ulaşma adımı", value: kHit >= 0 ? String(kHit) : "—" }, { label: "Reddedilen adım", value: q.m >= 2 ? String(o.rej) : "—" }],
      note: P.join(" "),
    };
  },
};
