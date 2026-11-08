// M2.8 — Gauss karışımı için EM, 2B: iterasyon iterasyon elipsler, sorumluluklar ve monoton artan log-olabilirlik.
import { fmt, range } from "./plot.js";

const LOG2PI = Math.log(2 * Math.PI);
function logN(x, m, S) { const a = S[0][0], b = S[0][1], d = S[1][1], det = a * d - b * b, dx = x[0] - m[0], dy = x[1] - m[1]; return -LOG2PI - 0.5 * Math.log(det) - (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / (2 * det); }
const lse = (v) => { const m = Math.max(...v); return m + Math.log(v.reduce((s, x) => s + Math.exp(x - m), 0)); };
export function data(kind, r) {
  const X = [];
  const g = (m, L, n) => { for (let i = 0; i < n; i++) { const z = [r.normal(), r.normal()]; X.push([m[0] + L[0][0] * z[0], m[1] + L[1][0] * z[0] + L[1][1] * z[1]]); } };
  if (kind === 0) { g([-2, -1], [[0.6, 0], [0, 0.6]], 100); g([2, -1], [[0.6, 0], [0, 0.6]], 100); g([0, 2], [[0.6, 0], [0, 0.6]], 100); }
  else if (kind === 1) { g([-1.5, 0], [[1.4, 0], [1.1, 0.35]], 150); g([1.8, 0.5], [[0.35, 0], [-0.25, 1.3]], 150); }
  else { g([0, 0], [[1.6, 0], [0, 1.6]], 220); g([1, 1], [[0.25, 0], [0, 0.25]], 80); }
  return X;
}
export function em(X, K, T, cov, r) {
  const n = X.length; let mu = range(K).map(() => [...X[Math.floor(r() * n)]]), S = range(K).map(() => [[1, 0], [0, 1]]), pi = range(K).map(() => 1 / K);
  const hist = [], LL = [];
  for (let t = 0; t <= T; t++) {
    const lp = X.map((x) => range(K).map((k) => Math.log(pi[k]) + logN(x, mu[k], S[k]))), ln = lp.map(lse);
    LL.push(ln.reduce((a, b) => a + b, 0) / n);
    const R = lp.map((row, i) => row.map((v) => Math.exp(v - ln[i])));
    hist.push({ mu: mu.map((m) => [...m]), S: S.map((s) => s.map((rw) => [...rw])), pi: [...pi], R });
    if (t === T) break;
    const Nk = range(K).map((k) => R.reduce((s, rw) => s + rw[k], 0) + 1e-9);
    pi = Nk.map((v) => v / n);
    mu = range(K).map((k) => [0, 1].map((j) => X.reduce((s, x, i) => s + R[i][k] * x[j], 0) / Nk[k]));
    S = range(K).map((k) => { let a = 0, b = 0, d = 0; X.forEach((x, i) => { const dx = x[0] - mu[k][0], dy = x[1] - mu[k][1], w = R[i][k]; a += w * dx * dx; b += w * dx * dy; d += w * dy * dy; }); a /= Nk[k]; b /= Nk[k]; d /= Nk[k];
      if (cov === 1) b = 0; if (cov === 2) { const v = (a + d) / 2; a = d = v; b = 0; }
      return [[a + 1e-4, b], [b, d + 1e-4]]; });
  }
  return { hist, LL };
}
function ellipse(m, S, sc) { const a = S[0][0], b = S[0][1], d = S[1][1], tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0)), l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-9), th = 0.5 * Math.atan2(2 * b, a - d); const xs = [], ys = []; for (let i = 0; i <= 60; i++) { const t = (2 * Math.PI * i) / 60, u = sc * Math.sqrt(l1) * Math.cos(t), v = sc * Math.sqrt(l2) * Math.sin(t); xs.push(m[0] + u * Math.cos(th) - v * Math.sin(th)); ys.push(m[1] + u * Math.sin(th) + v * Math.cos(th)); } return { xs, ys }; }

export default {
  params: [
    { id: "data", label: "Veri", options: ["3 küresel grup", "2 eğik, uzamış grup", "geniş + dar iç içe grup"], value: 1 },
    { id: "K", label: "Bileşen sayısı K", min: 1, max: 5, value: 2, int: true },
    { id: "cov", label: "Kovaryans türü", options: ["tam", "köşegen", "küresel (izotrop)"], value: 0 },
    { id: "t", label: "Gösterilen iterasyon", min: 0, max: 60, value: 5, int: true },
  ],
  seed: true,
  run(q, r) {
    const X = data(q.data, r), K = Math.round(q.K), T = 60, E = em(X, K, T, q.cov, r), h = E.hist[Math.min(Math.round(q.t), T)];
    const cols = [0, 2, 1, 3, 5];
    const hard = h.R.map((rw) => rw.indexOf(Math.max(...rw))), conf = h.R.map((rw) => Math.max(...rw));
    const unsure = X.filter((_, i) => conf[i] < 0.8);
    return {
      plots: [
        { title: `İterasyon ${Math.round(q.t)}: bileşenler (1σ ve 2σ elipsleri) ve en olası bileşene göre noktalar`, xlabel: "x₁", ylabel: "x₂", size: { w: 390, h: 380 }, xlim: [-5, 5], ylim: [-5, 5], series: [
          ...range(K).map((k) => ({ name: `bileşen ${k + 1} (π = ${fmt(h.pi[k], 2)})`, x: X.filter((_, i) => hard[i] === k).map((p) => p[0]), y: X.filter((_, i) => hard[i] === k).map((p) => p[1]), kind: "scatter", c: cols[k], r: 2.2 })),
          { name: "belirsiz (en büyük sorumluluk < 0.8)", x: unsure.map((p) => p[0]), y: unsure.map((p) => p[1]), kind: "scatter", c: 4, r: 4.5 },
          ...range(K).flatMap((k) => [1, 2].map((sc) => { const e = ellipse(h.mu[k], h.S[k], sc); return { name: "", x: e.xs, y: e.ys, c: cols[k], dash: sc === 2 }; })),
        ] },
        { title: "Ortalama log-olabilirlik ℓ/n, iterasyona göre", xlabel: "iterasyon", ylabel: "ℓ/n", series: [{ name: "ℓ/n", x: range(E.LL.length), y: E.LL, c: 0, dots: true }], marks: [{ x: Math.round(q.t), label: "gösterilen" }] },
      ],
      stats: [
        { label: "ℓ/n (gösterilen iterasyon)", value: E.LL[Math.round(q.t)] },
        { label: "ℓ/n (60. iterasyon)", value: E.LL[T] },
        { label: "Monoton mu? (en büyük düşüş)", value: (() => { let d = 0; for (let i = 1; i < E.LL.length; i++) d = Math.max(d, E.LL[i - 1] - E.LL[i]); return d < 1e-6 ? (d > 0 ? `evet (yalnız ${d.toExponential(0)} düzeyinde yuvarlama)` : "evet (düşüş yok)") : `hayır: düşüş ${d.toExponential(1)}`; })() },
        { label: "Belirsiz nokta sayısı (sorumluluk < 0.8)", value: unsure.length },
        { label: "Serbest parametre sayısı", value: K * (2 + [3, 2, 1][q.cov]) + (K - 1) },
      ],
      note: `E adımı: her nokta için sorumluluklar r_ik = P(bileşen k | x_i) — yumuşak atama (gri halkalı noktalar iki bileşen arasında kararsız). M adımı: her bileşen, sorumluluklarla ağırlıklandırılmış ortalama ve kovaryansa taşınır. Log-olabilirlik eğrisi <b>hiç düşmez</b> (girişteki kanıt). Sonuç başlatmaya bağlıdır: tohumu değiştir, bazen farklı bir yerel optimuma gider. ${q.cov === 2 ? "Küresel kovaryans + eşit varyans + sert atama limiti = k-means: eğik grupları ayıramaz. " : q.cov === 1 ? "Köşegen kovaryans eksene paralel elipsler çizer: eğik grupları kaba yaklaştırır. " : ""}${K > 3 ? "Fazla bileşen: bazıları bir grubu paylaşır ya da birkaç noktaya daralır (kovaryansa küçük bir taban eklenmiştir, yoksa olabilirlik sınırsız büyürdü)." : ""}`,
    };
  },
};
