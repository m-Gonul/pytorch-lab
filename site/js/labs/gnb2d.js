// M2.2 — Gaussian Naive Bayes, 2B: sınıf içi korelasyon (naif varsayımın ihlali), ortak/ayrı varyans → doğrusal/kuadratik sınır, kalibrasyon.
import { fmt, range } from "./plot.js";

export function sample(n, q, r) { // sınıf 1: merkez (+s/2, +s/2·0.3), sınıf 0: (−s/2, −…); sınıf içi kovaryans [[a², ρab], [ρab, b²]]
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const y = i % 2, sc = y ? q.vr : 1, z1 = r.normal(), z2 = r.normal();
    const e1 = sc * z1, e2 = sc * (q.rho * z1 + Math.sqrt(1 - q.rho * q.rho) * z2);
    X.push([(y ? 1 : -1) * q.sep / 2 + e1, (y ? 1 : -1) * q.sep * 0.15 + e2]); Y.push(y);
  }
  return { X, Y };
}
export function fitGNB(X, Y, shared) {
  const st = [0, 1].map((c) => { const P = X.filter((_, i) => Y[i] === c), n = P.length, m = [0, 1].map((j) => P.reduce((s, p) => s + p[j], 0) / n), v = [0, 1].map((j) => P.reduce((s, p) => s + (p[j] - m[j]) ** 2, 0) / n + 1e-9); return { n, m, v }; });
  if (shared) { const v = [0, 1].map((j) => (st[0].v[j] * st[0].n + st[1].v[j] * st[1].n) / (st[0].n + st[1].n)); st[0].v = v; st[1].v = [...v]; }
  const N = st[0].n + st[1].n;
  return { st, prior: [st[0].n / N, st[1].n / N] };
}
const lg = (x, m, v) => -0.5 * Math.log(2 * Math.PI * v) - ((x - m) ** 2) / (2 * v);
export const nbLogit = (M, x) => Math.log(M.prior[1]) + lg(x[0], M.st[1].m[0], M.st[1].v[0]) + lg(x[1], M.st[1].m[1], M.st[1].v[1]) - Math.log(M.prior[0]) - lg(x[0], M.st[0].m[0], M.st[0].v[0]) - lg(x[1], M.st[0].m[1], M.st[0].v[1]);
// gerçek Bayes logiti (tam kovaryanslı gerçek dağılımlarla)
function trueLogit(q, x) {
  const l = (y) => { const sc = y ? q.vr : 1, mu = [(y ? 1 : -1) * q.sep / 2, (y ? 1 : -1) * q.sep * 0.15], a = sc * sc, c = q.rho * sc * sc, det = a * a - c * c, d0 = x[0] - mu[0], d1 = x[1] - mu[1]; return -0.5 * Math.log(det) - (a * d0 * d0 - 2 * c * d0 * d1 + a * d1 * d1) / (2 * det); };
  return l(1) - l(0);
}
export function fitLR(X, Y, steps = 300) { let w = [0, 0], b = 0; const n = X.length; for (let t = 0; t < steps; t++) { let g0 = 0, g1 = 0, gb = 0; for (let i = 0; i < n; i++) { const z = w[0] * X[i][0] + w[1] * X[i][1] + b, p = 1 / (1 + Math.exp(-z)), d = p - Y[i]; g0 += d * X[i][0]; g1 += d * X[i][1]; gb += d; } w = [w[0] - (0.5 * g0) / n - 0.5 * 0.001 * w[0], w[1] - (0.5 * g1) / n - 0.5 * 0.001 * w[1]]; b -= (0.5 * gb) / n; } return (x) => w[0] * x[0] + w[1] * x[1] + b; }

export default {
  params: [
    { id: "sep", label: "Sınıf merkezleri arası uzaklık", min: 0.5, max: 5, step: 0.1, value: 2 },
    { id: "rho", label: "Sınıf içi korelasyon ρ (x₁, x₂)", min: -0.95, max: 0.95, step: 0.05, value: 0.8, help: "ρ ≠ 0: sınıf bilindiğinde öznitelikler bağımsız değil — naif varsayım ihlal ediliyor." },
    { id: "vr", label: "Sınıf 1'in std çarpanı", min: 0.4, max: 2.5, step: 0.05, value: 1, help: "≠ 1 ise iki sınıfın varyansı farklı: gerçek Bayes sınırı eğri (kuadratik)." },
    { id: "shared", label: "GNB varyansı", options: ["sınıf başına ayrı", "sınıflar arası ortak"], value: 0 },
    { id: "n", label: "Eğitim örneği", min: 10, max: 500, value: 100, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), { X, Y } = sample(n, q, r), M = fitGNB(X, Y, q.shared === 1), T = sample(2000, q, r);
    const R = 4.2, G = 120, nb = [[], []], tb = [[], []];
    const scan = (f, out) => { for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const x = -R + (i * 2 * R) / (G - 1), y = -R + (j * 2 * R) / (G - 1); if (i < G - 1 && Math.sign(f([x, y])) !== Math.sign(f([x + (2 * R) / (G - 1), y]))) { out[0].push(x); out[1].push(y); } else if (j < G - 1 && Math.sign(f([x, y])) !== Math.sign(f([x, y + (2 * R) / (G - 1)]))) { out[0].push(x); out[1].push(y); } } };
    scan((x) => nbLogit(M, x), nb); scan((x) => trueLogit(q, x), tb);
    const errOf = (f) => T.X.reduce((s, x, i) => s + ((f(x) > 0 ? 1 : 0) !== T.Y[i] ? 1 : 0), 0) / T.X.length;
    const bayes = errOf((x) => trueLogit(q, x)), nbE = errOf((x) => nbLogit(M, x));
    // kalibrasyon: NB'nin olasılığı ile gerçek posterior farkı
    const cal = T.X.slice(0, 600).map((x) => Math.abs(1 / (1 + Math.exp(-nbLogit(M, x))) - 1 / (1 + Math.exp(-trueLogit(q, x)))));
    return {
      plots: [
        { title: "Karar sınırları: GNB (turuncu) ve gerçek Bayes (yeşil)", xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-R, R], ylim: [-R, R], series: [
          { name: "sınıf 0", x: X.filter((_, i) => !Y[i]).map((v) => v[0]), y: X.filter((_, i) => !Y[i]).map((v) => v[1]), kind: "scatter", c: 0, r: 2.6 },
          { name: "sınıf 1", x: X.filter((_, i) => Y[i]).map((v) => v[0]), y: X.filter((_, i) => Y[i]).map((v) => v[1]), kind: "scatter", c: 3, r: 2.6 },
          { name: "GNB sınırı", x: nb[0], y: nb[1], kind: "scatter", c: 2, r: 1.2 },
          { name: "gerçek Bayes sınırı (en iyi olası)", x: tb[0], y: tb[1], kind: "scatter", c: 1, r: 1.2 },
        ] },
        { title: "Kalibrasyon: GNB olasılığı vs gerçek posterior (600 test örneği)", xlabel: "gerçek P(y = 1 | x)", ylabel: "GNB'nin P(y = 1 | x)", size: { w: 380, h: 370 }, xlim: [0, 1], ylim: [0, 1], series: [
          { name: "test örnekleri", x: T.X.slice(0, 600).map((x) => 1 / (1 + Math.exp(-trueLogit(q, x)))), y: T.X.slice(0, 600).map((x) => 1 / (1 + Math.exp(-nbLogit(M, x)))), kind: "scatter", c: 2, r: 1.8 },
          { name: "mükemmel kalibrasyon", x: [0, 1], y: [0, 1], c: 4, dash: true },
        ] },
      ],
      stats: [
        { label: "Bayes hatası (en iyi olası)", value: bayes },
        { label: "GNB test hatası", value: nbE },
        { label: "GNB olasılığı − gerçek posterior, ort. |fark|", value: cal.reduce((a, b) => a + b, 0) / cal.length },
        { label: "GNB parametre sayısı (2 sınıf × 2 öznitelik × (μ, σ²) + önsel)", value: q.shared === 1 ? 7 : 9 },
      ],
      note: `${Math.abs(q.rho) > 0.3 ? `<b>ρ = ${fmt(q.rho, 2)}:</b> sınıf içinde x₁ ve x₂ ilişkili; NB onları bağımsız sayar, aynı kanıtı iki kez sayar. Sınır gerçek Bayes sınırından sapar ve olasılıklar aşırı uçlara itilir (kalibrasyon farkı). ` : "ρ ≈ 0: naif varsayım burada doğru; GNB gerçek Bayes sınırına çok yakın. "}${q.shared === 1 ? "Ortak varyansla iki sınıfın kuadratik terimleri sadeleşir: GNB sınırı bir <b>doğrudur</b> — lojistik regresyonla aynı biçim, farklı kestirim. " : q.vr !== 1 ? "Varyanslar farklı: gerçek sınır ve GNB sınırı eğridir (kuadratik). " : ""}Sağ grafikte noktalar köşegenden uzaklaştıkça NB'nin olasılıkları yanlıştır; köşegenin uçlarına doğru bükülmüş bir bulut (orta değerler uçlara itilmiş) aşırı güvenin imzasıdır.`,
    };
  },
};
