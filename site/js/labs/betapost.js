// M11.7 — İki madeni para (Bernoulli kol), Beta önsel. Gözlenen tura/yazı sayılarıyla sonsal Beta(α₀ + tura, β₀ + yazı). Thompson
// örneklemesinin bir sonraki seçimi: her kol için sonsaldan k örnek çekilip ortalanır, ortalaması büyük olan seçilir. k = 1'de C1'i
// seçme olasılığı P(θ₁ > θ₂) (sonsal olasılık); k büyüdükçe seçim sonsal ortalamaların karşılaştırmasına (açgözlü) yaklaşır.
// Olasılıklar tohumlu Monte Carlo ile (20 000 tekrar) hesaplanır.
import { fmt, range } from "./plot.js";

function gamma(r, a) { if (a < 1) return gamma(r, a + 1) * Math.pow(r(), 1 / a); const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = r.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
const beta = (r, a, b) => { const x = gamma(r, a), y = gamma(r, b); return x / (x + y); };
function lbeta(a, b) { const lg = (z) => { const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lg(1 - z); z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i); const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x); }; return lg(a) + lg(b) - lg(a + b); }
const pdf = (x, a, b) => Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
export default {
  params: [
    { id: "h1", label: "C1: tura", min: 0, max: 50, value: 3, int: true },
    { id: "t1", label: "C1: yazı", min: 0, max: 50, value: 2, int: true },
    { id: "h2", label: "C2: tura", min: 0, max: 50, value: 12, int: true },
    { id: "t2", label: "C2: yazı", min: 0, max: 50, value: 10, int: true },
    { id: "k", label: "Thompson örnek sayısı k", min: 1, max: 50, value: 1, int: true },
    { id: "a0", label: "Önsel Beta(α₀, α₀)", min: 0.5, max: 10, step: 0.5, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const A1 = q.a0 + q.h1, B1 = q.a0 + q.t1, A2 = q.a0 + q.h2, B2 = q.a0 + q.t2, M = 20000;
    let p1 = 0, pk = 0;
    for (let i = 0; i < M; i++) {
      if (beta(r, A1, B1) > beta(r, A2, B2)) p1++;
      let s1 = 0, s2 = 0; for (let j = 0; j < q.k; j++) { s1 += beta(r, A1, B1); s2 += beta(r, A2, B2); } if (s1 > s2) pk++;
    }
    p1 /= M; pk /= M;
    const m1 = A1 / (A1 + B1), m2 = A2 / (A2 + B2), xs = range(199).map((i) => (i + 1) / 200);
    const sd = (a, b) => Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)));
    const P = [`Sonsallar: C1 ~ Beta(${A1}, ${B1}) (ortalama ${fmt(m1, 3)}, std ${fmt(sd(A1, B1), 3)}), C2 ~ Beta(${A2}, ${B2}) (ortalama ${fmt(m2, 3)}, std ${fmt(sd(A2, B2), 3)}).`];
    P.push(`Thompson (k = 1) bir sonraki atışta C1'i ${fmt(100 * p1, 3)}% olasılıkla seçer: bu, C1'in daha iyi olduğunun sonsal olasılığıdır. k = ${q.k} ile ${fmt(100 * pk, 3)}%. Açgözlü seçim ${m1 > m2 ? "her zaman C1" : m1 < m2 ? "her zaman C2" : "eşit"} olurdu.`);
    if (q.k > 1) P.push("k büyüdükçe örneklerin ortalaması sonsal ortalamaya yaklaşır ve seçim açgözlüye döner: keşif azalır.");
    if (q.h1 + q.t1 < q.h2 + q.t2) P.push("Az gözlenen kolun sonsalı geniştir; ortalaması daha düşük olsa bile ara sıra yüksek bir örnek çekilir ve denenir. Belirsizlik keşfi kendiliğinden yönlendirir.");
    return {
      plots: [{ title: "Sonsal yoğunluklar", xlabel: "θ (tura olasılığı)", ylabel: "yoğunluk", wide: true, size: { w: 620, h: 230 },
        series: [{ name: `C1: Beta(${A1}, ${B1})`, x: xs, y: xs.map((x) => pdf(x, A1, B1)), c: 0 }, { name: `C2: Beta(${A2}, ${B2})`, x: xs, y: xs.map((x) => pdf(x, A2, B2)), c: 2 }],
        marks: Math.abs(m1 - m2) < 0.12 ? [{ x: m1, label: "" }, { x: m2, label: "" }] : [{ x: m1, label: "ort. C1" }, { x: m2, label: "ort. C2" }] }],
      stats: [{ label: "P(θ₁ > θ₂) (k = 1)", value: fmt(p1, 3) }, { label: `P(C1 seçilir), k = ${q.k}`, value: fmt(pk, 3) }, { label: "Sonsal ortalamalar", value: `${fmt(m1, 3)} / ${fmt(m2, 3)}` }],
      note: P.join(" "),
    };
  },
};
