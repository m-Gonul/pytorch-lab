// M0.4 — Merkezi limit teoremi: n örneğin ortalaması, temel dağılım ne olursa olsun (sonlu varyansla) N(μ, σ²/n)'ye yaklaşır.
import { fmt, linspace, range } from "./plot.js";

export const BASES = [
  { name: "Düzgün U(0, 1)", mu: 0.5, v: 1 / 12, draw: (r) => r() },
  { name: "Üstel Exp(1) (çarpık)", mu: 1, v: 1, draw: (r) => -Math.log(1 - r()) },
  { name: "Bernoulli(0.1) (ayrık, çok çarpık)", mu: 0.1, v: 0.09, draw: (r) => (r() < 0.1 ? 1 : 0) },
  { name: "İki tepeli: ±2 civarı", mu: 0, v: 4 + 0.09, draw: (r) => (r() < 0.5 ? -2 : 2) + 0.3 * r.normal() },
  { name: "Cauchy (varyansı yok!)", mu: NaN, v: NaN, draw: (r) => Math.tan(Math.PI * (r() - 0.5)) },
];

export default {
  params: [
    { id: "base", label: "Temel dağılım", options: BASES.map((b) => b.name), value: 1 },
    { id: "n", label: "Ortalamadaki örnek sayısı n", min: 1, max: 200, value: 5, int: true, log: true, help: "Her tekrar, n bağımsız örneğin ortalamasını alır." },
    { id: "R", label: "Tekrar sayısı (histogramdaki ortalama sayısı)", min: 200, max: 20000, value: 4000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const B = BASES[q.base], n = Math.round(q.n), R = Math.round(q.R), means = [];
    for (let k = 0; k < R; k++) { let s = 0; for (let i = 0; i < n; i++) s += B.draw(r); means.push(s / n); }
    const m = means.reduce((a, b) => a + b, 0) / R, sd = Math.sqrt(means.reduce((a, b) => a + (b - m) ** 2, 0) / (R - 1));
    const skew = means.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / R;
    const cauchy = !Number.isFinite(B.v), se = Math.sqrt(B.v / n);
    const sorted = [...means].sort((a, b) => a - b), qt = (p) => sorted[Math.floor(p * (R - 1))];
    const lo = cauchy ? qt(0.02) : B.mu - 4.5 * se, hi = cauchy ? qt(0.98) : B.mu + 4.5 * se;
    const nb = 50, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
    for (const x of means) { const i = Math.floor((x - lo) / w); if (i >= 0 && i < nb) cnt[i]++; }
    const cx = range(nb).map((i) => lo + (i + 0.5) * w), gx = linspace(lo, hi, 200);
    const series = [{ name: `${R} ortalamanın histogramı (yoğunluk)`, x: cx, y: cnt.map((c) => c / (R * w)), kind: "bar", c: 0, bw: 0.95 }];
    if (!cauchy) series.push({ name: "N(μ, σ²/n) — MLT'nin öngörüsü", x: gx, y: gx.map((x) => Math.exp(-((x - B.mu) ** 2) / (2 * se * se)) / (se * Math.sqrt(2 * Math.PI))), c: 2 });
    // standart hatanın n'ye göre düşüşü: deneysel (küçük ek benzetim) ve teorik
    const ns = [1, 2, 4, 8, 16, 32, 64, 128], emp = ns.map((nn) => { const M = []; for (let k = 0; k < 300; k++) { let s = 0; for (let i = 0; i < nn; i++) s += B.draw(r); M.push(s / nn); } const mm = M.reduce((a, b) => a + b, 0) / 300; return cauchy ? (qtile(M, 0.75) - qtile(M, 0.25)) / 2 : Math.sqrt(M.reduce((a, b) => a + (b - mm) ** 2, 0) / 299); });
    return {
      plots: [
        { title: `${B.name}: n = ${n} örneğin ortalaması`, xlabel: "örneklem ortalaması x̄", ylabel: "yoğunluk", yzero: true, series, marks: cauchy ? [] : [{ x: B.mu, label: "μ" }] },
        { title: cauchy ? "Ortalamaların yayılımı (çeyrekler arası / 2) — n ile küçülmüyor" : "Ortalamanın standart hatası, n'ye göre (log ölçek)", xlabel: "n (log₂)", ylabel: "yayılım", ylog: true, series: [
          { name: "deneysel (300 tekrar)", x: ns.map(Math.log2), y: emp, kind: "scatter", c: 0, r: 4 },
          ...(cauchy ? [] : [{ name: "σ/√n", x: ns.map(Math.log2), y: ns.map((nn) => Math.sqrt(B.v / nn)), c: 2 }]),
        ] },
      ],
      stats: [
        { label: "Temel dağılımın μ, σ²", value: cauchy ? "tanımsız" : `${fmt(B.mu)}, ${fmt(B.v)}` },
        { label: "Ortalamaların ortalaması", value: m },
        { label: "Ortalamaların std sapması (deneysel)", value: sd },
        { label: "σ/√n (teorik standart hata)", value: cauchy ? "tanımsız" : se },
        { label: "Çarpıklık (0 = simetrik)", value: skew },
      ],
      note: cauchy
        ? "Cauchy dağılımının ortalaması ve varyansı <b>yoktur</b> (kuyruklar çok kalın). n örneğin ortalaması yine aynı Cauchy dağılımıdır: n artsa da yayılım küçülmez. MLT'nin koşulu (sonlu varyans) bozulunca sonuç da bozulur — ağır kuyruklu kayıplar ve gradyanlarda ortalamaya güvenmemenin nedeni."
        : `Temel dağılım ne kadar çarpık olursa olsun, ortalamaların dağılımı n büyüdükçe çan eğrisine yaklaşır ve genişliği σ/√n ile küçülür. Çarpık dağılımlarda yaklaşım daha yavaştır (çarpıklık ~1/√n hızla söner). SGD'de mini-batch gradyanının yaklaşık Gauss gürültülü olması ve batch büyüdükçe gürültünün 1/√B ile azalması bu teoremdir.`,
    };
  },
};
function qtile(a, p) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(p * (s.length - 1))]; }
