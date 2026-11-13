// M11.12 — GAE(λ): kusurlu eleştirmenle avantaj tahmininde yanlılık–varyans dengesi. Durumlar 1..S olan bir zincirde politika olasılık p ile sağa,
// 1 − p ile sola gider; S + 1'e varmak +1 ödül verir, 0'a varmak 0 (ikisi de uç). Her adımın ödülüne N(0, σ_r²) gürültü eklenir.
// Eleştirmen V̂(s) = κ·V(s) + σ_V·z_s (her eleştirmen çekiliminde z_s yeniden). Tahmin edilen nicelik başlangıç durumundaki iki eylemin avantaj farkı
// D = A(s0, sağ) − A(s0, sol) = Q(s0, sağ) − Q(s0, sol); V̂(s0) temel çizgisi bu farkta sadeleşir (politika gradyanında olduğu gibi).
import { fmt, range } from "./plot.js";

const LG = range(21).map((i) => i / 20);
function trueV(S, p, g) {
  let V = Array(S + 2).fill(0);
  for (let it = 0; it < 20000; it++) {
    let ch = 0; const W = V.slice();
    for (let s = 1; s <= S; s++) { const right = (s + 1 === S + 1 ? 1 : 0) + g * V[s + 1], left = g * V[s - 1]; W[s] = p * right + (1 - p) * left; ch = Math.max(ch, Math.abs(W[s] - V[s])); }
    V = W; if (ch < 1e-12) break;
  }
  return V; // V[0] = V[S+1] = 0 (uç)
}
export function measure(q, r) {
  const S = q.S, s0 = Math.ceil(S / 2), g = q.g, V = trueV(S, q.p, g);
  const Q = (a) => { const s1 = s0 + a; return (s1 === S + 1 ? 1 : 0) + g * V[s1]; };
  const D = Q(1) - Q(-1), C = 10, N = 600; // her eleştirmen ve eylem için N yörünge: ortalama ve tek örnek varyansı; M'lik ortalamanın HKO'su = yanlılık² + varyans / M
  const b2 = Array(LG.length).fill(0), va = Array(LG.length).fill(0); let Vh0 = null, lens = 0;
  for (let c = 0; c < C; c++) {
    const Vh = V.map((v, s) => (s === 0 || s === S + 1 ? 0 : q.kap * v + q.sv * r.normal()));
    if (!Vh0) Vh0 = Vh;
    const mean = { 1: Array(LG.length).fill(0), [-1]: Array(LG.length).fill(0) }, sq = { 1: Array(LG.length).fill(0), [-1]: Array(LG.length).fill(0) };
    for (const a0 of [1, -1]) for (let m = 0; m < N; m++) {
      let s = s0, a = a0; const del = [];
      for (let t = 0; t < 2000; t++) {
        const s1 = s + a, done = s1 === 0 || s1 === S + 1, rew = (s1 === S + 1 ? 1 : 0) + q.sr * r.normal();
        del.push(rew + (done ? 0 : g * Vh[s1]) - Vh[s]);
        if (done) break; s = s1; a = r() < q.p ? 1 : -1;
      }
      lens += del.length;
      LG.forEach((l, j) => { let A = 0, w = 1; for (const d of del) { A += w * d; w *= g * l; if (w === 0) break; } mean[a0][j] += A / N; sq[a0][j] += (A * A) / N; });
    }
    LG.forEach((l, j) => { const est = mean[1][j] - mean[-1][j], v1 = (sq[1][j] - mean[1][j] ** 2 + sq[-1][j] - mean[-1][j] ** 2) * N / (N - 1);
      b2[j] += Math.max((est - D) ** 2 - v1 / N, 0) / C; va[j] += Math.max(v1, 0) / q.M / C; }); // (est − D)² içindeki örnekleme gürültüsü v1/N düşülür
  }
  return { V, Vh0, D, b2, va, mse: b2.map((v, j) => v + va[j]), len: lens / (C * 2 * N), s0 };
}
export default {
  params: [
    { id: "S", label: "Zincir uzunluğu S", min: 3, max: 30, value: 10, int: true },
    { id: "p", label: "Sağa gitme olasılığı p", min: 0.2, max: 0.8, step: 0.05, value: 0.5 },
    { id: "g", label: "İndirim γ", min: 0.8, max: 1, step: 0.01, value: 0.98 },
    { id: "sr", label: "Adım ödülü gürültüsü σ_r", min: 0, max: 1, step: 0.05, value: 0.3 },
    { id: "sv", label: "Eleştirmen gürültüsü σ_V", min: 0, max: 0.5, step: 0.01, value: 0.2 },
    { id: "kap", label: "Eleştirmen ölçeği κ", min: 0, max: 1.5, step: 0.05, value: 1 },
    { id: "M", label: "Ortalaması alınan yörünge M", min: 1, max: 1000, value: 30, int: true, log: true },
    { id: "lam", label: "İşaretlenen λ", min: 0, max: 1, step: 0.05, value: 0.95 },
  ],
  seed: true,
  run(q, r) {
    const o = measure(q, r), j = Math.round(q.lam * 20), best = o.mse.indexOf(Math.min(...o.mse));
    const fl = (a) => a.map((v) => Math.max(v, 1e-6));
    const P = [`Gerçek fark D = Q(s0, sağ) − Q(s0, sol) = ${fmt(o.D, 4)} (s0 = ${o.s0}); ortalama bölüm uzunluğu ${fmt(o.len, 3)} adım.`];
    P.push(`λ = 0 (tek adımlı TD): yanlılık² ${fmt(o.b2[0], 3)}, varyans ${fmt(o.va[0], 3)}; λ = 1 (Monte Carlo): yanlılık² ${fmt(o.b2[20], 3)}, varyans ${fmt(o.va[20], 3)}. En küçük HKO λ = ${fmt(LG[best], 2)} noktasında (${fmt(o.mse[best], 3)}). Yanlılık² de örneklemeyle ölçülür (her eleştirmen ve eylem için 600 yörünge), bu yüzden 10⁻³ düzeyindeki değerler gürültüdür.`);
    if (q.sv === 0 && q.kap === 1) P.push(`Eleştirmen kusursuz: her λ yansızdır (kalan küçük yanlılık² örnekleme gürültüsüdür), λ yalnızca varyansı belirler ve en iyisi en az varyanslı λ = ${fmt(LG[best], 2)}.`);
    if (q.sr === 0) P.push("Ödül gürültüsü yok: varyans yalnızca politikanın ve ortamın rastgeleliğinden (yörüngenin nereye gittiğinden) geliyor.");
    P.push(`Yanlılık eleştirmenden gelir: λ küçükken tahmin V̂'ye yaslanır, λ → 1'de terimler teleskop gibi sadeleşir ve V̂ yalnızca s0'da kalır (farkta düşer). Varyans ise uzun ufuktaki ödül gürültüsünü toplar. ${q.M > 1 ? `M = ${q.M} yörüngenin ortalaması varyansı yaklaşık M kat küçültür; yanlılık değişmez, bu yüzden en iyi λ büyür.` : "M = 1 tek bir örneklemenin hatasıdır (PPO'da tek bir zaman adımının avantajı gibi)."}`);
    return {
      plots: [
        { title: "Yanlılık², varyans ve HKO (log)", xlabel: "λ", ylabel: "", side: true, size: { w: 300, h: 230 }, ylog: true, xlim: [0, 1], ylim: [10 ** Math.floor(Math.log10(Math.max(1e-6, Math.min(...fl(o.b2), ...fl(o.va))))), 10 ** Math.ceil(Math.log10(Math.max(...o.mse)))], marks: [{ x: q.lam, label: `λ = ${fmt(q.lam, 2)}` }],
          series: [{ name: "yanlılık²", x: LG, y: fl(o.b2), c: 0 }, { name: "varyans", x: LG, y: fl(o.va), c: 1 }, { name: "HKO", x: LG, y: fl(o.mse), c: 2 }] },
        { title: "Gerçek V ve bir eleştirmen çekilimi", xlabel: "durum", ylabel: "değer", side: true, size: { w: 300, h: 230 }, xlim: [0.5, q.S + 0.5],
          series: [{ name: "gerçek V", x: range(q.S).map((i) => i + 1), y: o.V.slice(1, q.S + 1), c: 0 }, { name: "eleştirmen", x: range(q.S).map((i) => i + 1), y: o.Vh0.slice(1, q.S + 1), c: 3, kind: "scatter" }] },
      ],
      stats: [
        { label: `λ = ${fmt(q.lam, 2)}: yanlılık² / varyans`, value: `${fmt(o.b2[j], 3)} / ${fmt(o.va[j], 3)}` },
        { label: `λ = ${fmt(q.lam, 2)}: HKO`, value: fmt(o.mse[j], 3) },
        { label: "En iyi λ (HKO)", value: fmt(LG[best], 2) },
        { label: "Gerçek D", value: fmt(o.D, 4) },
      ],
      note: P.join(" "),
    };
  },
};
