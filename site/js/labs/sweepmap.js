// M1.2 — Aşama E'deki 48'lik (lr, β) ızgarasının haritası: son kayıp ısı haritası,
// kararlılık sınırı eğrisi ve mercek ucu. Veri tohumsuz ve sabit: sonuç deterministik.
import { fmt, linspace, rng } from "./plot.js";

// Aşama E ile aynı problem: X sütun ölçekleri 1 ve s, y = 2x₁ + 0.3x₂, w₀ = 0, tam batch momentum GD.
export function makeData(n, s) {
  const r = rng(7), X = [], Y = [];   // tohum sabit: harita tohuma bağlı değil
  for (let i = 0; i < n; i++) {
    const x1 = r.normal(), x2 = r.normal() * s;
    X.push([x1, x2]);
    Y.push(2 * x1 + 0.3 * x2);
  }
  return { X, Y };
}
export function sweep(X, Y, lrs, betas, steps) {
  const n = X.length, out = [];
  for (let p = 0; p < lrs.length; p++) {
    let w1 = 0, w2 = 0, v1 = 0, v2 = 0;
    for (let t = 0; t < steps; t++) {
      let g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) { const e = w1 * X[i][0] + w2 * X[i][1] - Y[i]; g1 += e * X[i][0]; g2 += e * X[i][1]; }
      g1 = (2 * g1) / n; g2 = (2 * g2) / n;
      v1 = betas[p] * v1 + g1; v2 = betas[p] * v2 + g2;
      w1 -= lrs[p] * v1; w2 -= lrs[p] * v2;
      if (!Number.isFinite(w1) || Math.abs(w1) > 1e8) { out.push(NaN); break; }
    }
    if (out.length < p + 1) {
      let L = 0; for (let i = 0; i < n; i++) { const e = w1 * X[i][0] + w2 * X[i][1] - Y[i]; L += e * e; }
      out.push(L / n);
    }
  }
  return out;
}
// H = 2XᵀX/n'nin 2×2 özdeğerleri (kapalı form)
export function hessEig(X) {
  let a = 0, b = 0, d = 0; const n = X.length;
  for (const r of X) { a += r[0] * r[0]; b += r[0] * r[1]; d += r[1] * r[1]; }
  a = (2 * a) / n; b = (2 * b) / n; d = (2 * d) / n;
  const tr = a + d, disc = Math.sqrt(Math.max((a - d) ** 2 + 4 * b * b, 0));
  return { lmax: (tr + disc) / 2, lmin: (tr - disc) / 2 };
}

export default {
  params: [
    { id: "steps", label: "Adım sayısı", min: 20, max: 300, step: 1, value: 100, int: true },
    { id: "s", label: "İkinci öznitelik ölçeği s (κ ≈ s²)", min: 2, max: 20, step: 1, value: 8, int: true, help: "X₂ = s·randn: s büyüdükçe koşul sayısı κ ≈ s² artar, uygun η penceresi daralır." },
  ],
  run(p) {
    const n = 250, { X, Y } = makeData(n, p.s);
    const lrs = linspace(-4, -2, 6).map((e) => 10 ** e);          // Aşama E'nin logspace(-4,-2,6) ızgarası
    const betas = linspace(0, 0.95, 8);
    const P = [];
    for (const lr of lrs) for (const b of betas) P.push([lr, b]);
    const L = sweep(X, Y, P.map((q) => q[0]), P.map((q) => q[1]), Math.round(p.steps));
    const { lmax, lmin } = hessEig(X), kap = lmax / lmin;
    // en iyi kombinasyon
    let bi = -1; for (let i = 0; i < L.length; i++) if (Number.isFinite(L[i]) && (bi < 0 || L[i] < L[bi])) bi = i;
    const best = bi >= 0 ? { lr: P[bi][0], b: P[bi][1], L: L[bi] } : null;
    const bstar = ((Math.sqrt(kap) - 1) / (Math.sqrt(kap) + 1)) ** 2, etastar = (2 / (Math.sqrt(lmax) + Math.sqrt(lmin))) ** 2;
    // ısı haritası: x = β, y = log10 η; hücre merkezleri ızgarayla hizalı
    const x0 = -0.95 / 14, x1 = 0.95 * 15 / 14, y0 = -4.2, y1 = -1.8;
    const edge = betas.map((b) => Math.log10(Math.min(2 * (1 + b) / lmax, 10)));
    const lprof = (b) => P.map((q, i) => [q, L[i]]).filter((q) => q[0][1] === b).map((q) => [Math.log10(q[0][0]), q[1]]);
    const p0 = lprof(0), p8 = lprof(Number((betas[6]).toFixed(4)));
    return {
      plots: [
        { title: "Son kayıp (log renk): açık = düşük kayıp. Kırmızı eğri: η = 2(1+β)/λ_max", xlabel: "β", ylabel: "log₁₀ η", size: { w: 430, h: 330 },
          field: { x0, x1, y0, y1, nx: 8, ny: 6, z: L.map((v) => (Number.isFinite(v) ? v : Math.max(...L.filter(Number.isFinite), 1e12) * 1.01)), log: true }, xlim: [0, 0.95], ylim: [-4.05, -1.95],
          series: [
            { name: "η = 2(1+β)/λ_max (ıraksama sınırı)", x: betas, y: edge, c: 1 },
            { name: "mercek ucu (β*, η*)", x: [bstar], y: [Math.log10(etastar)], kind: "scatter", c: 1, r: 5 },
            { name: "ızgaranın en iyisi", x: best ? [best.b] : [], y: best ? [Math.log10(best.lr)] : [], kind: "scatter", c: 0, r: 5 },
          ] },
        { title: "β = 0 (GD) ve β ≈ 0.81 profilleri: son kayıp − η", xlabel: "log₁₀ η", ylabel: "son kayıp", ylog: true,
          series: [
            { name: "β = 0 (GD)", x: p0.map((q) => q[0]), y: p0.map((q) => Math.max(q[1], 1e-16)), c: 0, kind: "step", dots: true },
            { name: "β = 0.814 (6. sütun)", x: p8.map((q) => q[0]), y: p8.map((q) => Math.max(q[1], 1e-16)), c: 2, kind: "step", dots: true },
          ] },
      ],
      stats: [
        { label: "Izgaranın en iyisi (η, β)", value: best ? `η=${fmt(best.lr)}, β=${fmt(best.b)}` : "—" },
        { label: "En iyi son kayıp", value: best ? best.L : "—" },
        { label: "λ_max, λ_min (H = 2XᵀX/n)", value: `${fmt(lmax)}, ${fmt(lmin)}` },
        { label: "κ = λ_max/λ_min", value: kap },
        { label: "Teorik β* = ((√κ−1)/(√κ+1))²", value: bstar },
        { label: "Teorik η* = (2/(√λmax+√λmin))²", value: etastar },
      ],
      note: `Aşama E'de bu 48 kombinasyonu PyTorch'ta <b>tek tensörle</b> eğitmiştin; harita aynı hesabın tarayıcı karşılığı. Kayıp sol üstten sağ alta iner: η büyüdükçe ve β orta düzeyde kaldıkça hızlanma artar. Dik yön (λ_max ≈ ${fmt(lmax)}) kararlılığı η'yı sınırlar: kırmızı eğrinin üstü ıraksama. β = 0 profilinde yatık yön (λ_min ≈ ${fmt(lmin)}) her adımda hatanın yalnız 1 − ηλ_min oranını sildiği için kayıp yüksek kalır; β ≈ ${fmt(betas[6])} profilinde momentum yatık yöndeki etkin adımı η/(1−β) katına çıkarıp bu tabanı indirir — en iyi β ne 0 ne de sınır (0.95), ortada. Teorik uç η* = (2/(√λmax+√λmin))² ≈ ${fmt(etastar)} ızgaranın en iyi η'sına yakın: κ büyüdükçe (s parametresi) uygun pencere daralır.`,
    };
  },
};
