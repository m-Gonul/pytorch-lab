// M9.5 — Yalnız gürültü varken restorasyon, 2-B. 80 × 80'lik bir test deseni (dikdörtgenler, rampa, 1 piksellik ince çizgiler,
// noktalar; 8 bit) Gauss, tuz-biber ya da karışık gürültüyle bozulur. Ortalama süzgeçleri (aritmetik, geometrik, harmonik,
// kontraharmonik Q), medyan (tekrarlı geçiş), uyarlamalı ortalama (σ_η² kestirimi gerçeğin kaç katı) ve uyarlamalı medyan (S_max)
// karşılaştırılır. Ölçüt: PSNR (bütün görüntü, düz bölge, kenar bölgesi); bütün süzgeçlerin aynı girdideki PSNR'ı yan yana.
import { fmt, range } from "./plot.js";

const W = 80;
export function pattern() {
  const f = new Array(W * W).fill(90);
  const box = (x0, y0, x1, y1, v) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) f[y * W + x] = v; };
  box(6, 6, 36, 36, 190); box(44, 6, 74, 22, 235); box(44, 26, 74, 36, 25);
  for (let y = 44; y < 74; y++) for (let x = 6; x < 36; x++) f[y * W + x] = Math.round(30 + (200 * (x - 6)) / 29);
  for (const x of [46, 50, 54, 58]) box(x, 44, x + 1, 74, 220);
  for (const [x, y] of [[64, 46], [70, 52], [64, 60], [70, 68]]) box(x, y, x + 2, y + 2, 15);
  return f;
}
const clamp = (v) => Math.max(0, Math.min(255, v));
const win = (g, x, y, h) => { const o = []; for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) o.push(g[Math.min(W - 1, Math.max(0, y + j)) * W + Math.min(W - 1, Math.max(0, x + i))]); return o; };
const med = (a) => { const s = a.slice().sort((u, v) => u - v); return s[(s.length - 1) >> 1]; };
export function filt(g, kind, o) {
  const h = (o.k - 1) / 2, out = new Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const z = g[y * W + x];
    let v;
    if (kind === 0) { const a = win(g, x, y, h); v = a.reduce((s, u) => s + u, 0) / a.length; }
    else if (kind === 1) { const a = win(g, x, y, h); v = a.some((u) => u === 0) ? 0 : Math.exp(a.reduce((s, u) => s + Math.log(u), 0) / a.length); }
    else if (kind === 2) { const a = win(g, x, y, h); v = a.some((u) => u === 0) ? 0 : a.length / a.reduce((s, u) => s + 1 / u, 0); }
    else if (kind === 3) { const a = win(g, x, y, h); if (o.Q < 0 && a.some((u) => u === 0)) v = 0; else { const num = a.reduce((s, u) => s + u ** (o.Q + 1), 0), den = a.reduce((s, u) => s + u ** o.Q, 0); v = den > 0 ? num / den : 0; } }
    else if (kind === 4) v = med(win(g, x, y, h));
    else if (kind === 5) { const a = win(g, x, y, h), m = a.reduce((s, u) => s + u, 0) / a.length, s2 = a.reduce((s, u) => s + (u - m) ** 2, 0) / a.length; v = z - Math.min(1, o.ne / Math.max(s2, 1e-12)) * (z - m); }
    else {
      let hh = 1; v = null;
      while (v === null) {
        const a = win(g, x, y, hh).sort((u, w) => u - w), zmin = a[0], zmax = a[a.length - 1], zmed = a[(a.length - 1) >> 1];
        if (zmin < zmed && zmed < zmax) v = zmin < z && z < zmax ? z : zmed;
        else if (2 * hh + 1 < o.smax) hh++;
        else v = zmed;
      }
    }
    out[y * W + x] = clamp(v);
  }
  return out;
}
export const psnr = (a, b, idx) => { const I = idx || range(a.length); const m = I.reduce((s, i) => s + (a[i] - b[i]) ** 2, 0) / I.length; return m < 1e-6 ? Infinity : 10 * Math.log10((255 * 255) / m); };
const NAMES = ["aritmetik ortalama", "geometrik ortalama", "harmonik ortalama", "kontraharmonik", "medyan", "uyarlamalı ortalama", "uyarlamalı medyan"];
const SHORT = ["arit.", "geom.", "harm.", "kontra.", "medyan", "uyar. ort.", "uyar. med."];

export default {
  params: [
    { id: "nt", label: "Gürültü", options: ["Gauss", "tuz-biber", "yalnız biber", "yalnız tuz", "Gauss + tuz-biber"], value: 1 },
    { id: "sig", label: "Gauss σ", min: 2, max: 50, step: 1, value: 20, on: { nt: [0, 4] } },
    { id: "P", label: "Darbe yoğunluğu P", min: 0.01, max: 0.6, step: 0.01, value: 0.1, on: { nt: [1, 2, 3, 4] } },
    { id: "f", label: "Süzgeç", options: NAMES, value: 4 },
    { id: "k", label: "Pencere", options: ["3 × 3", "5 × 5", "7 × 7"], value: 0, on: { f: [0, 1, 2, 3, 4, 5] } },
    { id: "Q", label: "Kontraharmonik derece Q", min: -3, max: 3, step: 0.5, value: 1.5, on: { f: [3] } },
    { id: "passes", label: "Medyan geçiş sayısı", min: 1, max: 4, value: 1, int: true, on: { f: [4] } },
    { id: "nef", label: "σ_η² kestirimi / gerçek σ_η²", min: 0.1, max: 10, value: 1, log: true, on: { f: [5] } },
    { id: "smax", label: "S_max", options: ["3", "5", "7", "9", "11"], value: 2, on: { f: [6] } },
  ],
  seed: true,
  run(q, r) {
    const f = pattern(), k = 2 * q.k + 3, smax = 2 * q.smax + 3;
    const g = f.map((v) => {
      let u = v;
      if (q.nt === 0 || q.nt === 4) u = clamp(Math.round(u + q.sig * r.normal()));
      if (q.nt >= 1) { const a = r(); const ps = q.nt === 2 ? 0 : q.nt === 3 ? q.P : q.P / 2, pp = q.nt === 3 ? 0 : q.nt === 2 ? q.P : q.P / 2; if (a < ps) u = 255; else if (a < ps + pp) u = 0; }
      return u;
    });
    const nvar = g.reduce((s, v, i) => s + (v - f[i]) ** 2, 0) / g.length;
    const opt = (kind) => ({ k, Q: q.Q, ne: q.nef * nvar, smax });
    const run1 = (kind) => { if (kind !== 4) return filt(g, kind, opt(kind)); let o = g; for (let p = 0; p < q.passes; p++) o = filt(o, 4, opt(4)); return o; };
    const out = run1(q.f);
    const edge = [], flat = [];
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const a = win(f, x, y, 1); (Math.max(...a) > Math.min(...a) ? edge : flat).push(y * W + x); }
    const all = range(7).map((kind) => (kind === q.f ? out : run1(kind))), ps = all.map((o) => psnr(o, f)), best = ps.indexOf(Math.max(...ps));
    const P = [`Girdi PSNR ${fmt(psnr(g, f), 3)} dB → ${NAMES[q.f]} ${fmt(ps[q.f], 3)} dB (düz bölgede ${fmt(psnr(out, f, flat), 3)}, kenar bölgesinde ${fmt(psnr(out, f, edge), 3)} dB). Bu girdide en yüksek PSNR: <b>${NAMES[best]}</b> (${fmt(ps[best], 3)} dB).`];
    P.push([
      "Aritmetik ortalama gürültüyü bulanıklaştırarak azaltır; darbeleri ise komşulara yayar.",
      "Geometrik ortalama aritmetiğe yakın düzleştirir, ayrıntıyı biraz daha az kaybeder; ama penceredeki tek bir 0 (biber) çıkışı 0 yapar.",
      "Harmonik ortalama tuzda iyi çalışır (1/255 terimi küçüktür), biberde başarısız olur (1/0 terimi çıkışı 0'a çeker).",
      `Kontraharmonik Q > 0 biberi, Q < 0 tuzu siler; yanlış işaret gürültüyü kötüleştirir. Q = 0 aritmetik, Q = −1 harmonik ortalamadır. Seçili Q = ${q.Q}.`,
      `Medyan darbeleri sıralamanın uçlarına iter ve seçmez; ${q.passes > 1 ? `${q.passes} geçiş kalan darbeleri de temizler ama ince çizgileri ve köşeleri aşındırır` : "darbe yoğunluğu yükseldikçe (≈ P > 0.2) pencerede çoğunluk gürültü olur ve medyan da darbe olur"}.`,
      `Uyarlamalı ortalama yerel varyans σ_S², gürültü varyansından çok büyükse pikseli bırakır, yakınsa ortalamaya çeker. Kestirim gerçeğin ${fmt(q.nef, 3)} katı: ${q.nef < 0.7 ? "fazla küçük kestirim düzeltmeyi azaltır, çıkış girdiye benzer" : q.nef > 1.5 ? "fazla büyük kestirimde oran 1'e kırpılır ve süzgeç her yerde aritmetik ortalama gibi davranır" : "doğru kestirim, en iyimser durum"}. Darbe gürültüsünde σ_η² anlamlı bir ölçek değildir.`,
      `Uyarlamalı medyan penceresini S_max = ${smax}'e kadar büyütür; darbe olmayan pikselleri olduğu gibi bırakır (Seviye B), bu yüzden ince çizgileri korur.`,
    ][q.f]);
    if (ps[q.f] < psnr(g, f)) P.push(`Seçili süzgeç PSNR'ı <b>düşürdü</b>: desende ince çizgi ve nokta çok; düz bölgede kazanılan, kenar bölgesinde (${fmt(psnr(g, f, edge), 3)} → ${fmt(psnr(out, f, edge), 3)} dB) kaybedilenden az.`);
    const img = (a) => ({ w: W, h: W, data: a.map((v) => v / 255) });
    return {
      plots: [
        { title: "Gürültülü girdi", img: img(g), side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `Çıkış: ${NAMES[q.f]}`, img: img(out), side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Aynı girdide bütün süzgeçler (PSNR, dB)", xlabel: "süzgeç", ylabel: "PSNR (dB)", wide: true, size: { w: 620, h: 180 }, series: [{ name: "PSNR", x: range(7), y: ps.map((v) => Math.min(v, 60)), kind: "bar", c: 0 }, { name: "girdi", x: [-0.5, 6.5], y: [psnr(g, f), psnr(g, f)], c: 1, dash: true }], xticks: range(7).map((i) => [i, SHORT[i]]) },
      ],
      stats: [{ label: "PSNR girdi → çıkış", value: `${fmt(psnr(g, f), 3)} → ${fmt(ps[q.f], 3)} dB` }, { label: "Düz / kenar bölgesi PSNR", value: `${fmt(psnr(out, f, flat), 3)} / ${fmt(psnr(out, f, edge), 3)} dB` }, { label: "Gürültü varyansı (g − f)", value: fmt(nvar, 4) }, { label: "En iyi süzgeç", value: NAMES[best] }],
      note: P.join(" "),
    };
  },
};
