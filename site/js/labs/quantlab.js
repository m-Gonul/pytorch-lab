// M7.5 — Ağırlık nicemleme: W (64 × 256). Satırlar farklı ölçekli (log-normal) Gauss; bir oranda aykırı değer (× F). Ya da ReLU sonrası
// aktivasyon benzeri veri (hepsi ≥ 0). Blok (tensör / satır / g'lik grup) başına ölçek: simetrik s = max|w| / (2^(b−1) − 1), q ∈ [−2^(b−1), 2^(b−1) − 1];
// afin s = (max − min) / (2^b − 1), z = round(−min / s). Hata: ‖W − Ŵ‖/‖W‖ ve rastgele girdiyle çıktı hatası. Bit/ağırlık: b + 16 / blok boyu.
import { fmt, range } from "./plot.js";

const R = 64, C = 256;
export function quantBlock(v, bits, affine) {
  if (!affine) { const m = Math.max(...v.map(Math.abs)), qm = 2 ** (bits - 1) - 1, s = m / qm || 1; return v.map((x) => Math.max(-qm - 1, Math.min(qm, Math.round(x / s))) * s); }
  const lo = Math.min(0, ...v), hi = Math.max(0, ...v), L = 2 ** bits - 1, s = (hi - lo) / L || 1, z = Math.round(-lo / s);
  return v.map((x) => (Math.max(0, Math.min(L, Math.round(x / s) + z)) - z) * s);
}
export function quantize(W, bits, gran, g, affine) {
  if (gran === 0) { const flat = quantBlock(W.flat(), bits, affine); return W.map((_, i) => flat.slice(i * C, (i + 1) * C)); }
  const bs = gran === 1 ? C : g;
  return W.map((row) => { const out = []; for (let s = 0; s < C; s += bs) out.push(...quantBlock(row.slice(s, s + bs), bits, affine)); return out; });
}
const fro = (A) => Math.sqrt(A.reduce((s, r) => s + r.reduce((t, v) => t + v * v, 0), 0));
export default {
  params: [
    { id: "bits", label: "Bit sayısı b", min: 2, max: 8, value: 4, int: true },
    { id: "gran", label: "Ölçek kapsamı", options: ["tensör başına", "kanal (satır) başına", "g'lik grup başına"], value: 2 },
    { id: "g", label: "Grup boyu g", options: ["8", "16", "32", "64", "128"], value: 3, on: { gran: [2] } },
    { id: "scheme", label: "Şema", options: ["simetrik", "afin (zero-point)"], value: 0 },
    { id: "data", label: "Veri", options: ["ağırlık (sıfır ortalamalı)", "ReLU sonrası aktivasyon (≥ 0)"], value: 0 },
    { id: "out", label: "Aykırı değer oranı", min: 0, max: 0.02, step: 0.001, value: 0.002 },
    { id: "F", label: "Aykırı büyüklük çarpanı", min: 1, max: 50, value: 10, log: true },
    { id: "spread", label: "Satır ölçeklerinin yayılımı (log₂ std)", min: 0, max: 2, step: 0.1, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const G = [8, 16, 32, 64, 128][q.g];
    const rs = range(R).map(() => 2 ** (q.spread * r.normal()));
    const W = range(R).map((i) => range(C).map(() => { let v = rs[i] * r.normal(); if (q.data) v = Math.max(0, v); return r() < q.out ? v * q.F + (q.data ? Math.abs(v) * q.F : 0) : v; }));
    const X = range(16).map(() => range(C).map(() => r.normal())), mm = (A) => X.map((x) => A.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)));
    const err = (gran, bits, g = G) => { const Q = quantize(W, bits, gran, g, q.scheme === 1); return fro(W.map((row, i) => row.map((v, j) => v - Q[i][j]))) / fro(W); };
    const Q = quantize(W, q.bits, q.gran, G, q.scheme === 1), e = fro(W.map((row, i) => row.map((v, j) => v - Q[i][j]))) / fro(W);
    const Y = mm(W), Yq = mm(Q), eo = fro(Y.map((row, i) => row.map((v, j) => v - Yq[i][j]))) / fro(Y);
    const bs = q.gran === 0 ? R * C : q.gran === 1 ? C : G, bpw = q.bits + (16 * (q.scheme ? 2 : 1)) / bs;
    const bitsR = [2, 3, 4, 5, 6, 7, 8], Gs = [8, 16, 32, 64, 128];
    const P = [`${q.bits} bit, ${["tensör", "kanal", `${G}'lik grup`][q.gran]} başına ölçek, ${q.scheme ? "afin" : "simetrik"}: ağırlık hatası ${fmt(e, 3)}, çıktı hatası ${fmt(eo, 3)}. Bit/ağırlık ${fmt(bpw, 4)} (fp16 ölçek${q.scheme ? " ve zero-point" : ""} dahil; fp16'ya göre ${fmt(16 / bpw, 3)} kat küçük).`];
    if (q.out > 0 && q.F > 3) P.push(`Aykırı değerler bir bloğun max|w|'sini ${q.F} kat büyütür; o bloktaki diğer bütün değerler kaba bir ızgarada kalır. Blok küçüldükçe (tensör → kanal → grup) bir aykırı değerin bozduğu değer sayısı azalır: aynı bitte tensör ${fmt(err(0, q.bits), 3)}, kanal ${fmt(err(1, q.bits), 3)}, grup ${fmt(err(2, q.bits), 3)}.`);
    if (q.data === 1) P.push(q.scheme ? "Afin şema aralığı [0, max]'a yerleştirir; bütün kod değerleri kullanılır." : "Veri hep ≥ 0 iken simetrik şema kodların yarısını (negatifleri) hiç kullanmaz: etkin olarak 1 bit kaybedilir. Afin şemayla karşılaştır.");
    if (q.spread > 0.5 && q.gran === 0) P.push("Satırların ölçekleri farklı: tensör başına tek ölçek, küçük satırları çok kaba bir ızgaraya zorlar. Kanal başına ölçek bunu giderir.");
    return {
      plots: [
        { title: "Ağırlık hatası ve bit sayısı", xlabel: "bit", ylabel: "‖W − Ŵ‖ / ‖W‖", ylog: true, series: [
          { name: "tensör", x: bitsR, y: bitsR.map((b) => err(0, b)), c: 4 }, { name: "kanal", x: bitsR, y: bitsR.map((b) => err(1, b)), c: 1 }, { name: `grup (g = ${G})`, x: bitsR, y: bitsR.map((b) => err(2, b)), c: 0 }], marks: [{ x: q.bits, label: "b" }] },
        { title: `Grup boyu (b = ${q.bits})`, xlabel: "g (log₂)", ylabel: "hata", ylog: true, series: [{ name: "grup başına", x: Gs.map(Math.log2), y: Gs.map((g) => err(2, q.bits, g)), c: 0 }], xticks: Gs.map((g) => [Math.log2(g), String(g)]) },
      ],
      stats: [
        { label: "Ağırlık / çıktı hatası", value: `${fmt(e, 3)} / ${fmt(eo, 3)}` },
        { label: "Bit/ağırlık (ölçekler dahil)", value: bpw },
        { label: "Aynı bitte: tensör / kanal / grup", value: `${fmt(err(0, q.bits), 3)} / ${fmt(err(1, q.bits), 3)} / ${fmt(err(2, q.bits), 3)}` },
      ],
      note: P.join(" "),
    };
  },
};
