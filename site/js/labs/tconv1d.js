// M4.10 — 1-B transposed convolution'ın üç eşdeğer hesabı: (1) her girdi örneği çekirdeği kendi değeriyle ölçekleyip çıkışa
// "saçar" (scatter), (2) girdiye adım−1 sıfır ekle + k−1−p dolgu + ters çevrilmiş çekirdekle normal konvolüsyon, (3) konvolüsyon
// matrisinin devriği Cᵀx. Çıkış boyu (n−1)s − 2p + k + output_padding; her çıkış konumuna düşen katkı sayısı (dama tahtası).
import { fmt, range } from "./plot.js";

export function scatter(x, w, s, p, op = 0) {
  const k = w.length, full = (x.length - 1) * s + k + op, out = new Array(full).fill(0);
  x.forEach((xv, i) => w.forEach((wv, u) => { out[i * s + u] += xv * wv; }));
  return out.slice(p, full - p);
}
export function zeroInsert(x, w, s, p, op = 0) {
  const k = w.length, z = [];
  x.forEach((v, i) => { z.push(v); if (i < x.length - 1) for (let t = 0; t < s - 1; t++) z.push(0); });
  const pad = k - 1 - p, zp = [...new Array(pad).fill(0), ...z, ...new Array(pad + op).fill(0)], wf = [...w].reverse();
  return range(zp.length - k + 1).map((j) => wf.reduce((a, wv, u) => a + wv * zp[j + u], 0));
}
// normal konvolüsyon (çapraz korelasyon) y = C z, z ∈ R^m, adım s, dolgu p; Cᵀx transposed conv'dur
export function convMatrix(m, w, s, p) {
  const k = w.length, n = Math.floor((m + 2 * p - k) / s) + 1;
  return range(n).map((i) => range(m).map((j) => { const u = j + p - i * s; return u >= 0 && u < k ? w[u] : 0; }));
}
export function counts(n, k, s, p, op = 0) { const full = (n - 1) * s + k + op, c = new Array(full).fill(0); for (let i = 0; i < n; i++) for (let u = 0; u < k; u++) c[i * s + u]++; return c.slice(p, full - p); }
const KERNELS = ["bilineer (üçgen)", "sabit (hepsi 1)", "rastgele (öğrenilmiş gibi)"];
export function kernel(type, k, s, r) {
  if (type === 0) { const f = Math.floor((k + 1) / 2), c = k % 2 ? f - 1 : f - 0.5; return range(k).map((u) => 1 - Math.abs(u - c) / f); } // FCN'in bilineer başlatması
  if (type === 1) return range(k).map(() => 1);
  return range(k).map(() => r.normal());
}

export default {
  params: [
    { id: "n", label: "Girdi uzunluğu n", min: 2, max: 16, value: 6, int: true },
    { id: "k", label: "Çekirdek boyu k", min: 1, max: 9, value: 3, int: true },
    { id: "s", label: "Adım s (girdi ne kadar 'gerilir')", min: 1, max: 4, value: 2, int: true },
    { id: "p", label: "Dolgu p (çıkışın iki ucundan kırpılır)", min: 0, max: 4, value: 0, int: true },
    { id: "op", label: "output_padding (çıkışın sonuna ek)", min: 0, max: 3, value: 0, int: true },
    { id: "ker", label: "Çekirdek", options: KERNELS, value: 1 },
    { id: "inp", label: "Girdi", options: ["sabit (hepsi 1)", "rampa", "rastgele"], value: 0 },
  ],
  seed: true,
  seedOn: { ker: 2 },
  validate(q) {
    const e = [];
    if (q.op >= q.s) e.push({ level: "error", msg: `output_padding (${q.op}) adımdan (${q.s}) küçük olmalı: yalnız adımlı konvolüsyonda kaybolan son birkaç konumu geri eklemek içindir (PyTorch aynı kuralı uygular).` });
    if ((q.n - 1) * q.s + q.k + q.op - 2 * q.p < 1) e.push({ level: "error", msg: "Dolgu çok büyük: çıkış boyu 1'den küçük olur." });
    if (q.p > q.k - 1) e.push({ level: "warn", msg: `p = ${q.p} > k − 1 = ${q.k - 1}: 'sıfır ekle + konvolüsyon' görünümünde dolgu negatif olur (girdi kırpılır); saçma görünümü yine geçerli.` });
    return e;
  },
  run(q, r) {
    const x = q.inp === 0 ? range(q.n).map(() => 1) : q.inp === 1 ? range(q.n).map((i) => i + 1) : range(q.n).map(() => r.normal()), w = kernel(q.ker, q.k, q.s, r);
    const y = scatter(x, w, q.s, q.p, q.op), c = counts(q.n, q.k, q.s, q.p, q.op), L = y.length;
    let zi = null; if (q.p <= q.k - 1) zi = zeroInsert(x, w, q.s, q.p, q.op);
    // Cᵀ: m = L girdi uzunluklu bir konvolüsyonun (adım s, dolgu p) devriği; çıkış n'ye eşitse x uzunluğu tutar
    const C = convMatrix(L, w, q.s, q.p), ct = C.length === q.n ? range(L).map((j) => C.reduce((a, row, i) => a + row[j] * x[i], 0)) : null;
    const d1 = zi ? Math.max(...y.map((v, i) => Math.abs(v - zi[i]))) : NaN, d2 = ct ? Math.max(...y.map((v, i) => Math.abs(v - ct[i]))) : NaN;
    const a = Math.max(0, q.k - 1 - q.p), b = L - Math.max(0, q.k - 1 - q.p + q.op), inner = b > a ? c.slice(a, b) : c, mx = Math.max(...inner), mn = Math.min(...inner);
    return {
      plots: [
        { title: "Girdi x ve çıkış y (transposed conv)", xlabel: "konum", ylabel: "değer", series: [{ name: "girdi x (konumları s ile ölçeklenmiş: i·s − p)", x: x.map((_, i) => i * q.s - q.p), y: x, kind: "scatter", c: 2, r: 5 }, { name: "çıkış y", x: range(L), y, c: 0, dots: true }] },
        { title: "Her çıkış konumuna düşen girdi katkısı sayısı", xlabel: "çıkış konumu", ylabel: "katkı sayısı", yzero: true, ylim: [0, Math.max(...c) + 1], series: [{ name: "katkı sayısı", x: range(L), y: c, kind: "bar", c: 3 }] },
      ],
      stats: [
        { label: "Çıkış boyu: sayılan / formül (n−1)s − 2p + k + op", value: `${L} / ${(q.n - 1) * q.s - 2 * q.p + q.k + q.op}` },
        { label: "max |saçma − (sıfır ekle + konv.)|", value: zi ? (d1 < 1e-12 ? "0 (aynı)" : d1) : "— (p > k − 1)" },
        { label: "max |saçma − Cᵀx|", value: ct ? (d2 < 1e-12 ? "0 (aynı)" : d2) : `— (C, ${L} → ${C.length}; output_padding Cᵀ'nin girdisini değiştirir)` },
        { label: "İç bölgede katkı sayısı (en çok / en az)", value: `${mx} / ${mn}` },
      ],
      note: `Saçma görünümü: her girdi örneği x_i, çekirdeği kendisiyle ölçekleyip çıkışta i·s konumundan başlayarak ekler; örtüşen yerlerde katkılar toplanır. Aynı sonuç, girdiye (s − 1) sıfır ekleyip k − 1 − p dolgu ve ters çevrilmiş çekirdekle normal konvolüsyondan, ya da aynı ayarlı (adım s, dolgu p) normal konvolüsyonun matrisinin devriğinden gelir. ${q.k % q.s !== 0 && q.s > 1 ? `<b>k = ${q.k}, s = ${q.s}'ye bölünmüyor:</b> çıkış konumlarına ${mx} ve ${mn} katkı dönüşümlü düşüyor — sabit girdide bile çıkış dalgalanır (dama tahtası artefaktı). Öğrenilen çekirdek bunu telafi edebilir ama genelde tam edemez (Odena ve ark. 2016).` : q.s > 1 ? "k, s'nin katı: iç bölgede her çıkış konumuna eşit sayıda katkı düşüyor; sabit girdide (sabit çekirdekle) çıkış iç bölgede düz." : "Adım 1: transposed conv, tam dolgulu normal bir konvolüsyona eşdeğer (boyut k − 1 büyür)."}`,
    };
  },
};
