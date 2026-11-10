// M7.5 — Büyüklüğe göre budama: W (64 × 256), ağırlık dağılımı Gauss / Laplace / aykırılı. Yöntemler: yapısız global (en küçük |w|'ler),
// satır başına (her satırda aynı oran), 2:4 yapılı (her 4'lük grupta en büyük 2 kalır; seyreklik %50 sabit), nöron (bütün satırı
// L2 normuna göre at). Ölçüt: rastgele girdiyle çıktının göreli hatası ve atılan ağırlık enerjisinin oranı.
import { fmt, range } from "./plot.js";

const R = 64, C = 256, M = ["yapısız (global)", "satır başına", "2:4 yapılı", "nöron (satır) budama"];
function prune(W, sp, m) {
  if (m === 0) { const a = W.flat().map(Math.abs).sort((x, y) => x - y), t = a[Math.min(a.length - 1, Math.floor(sp * a.length))]; return W.map((r) => r.map((v) => (Math.abs(v) < t || (sp >= 1) ? 0 : v))); }
  if (m === 1) return W.map((r) => { const a = r.map(Math.abs).sort((x, y) => x - y), t = a[Math.min(C - 1, Math.floor(sp * C))]; return r.map((v) => (Math.abs(v) < t ? 0 : v)); });
  if (m === 2) return W.map((r) => r.map((v, j) => { const g = r.slice(j - (j % 4), j - (j % 4) + 4).map(Math.abs).sort((x, y) => y - x); return Math.abs(v) >= g[1] ? v : 0; }));
  const nrm = W.map((r) => Math.hypot(...r)), t = [...nrm].sort((x, y) => x - y)[Math.min(R - 1, Math.floor(sp * R))]; return W.map((r, i) => (nrm[i] < t ? r.map(() => 0) : r));
}
export default {
  params: [
    { id: "sp", label: "Seyreklik (atılan oran)", min: 0, max: 0.95, step: 0.05, value: 0.5 },
    { id: "m", label: "Yöntem", options: M, value: 0 },
    { id: "dist", label: "Ağırlık dağılımı", options: ["Gauss", "Laplace (ağır kuyruk)", "Gauss + %1 aykırı (× 10)"], value: 1 },
    { id: "rowsp", label: "Satır ölçeklerinin yayılımı (log₂ std)", min: 0, max: 2, step: 0.1, value: 0.5 },
  ],
  seed: true,
  run(q, r) {
    const lap = () => { const u = r() - 0.5; return -Math.sign(u) * Math.log(1 - 2 * Math.abs(u)) / Math.SQRT2; };
    const rs = range(R).map(() => 2 ** (q.rowsp * r.normal()));
    const W = range(R).map((i) => range(C).map(() => { let v = q.dist === 1 ? lap() : r.normal(); if (q.dist === 2 && r() < 0.01) v *= 10; return rs[i] * v; }));
    const X = range(32).map(() => range(C).map(() => r.normal())), Y = X.map((x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)));
    const ev = (P) => { const Yp = X.map((x) => P.map((row) => row.reduce((s, w, j) => s + w * x[j], 0))); const num = Y.reduce((s, row, i) => s + row.reduce((t, v, j) => t + (v - Yp[i][j]) ** 2, 0), 0), den = Y.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0); return Math.sqrt(num / den); };
    const energy = (P) => { const a = W.flat().reduce((s, v) => s + v * v, 0), b = P.flat().reduce((s, v) => s + v * v, 0); return 1 - b / a; };
    const sp = q.m === 2 ? 0.5 : q.sp, Pm = prune(W, sp, q.m), e = ev(Pm), en = energy(Pm), nz = Pm.flat().filter((v) => v !== 0).length / (R * C);
    const sps = range(19).map((i) => i * 0.05), curves = [0, 1, 3].map((m) => sps.map((s) => ev(prune(W, s, m)))), e24 = ev(prune(W, 0.5, 2));
    const P = [`${M[q.m]}, seyreklik ${fmt(1 - nz, 3)}: çıktı hatası ${fmt(e, 3)}, atılan ağırlık enerjisi ${fmt(en, 3)}.${q.m === 2 ? " 2:4 yapısı seyrekliği %50'ye sabitler; seyreklik kontrolü bu yöntemde etkisiz." : ""}`];
    P.push(`Ağır kuyruklu dağılımlarda enerjinin çoğu az sayıda büyük ağırlıktadır; küçükleri atmak az enerji kaybettirir. %50'de: yapısız ${fmt(ev(prune(W, 0.5, 0)), 3)}, satır başına ${fmt(ev(prune(W, 0.5, 1)), 3)}, 2:4 ${fmt(e24, 3)}, nöron ${fmt(ev(prune(W, 0.5, 3)), 3)}.`);
    P.push("Yapısız seyreklik en az hatayı verir ama donanımda hızlandırması zordur (düzensiz bellek erişimi); 2:4, NVIDIA Ampere ve sonrası tensör çekirdeklerinde ≈ 2 kat matmul hızı verir; nöron budama matrisi gerçekten küçültür ama en çok hatayı getirir. Uygulamada budamadan sonra ince ayarla hata geri kazanılır.");
    return {
      plots: [{ title: "Çıktı hatası ve seyreklik", xlabel: "seyreklik", ylabel: "göreli çıktı hatası", ylim: [0, 1.05], wide: true, size: { w: 620, h: 240 }, series: [
        { name: "yapısız (global)", x: sps, y: curves[0], c: 0 }, { name: "satır başına", x: sps, y: curves[1], c: 2 }, { name: "nöron", x: sps, y: curves[2], c: 1 },
        { name: "2:4 (yalnız %50)", x: [0.5], y: [e24], kind: "scatter", c: 3, r: 6 }], marks: [{ x: sp, label: "seçili" }] }],
      stats: [
        { label: "Seyreklik (gerçekleşen)", value: 1 - nz },
        { label: "Çıktı hatası", value: e },
        { label: "Atılan ağırlık enerjisi", value: en },
      ],
      note: P.join(" "),
    };
  },
};
