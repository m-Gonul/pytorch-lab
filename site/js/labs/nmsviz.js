// M4.9 — Maksimum olmayanı bastırma (NMS) ve Soft-NMS: sentetik bir sahnede her nesnenin etrafında birden çok gürültülü tespit
// ve rastgele yanlış pozitifler; açgözlü NMS (IoU eşiği) ya da Gauss Soft-NMS, ardından skor eşiği. Tek görüntülük AP@0.5 ve
// NMS eşiğine göre AP eğrisi.
import { fmt, range } from "./plot.js";

export const iou = (a, b) => { const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), i = iw * ih; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
export function nms(D, thr) { const o = [...D].sort((a, b) => b.s - a.s), keep = []; for (const d of o) if (keep.every((k) => iou(k.b, d.b) <= thr)) keep.push(d); return keep; }
export function softNms(D, sigma) {
  let rest = D.map((d) => ({ ...d })); const out = [];
  while (rest.length) { let m = 0; rest.forEach((d, i) => { if (d.s > rest[m].s) m = i; }); const top = rest.splice(m, 1)[0]; out.push(top); for (const d of rest) d.s *= Math.exp(-(iou(top.b, d.b) ** 2) / sigma); }
  return out;
}
// tek görüntü AP@t (tüm noktalı, zarf ile): skora göre sırala, IoU ≥ t olan eşleşmemiş en iyi GT ile eşleştir
export function ap(dets, gts, t = 0.5) {
  const o = [...dets].sort((a, b) => b.s - a.s), used = new Set(), P = [], R = []; let tp = 0;
  o.forEach((d, k) => { let best = -1, bi = -1; gts.forEach((g, j) => { const v = iou(d.b, g); if (!used.has(j) && v >= t && v > best) { best = v; bi = j; } }); if (bi >= 0) { used.add(bi); tp++; } P.push(tp / (k + 1)); R.push(tp / gts.length); });
  let a = 0; for (let k = 0; k < P.length; k++) a += (R[k] - (k ? R[k - 1] : 0)) * Math.max(...P.slice(k));
  return { ap: a, tp, n: o.length };
}
export function scene(q, r) {
  const G = [];
  for (let i = 0; i < q.n; i++) {
    const w = 14 + 12 * r(), h = 14 + 12 * r(); let cx, cy;
    if (G.length && r() < q.crowd) { const p = G[G.length - 1], ang = 2 * Math.PI * r(), d = (0.35 + 0.3 * r()) * w; cx = (p[0] + p[2]) / 2 + d * Math.cos(ang); cy = (p[1] + p[3]) / 2 + d * Math.sin(ang); }
    else { cx = 12 + 76 * r(); cy = 12 + 76 * r(); }
    G.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
  }
  const D = [];
  G.forEach((g) => { const w = g[2] - g[0], h = g[3] - g[1]; for (let k = 0; k < 6; k++) { const cx = (g[0] + g[2]) / 2 + 0.1 * w * r.normal(), cy = (g[1] + g[3]) / 2 + 0.1 * h * r.normal(), sw = w * Math.exp(0.1 * r.normal()), sh = h * Math.exp(0.1 * r.normal()), b = [cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2]; D.push({ b, s: Math.min(0.99, Math.max(0.01, 0.9 * iou(b, g) + 0.1 * r())) }); } });
  for (let k = 0; k < q.fp; k++) { const w = 10 + 15 * r(), h = 10 + 15 * r(), x = 100 * r() - w / 2, y = 100 * r() - h / 2; D.push({ b: [x, y, x + w, y + h], s: 0.5 * r() }); }
  return { G, D };
}
const rects = (B) => { const x = [], y = []; for (const b of B) { x.push(b[0], b[2], b[2], b[0], b[0], NaN); y.push(b[1], b[1], b[3], b[3], b[1], NaN); } return { x, y }; };

export default {
  params: [
    { id: "n", label: "Nesne sayısı", min: 1, max: 12, value: 6, int: true },
    { id: "crowd", label: "Kalabalık (bir nesnenin öncekiyle çakışacak yerde olma olasılığı)", min: 0, max: 1, step: 0.05, value: 0.5 },
    { id: "fp", label: "Rastgele yanlış pozitif sayısı", min: 0, max: 30, value: 8, int: true },
    { id: "method", label: "Bastırma", options: ["açgözlü NMS", "Soft-NMS (Gauss)"], value: 0 },
    { id: "thr", label: "NMS IoU eşiği", min: 0.1, max: 0.95, step: 0.05, value: 0.5, on: { method: [0] } },
    { id: "sigma", label: "Soft-NMS σ", min: 0.05, max: 2, step: 0.05, value: 0.5, on: { method: [1] } },
    { id: "smin", label: "Skor eşiği (bastırmadan sonra)", min: 0, max: 0.9, step: 0.05, value: 0.3 },
  ],
  seed: true,
  run(q, r) {
    const { G, D } = scene(q, r), sup = q.method === 0 ? nms(D, q.thr) : softNms(D, q.sigma), kept = sup.filter((d) => d.s >= q.smin);
    const keptSet = new Set(kept.map((d) => d.b)), gone = D.filter((d) => !keptSet.has(d.b));
    const A = ap(kept, G), raw = ap(D.filter((d) => d.s >= q.smin), G), ths = range(17).map((i) => 0.1 + 0.05 * i), curve = ths.map((t) => ap(nms(D, t).filter((d) => d.s >= q.smin), G).ap);
    const soft = ap(softNms(D, q.sigma).filter((d) => d.s >= q.smin), G).ap;
    const rg = rects(G), rk = rects(kept.map((d) => d.b)), rs = rects(gone.map((d) => d.b));
    return {
      plots: [
        { title: "Sahne: gerçek kutular, tutulan ve atılan tespitler", xlabel: "x", ylabel: "y", xlim: [0, 100], ylim: [0, 100], size: { w: 420, h: 400 },
          series: [{ name: "atılan", x: rs.x, y: rs.y, c: 4 }, { name: "gerçek (GT)", x: rg.x, y: rg.y, c: 1, dash: true }, { name: "tutulan", x: rk.x, y: rk.y, c: 0 }] },
        { title: "Tek görüntü AP@0.5 – NMS IoU eşiği", xlabel: "NMS IoU eşiği", ylabel: "AP@0.5", ylim: [0, 1.02], series: [{ name: "açgözlü NMS", x: ths, y: curve, c: 0, dots: true }], marks: [...(q.method === 0 ? [{ x: q.thr, label: "seçili eşik" }] : []), { y: soft, label: `Soft-NMS (σ = ${fmt(q.sigma, 2)}) AP` }] },
      ],
      stats: [
        { label: "Tespit (bastırmadan önce) → tutulan", value: `${D.length} → ${kept.length}` },
        { label: "Doğru pozitif (IoU ≥ 0.5) / gerçek nesne", value: `${A.tp} / ${G.length}` },
        { label: "AP@0.5 (bastırmadan, yalnız skor eşiği)", value: raw.ap },
        { label: "AP@0.5 (seçili bastırmayla)", value: A.ap },
        { label: `AP@0.5 (Soft-NMS, σ = ${fmt(q.sigma, 2)})`, value: soft },
      ],
      note: `Aynı nesne komşu konumlarda tekrar tekrar bulunur; bastırma olmadan yinelenen kutular yanlış pozitif sayılır ve precision'ı düşürür. Açgözlü NMS: en yüksek skorlu kutuyu tut, onunla IoU'su eşiği aşanları sil, kalanlarla tekrarla. ${q.method === 0 ? (q.thr < 0.35 && q.crowd > 0.3 ? "<b>Eşik düşük ve sahne kalabalık:</b> birbirine yakın <b>farklı</b> nesnelerin kutuları da birbirini siliyor — recall düşüyor." : q.thr > 0.75 ? "<b>Eşik yüksek:</b> aynı nesnenin yinelenen kutuları hayatta kalıyor — precision düşüyor." : "Eşik ortada: iki hata türü dengede. AP eğrisinin tepesi sahnenin kalabalığına bağlı.") : "Soft-NMS kutuları silmez, skorlarını IoU ile orantılı küçültür (e^(−IoU²/σ)); kalabalık sahnede yakın bir başka nesnenin kutusu düşük ama sıfır olmayan skorla sıralamada kalır."} Yanlış pozitiflerin çoğunu skor eşiği temizler; NMS yalnız çakışan kutuları ele alır. Bu basit sahnede Soft-NMS'in AP@0.5'te açgözlü NMS'i geçmesi garanti değildir (yinelenen kutular düşük skorla sıralamanın sonunda kalır); Bodla ve ark.'nın bildirdiği kazanç özellikle kalabalık sahnelerde ve COCO'nun yüksek IoU eşiklerinde (AP@[.50:.95]).`,
    };
  },
};
