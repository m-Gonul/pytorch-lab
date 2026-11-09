// M4.10 — Bölütleme ölçütleri bir arada: piksel doğruluğu, ön plan IoU'su, arka plan IoU'su, mIoU ve Dice. Gerçek maske bir disk;
// tahmin kaydırılmış/ölçeklenmiş, sınırı gürültülü bir disk. Küçük nesnede piksel doğruluğunun neden yanıltıcı olduğu ve Dice = 2J/(1+J).
import { fmt, range } from "./plot.js";

export function masks(q, r, N = 64) {
  const ph = range(16).map(() => r() * 2 * Math.PI), amp = range(16).map((k) => (q.noise * r.normal()) / (1 + k)), cx = N / 2, cy = N / 2, R = q.R, Rp = q.R * q.sc;
  const G = [], P = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    G.push(Math.hypot(j + 0.5 - cx, i + 0.5 - cy) <= R ? 1 : 0);
    const dx = j + 0.5 - cx - q.dx, dy = i + 0.5 - cy, th = Math.atan2(dy, dx), rr = Rp * (1 + amp.reduce((s, a, k) => s + a * Math.cos((k + 1) * th + ph[k]), 0));
    P.push(Math.hypot(dx, dy) <= rr ? 1 : 0);
  }
  return { G, P, N };
}
export function metrics(G, P) {
  let tp = 0, fp = 0, fn = 0, tn = 0; G.forEach((g, k) => { if (g && P[k]) tp++; else if (P[k]) fp++; else if (g) fn++; else tn++; });
  const iouF = tp / Math.max(1, tp + fp + fn), iouB = tn / Math.max(1, tn + fp + fn), dice = (2 * tp) / Math.max(1, 2 * tp + fp + fn);
  return { tp, fp, fn, tn, acc: (tp + tn) / G.length, iouF, iouB, miou: (iouF + iouB) / 2, dice };
}

export default {
  params: [
    { id: "R", label: "Nesne yarıçapı (64 × 64 görüntüde, piksel)", min: 2, max: 28, value: 6, step: 0.5 },
    { id: "dx", label: "Tahminin kayması (piksel)", min: 0, max: 20, step: 0.5, value: 3 },
    { id: "sc", label: "Tahmin yarıçapı / gerçek yarıçap", min: 0.3, max: 2, step: 0.05, value: 1 },
    { id: "noise", label: "Sınır gürültüsü", min: 0, max: 0.6, step: 0.02, value: 0.1 },
  ],
  seed: true,
  seedOn: { noise: 0.02 },
  run(q, r) {
    const { G, P, N } = masks(q, r), M = metrics(G, P), js = range(101).map((i) => i / 100);
    const img = G.map((g, k) => (g && P[k] ? 1 : P[k] ? 0.65 : g ? 0.35 : 0));
    return {
      plots: [
        { title: "Beyaz TP · açık gri FP · koyu gri FN · siyah TN", img: { w: N, h: N, data: img }, side: true, size: { w: 330, h: 320 } },
        { title: "Dice – IoU (aynı maske çifti için)", xlabel: "IoU (Jaccard)", ylabel: "Dice", xlim: [0, 1], ylim: [0, 1], series: [{ name: "Dice = 2J / (1 + J)", x: js, y: js.map((j) => (2 * j) / (1 + j)), c: 0 }, { name: "bu tahmin", x: [M.iouF], y: [M.dice], kind: "scatter", c: 2, r: 6 }, { name: "Dice = J", x: [0, 1], y: [0, 1], c: 4, dash: true }] },
      ],
      stats: [
        { label: "Piksel doğruluğu", value: M.acc },
        { label: "IoU (ön plan) = TP / (TP + FP + FN)", value: M.iouF },
        { label: "IoU (arka plan)", value: M.iouB },
        { label: "mIoU (iki sınıfın ortalaması)", value: M.miou },
        { label: "Dice = 2TP / (2TP + FP + FN)", value: M.dice },
        { label: "Nesnenin görüntüdeki payı", value: `%${fmt((100 * (M.tp + M.fn)) / G.length, 3)}` },
      ],
      note: `${M.acc > 0.95 && M.iouF < 0.5 ? `<b>Piksel doğruluğu yanıltıcı:</b> %${fmt(100 * M.acc, 3)} doğru, ama nesnenin IoU'su yalnız ${fmt(M.iouF, 3)}. Nesne görüntünün küçük bir kısmı olduğundan "her şey arka plan" tahmini bile yüksek doğruluk alır. ` : ""}IoU ve Dice yalnız TP, FP, FN'e bakar (TN'i yok sayar); bu yüzden küçük nesnelerde anlamlıdır. İkisi aynı sıralamayı verir: Dice = 2J/(1 + J) tekdüze bir dönüşüm (Dice ≥ IoU, eşitlik yalnız 0 ve 1'de). mIoU, her sınıfın IoU'sunun ortalaması: büyük arka plan sınıfının IoU'su yüksek olduğu için mIoU iki sınıflı problemde ön plan IoU'sundan iyimserdir. Küçük nesnede birkaç piksellik kayma IoU'yu çok düşürür; büyük nesnede aynı kayma az etkiler (yarıçapı değiştir).`,
    };
  },
};
