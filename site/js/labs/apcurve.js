// M4.9 — Precision–recall eğrisi ve ortalama hassasiyet (AP): skorları sıralanmış tespitlerden PR eğrisi, "en az bu recall'daki
// en yüksek precision" zarfı, 11 noktalı AP (VOC 2007), tüm noktalı AP (VOC 2010+) ve 101 noktalı AP (COCO).
import { fmt, range } from "./plot.js";

export function prCurve(tp, nGT) { const P = [], R = []; let c = 0; tp.forEach((t, k) => { c += t; P.push(c / (k + 1)); R.push(c / nGT); }); return { P, R }; }
export function envelope(P) { const E = [...P]; for (let k = E.length - 2; k >= 0; k--) E[k] = Math.max(E[k], E[k + 1]); return E; }
// (r_{k−1}, r_k] aralığında y_k: AP toplamının çizdiği basamaklar
const stair = (R, Y) => { const x = [0], y = [Y[0]]; R.forEach((r, k) => { x.push(r, r); y.push(Y[k], k + 1 < R.length ? Y[k + 1] : Y[k]); }); return { x, y }; };
export const apAll = (P, R) => { const E = envelope(P); let a = 0; for (let k = 0; k < P.length; k++) a += (R[k] - (k ? R[k - 1] : 0)) * E[k]; return a; };
export const apN = (P, R, n) => range(n).reduce((s, i) => { const t = i / (n - 1), ok = P.filter((_, k) => R[k] >= t - 1e-12); return s + (ok.length ? Math.max(...ok) : 0); }, 0) / n;

export default {
  params: [
    { id: "nGT", label: "Gerçek nesne sayısı", min: 2, max: 200, value: 20, int: true, log: true },
    { id: "miss", label: "Hiç tespit edilmeyen nesne oranı (recall tavanı = 1 − oran)", min: 0, max: 0.8, step: 0.05, value: 0.1 },
    { id: "nFP", label: "Yanlış pozitif sayısı", min: 0, max: 400, value: 30, int: true },
    { id: "sep", label: "Skor ayrışması d′ (doğru ile yanlışın skor farkı, std cinsinden)", min: 0, max: 5, step: 0.1, value: 1.5 },
  ],
  seed: true,
  run(q, r) {
    const nTP = Math.round(q.nGT * (1 - q.miss)), D = [...range(nTP).map(() => ({ s: q.sep + r.normal(), t: 1 })), ...range(q.nFP).map(() => ({ s: r.normal(), t: 0 }))].sort((a, b) => b.s - a.s);
    if (!D.length) return { plots: [], stats: [] };
    const { P, R } = prCurve(D.map((d) => d.t), q.nGT), E = envelope(P), aAll = apAll(P, R), a11 = apN(P, R, 11), a101 = apN(P, R, 101);
    let auc = 0; for (let k = 0; k < P.length; k++) auc += (R[k] - (k ? R[k - 1] : 0)) * P[k];
    const r11 = range(11).map((i) => i / 10), p11 = r11.map((t) => { const ok = P.filter((_, k) => R[k] >= t - 1e-12); return ok.length ? Math.max(...ok) : 0; });
    return {
      plots: [{ title: "Precision–recall eğrisi", xlabel: "recall", ylabel: "precision", xlim: [0, 1], ylim: [0, 1.03], series: [
        { name: "ham PR (her eşik)", ...stair(R, P), c: 4 },
        { name: "zarf: en az bu recall'daki en yüksek precision", ...stair(R, E), c: 0 },
        { name: "11 nokta", x: r11, y: p11, kind: "scatter", c: 2, r: 4 }], marks: [{ x: nTP / q.nGT, label: "recall tavanı" }] }],
      stats: [
        { label: "Tüm noktalı AP (VOC 2010+): zarfın altındaki alan", value: aAll },
        { label: "11 noktalı AP (VOC 2007)", value: a11 },
        { label: "101 noktalı AP (COCO)", value: a101 },
        { label: "Ham eğrinin altındaki alan (zarfsız)", value: auc },
        { label: "Ulaşılan en yüksek recall", value: R[R.length - 1] },
      ],
      note: `Tespitler skora göre sıralanır; her eşikte (ilk k tespit) precision = TP/k, recall = TP/(gerçek nesne sayısı). Ham eğri testere dişlidir: her yanlış pozitif precision'ı düşürür, sonraki doğru pozitif yükseltir. Recall arttıkça precision'ın yükseldiği yerlerde gerçek bir ödünleşim yoktur; bu yüzden AP <b>zarfı</b> kullanır: her recall düzeyinde, en az o recall'u veren en yüksek precision. ${q.miss > 0 ? `Hiç bulunamayan nesneler recall'u ${fmt(nTP / q.nGT, 3)}'te keser; bu tavanın ötesindeki noktalarda precision 0 sayılır — tüm noktalı AP en fazla ${fmt(nTP / q.nGT, 3)} olabilir (11 noktalı AP, r = 0 noktasını da saydığı için bu tavanı biraz aşabilir: en fazla (⌊10·tavan⌋ + 1)/11).` : ""} 11 noktalı ve tüm noktalı AP aynı eğrinin iki farklı özetidir; az tespitte aralarındaki fark büyüyebilir (burada ${fmt(Math.abs(a11 - aAll), 3)}). Raporlarken hangisi olduğunu belirtmek gerekir.`,
    };
  },
};
