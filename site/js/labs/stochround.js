// M7.3 — Saf bf16 eğitimde yuvarlama: w ← round(w − η g). En yakına yuvarlamada |η g| < ulp(w)/2 olan güncellemeler kaybolur ve w
// takılır. Stokastik yuvarlama: w − ηg iki komşu bf16 sayısı arasında, uzaklıklarıyla ters orantılı olasılıkla birine yuvarlanır;
// beklentide yansızdır (E[SR(x)] = x). Problem: f(w) = ½(w − w*)², g = w − w*. fp32 başvurusu da çizilir.
import { fmt, range } from "./plot.js";
import { roundTo, ulp, FMTS } from "./floatfmt.js";

const BF = FMTS[2], F16 = FMTS[3];
export function sr(x, F, r) {
  // alt komşu lo ≤ x ve üst komşu hi > x; 2'nin kuvveti sınırlarında aralık değişir, bu yüzden ulp sınırın doğru tarafından okunur
  const n = roundTo(x, F), eps = (v) => Math.abs(v) * 1e-12 || 1e-300;
  const lo = n <= x ? n : n - ulp(n - eps(n), F);
  if (lo === x) return lo;
  const hi = lo + ulp(lo + eps(lo), F), p = (x - lo) / (hi - lo);
  return r() < p ? roundTo(hi, F) : roundTo(lo, F);
}
export default {
  params: [
    { id: "fmt", label: "Ağırlık biçimi", options: ["bf16 (m7)", "fp16 (m10)"], value: 0 },
    { id: "w0", label: "Başlangıç w₀", min: 1, max: 100, value: 10, log: true },
    { id: "target", label: "Hedef w*", min: 0, max: 10, step: 0.1, value: 9 },
    { id: "lr", label: "Öğrenme oranı η", min: 1e-5, max: 0.1, value: 0.001, log: true },
    { id: "steps", label: "Adım", min: 100, max: 20000, value: 3000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const F = q.fmt ? F16 : BF; let a = roundTo(q.w0, F), b = a, c = q.w0; const A = [a], B = [b], C = [c];
    for (let t = 0; t < q.steps; t++) { a = roundTo(a - q.lr * (a - q.target), F); b = sr(b - q.lr * (b - q.target), F, r); c = c - q.lr * (c - q.target); A.push(a); B.push(b); C.push(c); }
    const stall = Math.abs(A.at(-1) - C.at(-1)) > 2 * ulp(C.at(-1) || 1, F);
    const gap = ulp(q.w0, F) / 2, firstUpd = q.lr * Math.abs(q.w0 - q.target);
    const P = [`${F.n}: w₀ = ${q.w0} civarında komşu sayılar arası ${fmt(ulp(q.w0, F), 3)}; ilk güncelleme η|g| = ${fmt(firstUpd, 3)}${firstUpd < gap ? ", yarım aralıktan küçük" : ""}.`];
    P.push(`Son değerler: en yakına ${fmt(A.at(-1), 5)}, stokastik ${fmt(B.at(-1), 5)}, fp32 ${fmt(C.at(-1), 5)} (hedef ${q.target}).`);
    P.push(stall ? `En yakına yuvarlama <b>takıldı</b>: η|w − w*| < ulp(w)/2 olduğu andan itibaren her güncelleme aynı sayıya geri yuvarlanır. Stokastik yuvarlama küçük güncellemeleri olasılıkla uygular; beklentide fp32 yörüngesini izler (gürültüyle).` : "En yakına yuvarlama da hedefe ulaştı: güncellemeler bu ölçekte temsil edilebiliyor.");
    const ts = range(q.steps + 1), stride = Math.max(1, Math.floor(ts.length / 600)), sub = (v) => v.filter((_, i) => i % stride === 0);
    return {
      plots: [{ title: "w'nin yörüngesi", xlabel: "adım", ylabel: "w", wide: true, size: { w: 620, h: 240 }, series: [
        { name: "fp32", x: sub(ts), y: sub(C), c: 4, dash: true }, { name: "en yakına", x: sub(ts), y: sub(A), c: 1 }, { name: "stokastik", x: sub(ts), y: sub(B), c: 0 }], marks: [{ y: q.target, label: "w*" }] }],
      stats: [
        { label: "Son w: en yakına / stokastik / fp32", value: `${fmt(A.at(-1), 4)} / ${fmt(B.at(-1), 4)} / ${fmt(C.at(-1), 4)}` },
        { label: "ulp(w₀) / ilk güncelleme", value: `${fmt(ulp(q.w0, F), 3)} / ${fmt(firstUpd, 3)}` },
        { label: "En yakına takıldı mı?", value: stall ? "evet" : "hayır" },
      ],
      note: P.join(" "),
    };
  },
};
