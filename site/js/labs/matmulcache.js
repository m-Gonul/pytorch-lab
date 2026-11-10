// M10.3 — Matris çarpımı C = A·B (n × n, fp32, satır öncelikli) için bellek erişim izinin önbellek simülasyonu. Önbellek: S KB, 64 baytlık
// satır, 8 yollu set-ilişkili, LRU. Döngü sıraları: ijk (C[i][j] yazmaçta biriktirilir; B sütun boyunca yürür), ikj (A[i][k] yazmaçta;
// B ve C satır boyunca), jki (A ve C sütun boyunca), devrik B (Bᵀ önceden hazır; ijk ama iki satır boyunca), karolu ikj (b × b bloklar).
// Kaba süre modeli: n³ FMA (1 çevrim) + ıska sayısı × ıska cezası.
import { fmt, range } from "./plot.js";

function makeCache(kb, ways = 8) {
  const lines = (kb * 1024) / 64, sets = Math.max(1, Math.floor(lines / ways)), tag = new Float64Array(sets * ways).fill(-1), age = new Float64Array(sets * ways);
  let t = 0, miss = 0, acc = 0;
  return {
    access(addr) { acc++; const ln = Math.floor(addr / 64), s = ln % sets, b = s * ways; t++; let lru = b; for (let w = b; w < b + ways; w++) { if (tag[w] === ln) { age[w] = t; return; } if (age[w] < age[lru]) lru = w; } miss++; tag[lru] = ln; age[lru] = t; },
    get miss() { return miss; }, get acc() { return acc; },
  };
}
export function simulate(n, order, b, kb) {
  const C0 = makeCache(kb), A = 0, B = n * n * 4 + 4096, C = 2 * (n * n * 4 + 4096), BT = 3 * (n * n * 4 + 4096), ad = (base, i, j) => base + 4 * (i * n + j);
  if (order === 0) { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { for (let k = 0; k < n; k++) { C0.access(ad(A, i, k)); C0.access(ad(B, k, j)); } C0.access(ad(C, i, j)); } }
  else if (order === 1) { for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { C0.access(ad(A, i, k)); for (let j = 0; j < n; j++) { C0.access(ad(B, k, j)); C0.access(ad(C, i, j)); } } }
  else if (order === 2) { for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) { C0.access(ad(B, k, j)); for (let i = 0; i < n; i++) { C0.access(ad(A, i, k)); C0.access(ad(C, i, j)); } } }
  else if (order === 3) { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { for (let k = 0; k < n; k++) { C0.access(ad(A, i, k)); C0.access(ad(BT, j, k)); } C0.access(ad(C, i, j)); } }
  else { for (let ii = 0; ii < n; ii += b) for (let kk = 0; kk < n; kk += b) for (let jj = 0; jj < n; jj += b) for (let i = ii; i < Math.min(n, ii + b); i++) for (let k = kk; k < Math.min(n, kk + b); k++) { C0.access(ad(A, i, k)); for (let j = jj; j < Math.min(n, jj + b); j++) { C0.access(ad(B, k, j)); C0.access(ad(C, i, j)); } } }
  return { miss: C0.miss, acc: C0.acc };
}
const NAMES = ["ijk", "ikj", "jki", "devrik B (ijk)", "karolu ikj"];
export default {
  params: [
    { id: "order", label: "Döngü sırası", options: NAMES, value: 0 },
    { id: "n", label: "Matris boyu n", min: 16, max: 96, step: 4, value: 64, int: true },
    { id: "b", label: "Karo boyu b", min: 4, max: 48, value: 16, int: true, on: { order: [4] } },
    { id: "kb", label: "Önbellek boyu (KB)", min: 4, max: 64, value: 16, int: true, log: true },
    { id: "pen", label: "Iska cezası (çevrim)", min: 10, max: 300, value: 100, int: true },
  ],
  run(q) {
    const n = q.n, F = n ** 3, res = NAMES.map((_, o) => simulate(n, o, q.b, q.kb)), cur = res[q.order];
    const t = (r) => F + r.miss * q.pen, best = Math.min(...res.map(t));
    const ws = (3 * n * n * 4) / 1024;
    const P = [`${NAMES[q.order]}: ${cur.acc} erişimde ${cur.miss} ıska (FMA başına ${fmt(cur.miss / F, 3)}); tahmini süre ${fmt(t(cur), 4)} çevrim, en iyi sıranın ${fmt(t(cur) / best, 3)} katı.`];
    P.push(`Üç matrisin toplamı ${fmt(ws, 3)} KB, önbellek ${q.kb} KB.${ws <= q.kb ? " Hepsi önbelleğe sığıyor: sıranın etkisi küçük (yalnız ilk yükleme ıskaları)." : " Sığmıyor: erişim deseni belirleyici."}`);
    if (q.order === 0) P.push("ijk'de iç döngü B'nin sütununda yürüyor: her erişim farklı bir satıra düşüyor (adım 4n bayt), bir satırdan yalnız 4 bayt kullanılıyor.");
    if (q.order === 4) { const tw = (3 * q.b * q.b * 4) / 1024; P.push(`Üç b × b karonun toplamı ${fmt(tw, 3)} KB (önbellek ${q.kb} KB); karolu sürüm ikj'nin ${fmt(res[1].miss / Math.max(1, cur.miss), 3)} katı daha az ıska veriyor.`); }
    if ((q.order === 0 || q.order === 2) && (n & (n - 1)) === 0 && ws > q.kb) P.push(`n = ${n} 2'nin kuvveti: bir sütundaki ardışık erişimler 4n baytlık adımla aynı birkaç sete düşüyor ve 8 yol yetmiyor (çakışma ıskaları). n'yi 4 artırıp azaltmayı dene.`);
    return {
      plots: [{ title: "Iska / FMA (bütün sıralar)", xlabel: "döngü sırası", ylabel: "ıska / FMA", wide: true, size: { w: 620, h: 220 }, xlim: [-0.6, 4.6], xticks: NAMES.map((nm, i) => [i, nm]), series: [{ name: "ıska / FMA", x: range(5), y: res.map((r) => r.miss / F), kind: "bar", c: 0 }], marks: [{ x: q.order, label: "seçili" }] }],
      stats: [{ label: "Iska / FMA", value: fmt(cur.miss / F, 3) }, { label: "Iska oranı", value: `${fmt((100 * cur.miss) / cur.acc, 3)}%` }, { label: "Süre / en iyi sıra", value: fmt(t(cur) / best, 3) }],
      note: P.join(" "),
    };
  },
};
