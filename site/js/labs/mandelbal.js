// M10.1 — Yük dengeleme: Mandelbrot kümesinin satırlarını p thread'e dağıtmak. Satır maliyeti = satırdaki piksellerin kaçış iterasyonları
// toplamı (gerçekten hesaplanır, 96 × 128 ızgara, [−2, 0.6] × [−1.2, 1.2]). Dağıtımlar: blok (ardışık ⌈H/p⌉ satır), döngüsel (satır i →
// thread i mod p), döngüsel blok (c satırlık parçalar döngüsel), dinamik (boşalan thread sıradaki c satırlık parçayı alır; her alışta
// sabit ek yük o). Süre = en çok yüklü thread'in toplamı (dinamikte olay simülasyonu). Hızlanma = toplam iş / süre.
import { fmt, range } from "./plot.js";

const H = 96, W = 128;
export function rowCosts(maxit) {
  return range(H).map((y) => { let s = 0; const ci = -1.2 + (2.4 * (y + 0.5)) / H; for (let x = 0; x < W; x++) { const cr = -2 + (2.6 * (x + 0.5)) / W; let zr = 0, zi = 0, k = 0; while (k < maxit && zr * zr + zi * zi <= 4) { const t = zr * zr - zi * zi + cr; zi = 2 * zr * zi + ci; zr = t; k++; } s += k + 1; } return s; });
}
export function distribute(cost, p, mode, c, o) {
  const load = new Float64Array(p), owner = new Int32Array(H);
  if (mode === 0) { const b = Math.ceil(H / p); for (let y = 0; y < H; y++) { owner[y] = Math.min(p - 1, Math.floor(y / b)); load[owner[y]] += cost[y]; } }
  else if (mode === 1) for (let y = 0; y < H; y++) { owner[y] = y % p; load[y % p] += cost[y]; }
  else if (mode === 2) for (let y = 0; y < H; y++) { owner[y] = Math.floor(y / c) % p; load[owner[y]] += cost[y]; }
  else { // dinamik: en erken boşalan thread sıradaki parçayı alır
    for (let y0 = 0; y0 < H; y0 += c) { let q = 0; for (let k = 1; k < p; k++) if (load[k] < load[q]) q = k; load[q] += o; for (let y = y0; y < Math.min(H, y0 + c); y++) { owner[y] = q; load[q] += cost[y]; } }
  }
  return { load, owner, T: Math.max(...load) };
}
export default {
  params: [
    { id: "mode", label: "Dağıtım", options: ["blok", "döngüsel", "döngüsel blok (c satır)", "dinamik (c satırlık parçalar)"], value: 0 },
    { id: "p", label: "Thread sayısı p", min: 1, max: 32, value: 8, int: true },
    { id: "c", label: "Parça boyu c (satır)", min: 1, max: 24, value: 4, int: true, on: { mode: [2, 3] } },
    { id: "o", label: "Dinamik: parça başına ek yük (ortalama satır maliyetinin katı)", min: 0, max: 2, step: 0.05, value: 0.1, on: { mode: [3] } },
    { id: "maxit", label: "En çok iterasyon", min: 20, max: 1000, value: 200, int: true, log: true },
  ],
  run(q) {
    const cost = rowCosts(q.maxit), tot = cost.reduce((a, b) => a + b, 0), mean = tot / H, o = q.o * mean;
    const D = distribute(cost, q.p, q.mode, q.c, o), sp = tot / D.T, imb = D.T / (D.load.reduce((a, b) => a + b, 0) / q.p);
    const ps = range(32).map((k) => k + 1), curve = (m) => ps.map((p) => tot / distribute(cost, p, m, q.c, o).T);
    const names = ["blok", "döngüsel", "döngüsel blok", "dinamik"];
    const P = [`${names[q.mode]} dağıtım, p = ${q.p}: hızlanma ${fmt(sp, 3)} (ideal ${q.p}), dengesizlik (en yüklü / ortalama) ${fmt(imb, 3)}. En pahalı satır ${fmt(Math.max(...cost) / mean, 3)} × ortalama, en ucuzu ${fmt(Math.min(...cost) / mean, 3)} ×.`];
    if (q.mode === 0) P.push("Pahalı satırlar (kümenin içi, maxit iterasyon) ortada toplanıyor: ortadaki thread'ler yükü taşırken kenardakiler erken bitiyor.");
    if (q.mode >= 2) { let mc = 0; for (let y0 = 0; y0 < H; y0 += q.c) { let a = q.mode === 3 ? o : 0; for (let y = y0; y < Math.min(H, y0 + q.c); y++) a += cost[y]; mc = Math.max(mc, a); } P.push(`En pahalı ${q.c} satırlık parça tek bir thread'e düşer: hızlanma, thread sayısından bağımsız olarak toplam iş / bu parça = ${fmt(tot / mc, 3)} ile sınırlı.`); }
    if (q.mode >= 2 && Math.ceil(H / q.c) < 2 * q.p) P.push(`Parça sayısı (${Math.ceil(H / q.c)}) thread sayısının iki katından az: bazı thread'ler boşta kalıyor ya da son parçalar dengesizlik yaratıyor.`);
    if (q.mode === 3 && q.o > 0.5) P.push("Parça başına ek yük büyük: küçük c ile sık alış maliyeti hızlanmayı yiyor; c'yi büyütmek ek yükü azaltır ama dengeyi bozar.");
    return {
      plots: [
        { title: "Satır maliyeti (iterasyon toplamı)", xlabel: "satır", ylabel: "maliyet / ortalama", side: true, size: { w: 300, h: 230 }, series: [{ name: "maliyet", x: range(H), y: cost.map((v) => v / mean), kind: "bar", c: 0 }] },
        { title: "Thread yükleri", xlabel: "thread", ylabel: "yük / ideal", side: true, size: { w: 300, h: 230 }, series: [{ name: "yük", x: range(q.p), y: Array.from(D.load, (v) => (v * q.p) / tot), kind: "bar", c: 1 }] },
        { title: "Hızlanma ve thread sayısı", xlabel: "p", ylabel: "hızlanma", wide: true, size: { w: 620, h: 210 }, series: [{ name: "ideal", x: ps, y: ps, c: 4, dash: true }, ...[0, 1, 2, 3].map((m) => ({ name: names[m], x: ps, y: curve(m), c: m }))], marks: [{ x: q.p, label: "seçili p" }] },
      ],
      stats: [{ label: "Hızlanma", value: fmt(sp, 3) }, { label: "Dengesizlik (max / ortalama)", value: fmt(imb, 3) }, { label: "Verim", value: `${fmt((100 * sp) / q.p, 3)}%` }],
      note: P.join(" "),
    };
  },
};
