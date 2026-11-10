// M7.6 — Dinamik batch'li çıkarım sunucusu (olay benzetimi). İstekler Poisson(λ) gelir. Tek hızlandırıcı, batch süresi t(B) = t₀ + c·B
// (varsayılan t(1) = 5 ms, t(8) = 20 ms). Sunucu boşken kuyruk Bmax'a ulaşınca ya da en eski istek w ms beklemişse en fazla Bmax
// isteği alır. Gecikme = tamamlanma − varış. Kararlılık koşulu: λ < Bmax / t(Bmax).
import { fmt, range } from "./plot.js";

export function simulate(q, r, lam) {
  const T = q.T, arr = []; let t = 0; while (true) { t += -Math.log(1 - r()) / lam; if (t > T) break; arr.push(t); }
  const tb = (B) => (q.t0 + q.c * B) / 1000, w = q.wait / 1000; let free = 0, i = 0; const lat = [], bs = []; let busy = 0;
  while (i < arr.length) {
    const start0 = Math.max(free, arr[i]);
    // Bmax'a ulaşma anı ya da en eski isteğin bekleme sınırı, hangisi önce
    const fillT = i + q.Bmax - 1 < arr.length ? arr[i + q.Bmax - 1] : Infinity, start = Math.max(start0, Math.min(fillT, arr[i] + w));
    let j = i; while (j < arr.length && arr[j] <= start && j - i < q.Bmax) j++;
    const B = Math.max(1, j - i), d = tb(B); for (let k = i; k < i + B; k++) lat.push(start + d - arr[k]);
    bs.push(B); busy += d; free = start + d; i += B;
  }
  lat.sort((a, b) => a - b);
  const pct = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] * 1000;
  return { lat, p50: pct(0.5), p99: pct(0.99), thr: lat.length / Math.max(free, T), util: busy / Math.max(free, T), mb: bs.reduce((a, b) => a + b, 0) / bs.length, n: arr.length };
}
export default {
  params: [
    { id: "lam", label: "İstek hızı λ (istek/s)", min: 10, max: 2000, value: 300, log: true },
    { id: "Bmax", label: "En büyük batch Bmax", min: 1, max: 64, value: 8, int: true },
    { id: "wait", label: "En uzun bekleme w (ms)", min: 0, max: 50, step: 0.5, value: 5 },
    { id: "t0", label: "Batch başına sabit süre t₀ (ms)", min: 0.1, max: 20, step: 0.01, value: 2.86 },
    { id: "c", label: "Örnek başına süre c (ms)", min: 0.05, max: 10, step: 0.01, value: 2.14 },
    { id: "T", label: "Benzetim süresi (s)", min: 1, max: 20, value: 5, int: true },
  ],
  seed: true,
  run(q, r) {
    const S = simulate(q, r, q.lam), cap = q.Bmax / ((q.t0 + q.c * q.Bmax) / 1000), cap1 = 1 / ((q.t0 + q.c) / 1000);
    const lams = range(12).map((i) => 10 * 1.6 ** i).filter((l) => l <= 2000);
    const sweep = lams.map((l) => simulate(q, r, l)), sweep1 = lams.map((l) => simulate({ ...q, Bmax: 1 }, r, l));
    const nb = 40, mx = Math.max(S.p99 * 1.5, 1), bw = mx / nb, h = new Array(nb).fill(0); S.lat.forEach((v) => { const k = Math.floor((v * 1000) / bw); if (k < nb) h[k]++; });
    const P = [`λ = ${q.lam} istek/s, Bmax = ${q.Bmax}, w = ${q.wait} ms: p50 gecikme ${fmt(S.p50, 3)} ms, p99 ${fmt(S.p99, 3)} ms; verim ${fmt(S.thr, 4)} istek/s; ortalama batch ${fmt(S.mb, 3)}; kullanım ${fmt(S.util, 3)}.`];
    P.push(`Kapasite: batch 1 ile en fazla ${fmt(cap1, 4)} istek/s, Bmax = ${q.Bmax} ile ${fmt(cap, 4)} istek/s (t(1) = ${fmt(q.t0 + q.c, 3)} ms, t(${q.Bmax}) = ${fmt(q.t0 + q.c * q.Bmax, 3)} ms). ${q.lam >= cap ? "<b>λ kapasiteyi aşıyor</b>: kuyruk sınırsız büyür ve gecikme benzetim süresiyle artar." : q.lam > cap1 ? "Batch 1 bu yükü taşıyamazdı; batch'leme kapasiteyi artırıyor." : "Yük düşük: batch'ler küçük kalıyor ve gecikmenin çoğu bekleme sınırından geliyor."}`);
    P.push("Batch'leme sabit maliyeti (t₀: çekirdek başlatma, ağırlık okuma) örneklere böler ve verimi artırır; karşılığında istekler batch dolana kadar bekler. w, bu bekleme ile verim arasındaki düğmedir.");
    return {
      plots: [
        { title: "Gecikme dağılımı (ms)", xlabel: "gecikme (ms)", ylabel: "istek", yzero: true, series: [{ name: "istekler", x: range(nb).map((k) => (k + 0.5) * bw), y: h, kind: "bar", c: 0 }], marks: [{ x: S.p50, label: "p50" }, { x: S.p99, label: "p99" }] },
        { title: "p99 gecikme ve yük", xlabel: "λ (istek/s, log₁₀)", ylabel: "p99 (ms)", ylog: true, series: [{ name: `Bmax = ${q.Bmax}`, x: lams.map(Math.log10), y: sweep.map((s) => s.p99), c: 0 }, { name: "batch'siz (Bmax = 1)", x: lams.map(Math.log10), y: sweep1.map((s) => s.p99), c: 1 }], marks: [{ x: Math.log10(q.lam), label: "λ" }] },
      ],
      stats: [
        { label: "p50 / p99 (ms)", value: `${fmt(S.p50, 3)} / ${fmt(S.p99, 3)}` },
        { label: "Verim (istek/s)", value: S.thr },
        { label: "Ortalama batch / kullanım", value: `${fmt(S.mb, 3)} / ${fmt(S.util, 3)}` },
        { label: "Kapasite: B = 1 / Bmax", value: `${fmt(cap1, 4)} / ${fmt(cap, 4)}` },
      ],
      note: P.join(" "),
    };
  },
};
