// M0.5 — Zaman ölçümünün istatistiği (benzetim): gerçek süre T, ısınma, gürültü ve ara sıra kesintiler.
// Hangi özet (ortalama, medyan, en küçük, kırpılmış ortalama) gerçek T'ye en yakın ve en kararlı?
import { fmt, range } from "./plot.js";

export function simulate(q, r) {
  const n = Math.round(q.runs), t = [];
  for (let i = 0; i < n; i++) {
    let x = q.T * Math.exp(q.noise * r.normal()); // çarpımsal (log-normal) gürültü: süreler hep pozitif ve sağa çarpık
    if (i < q.warm) x += q.T * 4 * Math.exp(-i / 2); // ısınma: ilk çağrılar (tembel ilklendirme, önbellek, JIT) yavaş
    if (r() < q.spike) x += q.T * (2 + 6 * r()); // işletim sistemi kesintisi, GC, başka süreç
    if (q.sync === 1) x = 0.012 + 0.002 * r(); // GPU: senkronizasyonsuz ölçüm yalnız kernel başlatmayı (~10 µs) görür
    t.push(x);
  }
  return t;
}
const median = (a) => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const trimmed = (a, f = 0.1) => { const s = [...a].sort((x, y) => x - y), k = Math.floor(s.length * f), u = s.slice(k, s.length - k); return u.reduce((x, y) => x + y, 0) / u.length; };
export const summaries = (a) => ({ mean: a.reduce((x, y) => x + y, 0) / a.length, median: median(a), min: Math.min(...a), trim: trimmed(a) });

export default {
  params: [
    { id: "T", label: "Gerçek süre T (ms)", min: 0.1, max: 50, value: 5, log: true, hardMin: 0.001 },
    { id: "runs", label: "Tekrar sayısı", min: 3, max: 500, value: 30, int: true, log: true },
    { id: "warm", label: "Isınma etkisi olan ilk çağrı sayısı", min: 0, max: 10, value: 3, int: true, help: "İlk çağrılar: tembel ilklendirme, bellek ayırıcı, önbellek, torch.compile derlemesi." },
    { id: "noise", label: "Gürültü düzeyi (log-normal σ)", min: 0, max: 0.6, step: 0.01, value: 0.08 },
    { id: "spike", label: "Kesinti olasılığı (her tekrar)", min: 0, max: 0.3, step: 0.005, value: 0.05 },
    { id: "skip", label: "Atılan ilk ölçüm (ısınma turu)", min: 0, max: 20, value: 0, int: true },
    { id: "sync", label: "GPU senkronizasyonu", options: ["var (torch.cuda.synchronize)", "yok (yalnız başlatma ölçülür)"], value: 0 },
  ],
  seed: true,
  validate(q) { return q.skip >= q.runs ? [{ level: "error", msg: `Atılan ilk ölçüm sayısı (${q.skip}) tekrar sayısından (${Math.round(q.runs)}) küçük olmalı: geriye ölçüm kalmaz.` }] : []; },
  run(q, r) {
    const all = simulate(q, r), kept = all.slice(q.skip), S = summaries(kept);
    // aynı ayarla 200 bağımsız deney: her özetin T etrafındaki yayılımı
    const reps = range(200).map(() => summaries(simulate(q, r).slice(q.skip)));
    const err = (k) => { const v = reps.map((x) => x[k]); const m = v.reduce((a, b) => a + b, 0) / v.length; return { bias: m - q.T, sd: Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length), rmse: Math.sqrt(v.reduce((a, b) => a + (b - q.T) ** 2, 0) / v.length) }; };
    const E = { mean: err("mean"), median: err("median"), min: err("min"), trim: err("trim") };
    const keys = ["mean", "median", "trim", "min"], names = { mean: "ortalama", median: "medyan", trim: "%10 kırpılmış ort.", min: "en küçük" };
    return {
      plots: [
        { title: "Her tekrarın ölçülen süresi", xlabel: "tekrar", ylabel: "ms", ylog: true, series: [
          { name: "ölçüm", x: range(all.length), y: all, kind: "scatter", c: 0, r: 2.8 },
          { name: "atılan ısınma turları", x: range(q.skip), y: all.slice(0, q.skip), kind: "scatter", c: 4, r: 3.5 },
        ], marks: [{ y: q.T, label: "gerçek T" }, { y: S.median, label: "medyan" }] },
        { title: "200 bağımsız deneyde özetlerin hatası (RMSE, küçük daha iyi)", xlabel: "özet", ylabel: "hata (ms)", yzero: true, xlim: [0.4, 4.6], xticks: [[1, "ortalama"], [2, "medyan"], [3, "kırpılmış"], [4, "en küçük"]], series: [
          { name: "RMSE (yan + yayılım)", x: [1, 2, 3, 4], y: keys.map((k) => Math.max(E[k].rmse, 1e-6)), kind: "bar", c: 0 },
          { name: "yalnız yayılım (std)", x: [1, 2, 3, 4], y: keys.map((k) => Math.max(E[k].sd, 1e-6)), kind: "scatter", c: 2, r: 5 },
        ] },
      ],
      stats: [
        ...keys.map((k) => ({ label: `${names[k]} (bu deney)`, value: `${fmt(S[k], 4)} ms — yan ${E[k].bias >= 0 ? "+" : ""}${fmt(E[k].bias, 2)}, std ${fmt(E[k].sd, 2)}` })),
        { label: "Gerçek T", value: q.sync === 1 ? `${fmt(q.T)} ms (ama ölçülen ~0.012 ms!)` : q.T, unit: q.sync === 1 ? "" : "ms" },
      ],
      note: q.sync === 1
        ? "<b>Senkronizasyon yok:</b> GPU işlemleri eşzamansızdır; Python satırı kernel'ı kuyruğa koyup hemen döner. Senkronize etmeden ölçülen süre gerçek hesabı değil, yalnız başlatmayı gösterir — '1000 kat hızlandı' yanılgısının en yaygın nedeni. torch.utils.benchmark.Timer senkronizasyonu kendisi yapar."
        : `Süre dağılımı sağa çarpıktır: gürültü yalnız <b>ekler</b> (işlem asla gerçek süresinden hızlı bitmez). Ortalama, kesintiler ve ısınma turlarıyla yukarı çekilir; <b>medyan</b> ve kırpılmış ortalama bunlara dayanıklıdır. En küçük değer de dayanıklıdır ama gürültü azsa T'nin altına iner (log-normal modelde), çok tekrarda iyimser yanlıdır. Pratik kural: ısınma turlarını at, çok tekrar al, medyanı ve yayılımı (IQR) raporla.`,
    };
  },
};
