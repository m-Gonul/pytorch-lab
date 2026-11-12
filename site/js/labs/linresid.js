// M1.1 — Artık tanılama: doğrusal model yanlış gerçek modele uygulanınca artıklar deseni ve R²'yi korur mu?
// Üç senaryo: doğrusal (ideal), karesel (eksik öznitelik → U deseni), hunimsi yayılım (heteroskedastik).
import { fmt, range } from "./plot.js";

export function olsfit(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const w = sxy / sxx, b = my - w * mx;
  const e = x.map((xi, i) => w * xi + b - y[i]); // artık r = tahmin − y
  const mse = e.reduce((a, v) => a + v * v, 0) / n;
  const syy = y.reduce((a, v) => a + (v - my) ** 2, 0) / n;
  return { w, b, e, mse, r2: syy > 1e-12 ? 1 - mse / syy : 0, rmse: Math.sqrt(mse) };
}
// artıkları merkezlenmiş x²'ye uyduran katsayı: U deseninin nicel imzası.
// x²'nin ortalaması bias terimine emildiği için merkezlemeden uydurmak katsayıyı düşük gösterir.
export const quadSig = (x, e) => {
  const z = x.map((v) => v * v);
  const m = z.reduce((a, b) => a + b, 0) / z.length;
  const zz = z.map((v) => v - m);
  return zz.reduce((a, v, i) => a + e[i] * v, 0) / zz.reduce((a, v) => a + v * v, 0);
};

const TRUE = [
  { name: "doğrusal", f: (x) => 1.5 * x - 1, hz: false, curve: true },
  { name: "karesel", f: (x) => 3 * x + 0.8 * x * x - 1, hz: false, curve: true },
  { name: "huni", f: (x) => 1.5 * x - 1, hz: true, curve: false },
];

export default {
  params: [
    { id: "gercek", label: "Gerçek model", options: ["doğrusal: y = 1.5x − 1 + ε", "karesel: y = 3x + 0.8x² − 1 + ε", "huni: y = 1.5x − 1 + σ|x|·ε"], value: 0, help: "Doğrusal modeli (doğruyla uydurmaya) zorlarsın; artıklar gerçeği ele verir." },
    { id: "n", label: "Örnek sayısı n", min: 10, max: 200, value: 60, int: true },
    { id: "sigma", label: "Gürültü σ", min: 0, max: 2, step: 0.05, value: 0.5 },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), T = TRUE[q.gercek], x = [], y = [];
    for (let i = 0; i < n; i++) {
      const xi = -2 + 4 * r();
      x.push(xi);
      y.push(T.f(xi) + (T.hz ? q.sigma * Math.abs(xi) : q.sigma) * r.normal());
    }
    const F = olsfit(x, y), xl = [-2.3, 2.3];
    // artık histogramı: 15 kutu, [−max, max] aralığı
    const em = Math.max(0.5, ...F.e.map(Math.abs));
    const K = 15, cnt = Array(K).fill(0);
    for (const v of F.e) cnt[Math.min(K - 1, Math.floor(((v + em) / (2 * em)) * K))]++;
    const c2 = quadSig(x, F.e);
    return {
      plots: [
        { title: "Veri ve zorlanmış doğrusal uydurma", xlabel: "x", ylabel: "y", series: [
          { name: "veri", x, y, kind: "scatter", c: 0, r: 3 },
          { name: "uydurma ŷ = wx + b", x: xl, y: xl.map((v) => F.w * v + F.b), c: 1 },
          ...(T.curve ? [{ name: "gerçek eğri", x: range(41).map((i) => -2 + i * 0.1), y: range(41).map((i) => T.f(-2 + i * 0.1)), c: 4, dash: true }] : []),
        ] },
        { title: "Artıklar r = tahmin − y, x'e karşı", xlabel: "x", ylabel: "r", series: [
          { name: "artıklar", x, y: F.e, kind: "scatter", c: 2, r: 3 },
        ], marks: [{ y: 0, label: "" }] },
        { title: "Artık dağılımı (15 kutu)", xlabel: "r", ylabel: "adet", series: [
          { name: "histogram", x: cnt.map((_, i) => -em + ((i + 0.5) / K) * 2 * em), y: cnt, kind: "bar", c: 3 },
        ] },
      ],
      stats: [
        { label: "uydurma w, b", value: `${fmt(F.w)}, ${fmt(F.b)}` },
        { label: "R²", value: F.r2 },
        { label: "RMSE", value: F.rmse },
        { label: "artıkları (x²−x̄²)'ye uyduran katsayı (U imzası)", value: c2 },
      ],
      note: T.name === "doğrusal"
        ? `Model doğru uydur: artıklar desensiz bir bulut, ortalaması sıfır, dağılımı x'ten bağımsız; histogram tek tepeli ve simetrik — Gauss varsayımı tutarlı. (x²−x̄²) imzası ${fmt(c2)} ≈ 0: artıkların yapısında eğrilik yok. R² = ${fmt(F.r2)} burada "model doğru" demektir.`
        : T.name === "karesel"
        ? `<b>U deseni:</b> gerçek ilişki eğri (mavi kesikli 3x + 0.8x² − 1); doğrusal model doğrusal kısmı emer, eğriliği artıklarda bırakır. (x²−x̄²) imzası ${fmt(c2)} ≈ −0.8 — eksik terimin katsayısı tam olarak artıklarda gizli (x²'nin ortalaması bias'a emildiği için imza (x²−x̄²)'ye uydurularak ölçülür). R² = ${fmt(F.r2)} hâlâ ~0.9: R² yanlış biçimi gizler (yukarıdaki Anscombe şekli). Çözüm: özniteliği genişlet — [x, x²] ile aynı formüller (E1 alıştırması).`
        : `<b>Huni (heteroskedastik):</b> artıkların yayılımı |x|'le büyür; ortalama sıfır ama varyans sabit değil. Histogram simetrik ama kuyrukları geniştir. Bu, regresyonun yansızlığını bozmaz ama standart hata formülleri (σ²(X̃ᵀX̃)⁻¹) geçersizleşir: ağırlıklı en küçük kareler gerekir (yukarıdaki ayrıntı kutusu). n'i büyütüp σ ile oyna: desen örnek sayısıyla netleşir.`,
    };
  },
};
