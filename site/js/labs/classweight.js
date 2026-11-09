// M3.9 — Dengesiz sınıflar ve sınıf ağırlıkları: ağırlıklı çapraz entropi ile lojistik regresyon (Newton/IRLS).
// Ağırlık w, öğrenilen logit'i log w kadar kaydırır (öncel kayması); recall ↑, precision ↓, olasılıklar kalibrasyonunu kaybeder.
import { fmt, linspace, range } from "./plot.js";

export function fitLR(X, y, sw, iters = 25, l2 = 1e-4) {
  let b = [0, 0, 0];
  for (let it = 0; it < iters; it++) {
    const g = [0, 0, 0], H = [[l2, 0, 0], [0, l2, 0], [0, 0, l2]];
    for (let i = 0; i < X.length; i++) {
      const v = [1, X[i][0], X[i][1]], z = v[0] * b[0] + v[1] * b[1] + v[2] * b[2], p = 1 / (1 + Math.exp(-z)), w = sw[i];
      for (let a = 0; a < 3; a++) { g[a] += w * (p - y[i]) * v[a]; for (let c = 0; c < 3; c++) H[a][c] += w * p * (1 - p) * v[a] * v[c]; }
    }
    for (let a = 0; a < 3; a++) g[a] += l2 * b[a];
    // 3×3 çözüm (Cramer)
    const det = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    const D = det(H), step = [0, 1, 2].map((k) => det(H.map((row, i) => row.map((v, j) => (j === k ? g[i] : v)))) / D);
    b = b.map((v, k) => v - step[k]);
    if (Math.hypot(...step) < 1e-10) break;
  }
  return b;
}
export function data(n, pi, sep, r) {
  const X = [], y = [];
  for (let i = 0; i < n; i++) { const c = r() < pi ? 1 : 0, m = c ? sep / Math.SQRT2 : 0; X.push([m + r.normal(), m + r.normal()]); y.push(c); }
  return { X, y };
}

export default {
  params: [
    { id: "pi", label: "Azınlık sınıfı oranı π", min: 0.005, max: 0.5, value: 0.05, log: true },
    { id: "sep", label: "Sınıf ayrıklığı (ortalamalar arası uzaklık / std)", min: 0.5, max: 4, step: 0.1, value: 2 },
    { id: "wmode", label: "Azınlık sınıfının ağırlığı", options: ["yok (w = 1)", "ters frekans w = (1 − π)/π", "elle"], value: 1 },
    { id: "w", label: "w (elle)", min: 1, max: 200, value: 5, log: true, on: { wmode: [2] } },
    { id: "corr", label: "Çıkarımda logit'ten log w çıkar (öncel düzeltmesi)", options: ["hayır", "evet"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const N = 3000, w = q.wmode === 0 ? 1 : q.wmode === 1 ? (1 - q.pi) / q.pi : q.w, Tr = data(N, q.pi, q.sep, r), Te = data(20000, q.pi, q.sep, r);
    const b0 = fitLR(Tr.X, Tr.y, Tr.y.map(() => 1)), bw = fitLR(Tr.X, Tr.y, Tr.y.map((c) => (c ? w : 1))), shift = q.corr ? Math.log(w) : 0;
    const prob = (b, x, s = 0) => 1 / (1 + Math.exp(-(b[0] - s + b[1] * x[0] + b[2] * x[1])));
    const evalM = (b, s) => { let tp = 0, fp = 0, fn = 0, tn = 0, ps = 0; Te.X.forEach((x, i) => { const p = prob(b, x, s); ps += p; const h = p >= 0.5; if (h && Te.y[i]) tp++; else if (h) fp++; else if (Te.y[i]) fn++; else tn++; }); const rec = tp / Math.max(1, tp + fn), spec = tn / Math.max(1, tn + fp); return { rec, prec: tp / Math.max(1, tp + fp), bal: (rec + spec) / 2, acc: (tp + tn) / Te.X.length, mp: ps / Te.X.length }; };
    const E0 = evalM(b0, 0), Ew = evalM(bw, shift);
    // kalibrasyon: 10 kutu
    const bins = range(10).map(() => [0, 0, 0]); Te.X.forEach((x, i) => { const p = prob(bw, x, shift), k = Math.min(9, Math.floor(p * 10)); bins[k][0] += p; bins[k][1] += Te.y[i]; bins[k][2]++; });
    const cb = bins.filter((v) => v[2] >= 20), line = (b, s) => { const xs = linspace(-3.5, 5.5, 2); return Math.abs(b[2]) < 1e-12 ? null : { x: xs, y: xs.map((x) => -(b[0] - s + b[1] * x) / b[2]) }; };
    const sub = range(Tr.X.length).filter((i) => Tr.y[i] === 1 || i % Math.max(1, Math.round(0.1 / Math.max(q.pi, 0.01))) === 0).slice(0, 900);
    const L0 = line(b0, 0), Lw = line(bw, shift);
    return {
      plots: [
        { title: "Eğitim verisi (çoğunluk seyreltilerek çizildi) ve karar sınırları (p = 0.5)", xlabel: "x₁", ylabel: "x₂", size: { w: 400, h: 380 }, xlim: [-3.5, 5.5], ylim: [-3.5, 5.5],
          series: [{ name: "çoğunluk (y = 0)", x: sub.filter((i) => !Tr.y[i]).map((i) => Tr.X[i][0]), y: sub.filter((i) => !Tr.y[i]).map((i) => Tr.X[i][1]), kind: "scatter", c: 0, r: 2 }, { name: "azınlık (y = 1)", x: sub.filter((i) => Tr.y[i]).map((i) => Tr.X[i][0]), y: sub.filter((i) => Tr.y[i]).map((i) => Tr.X[i][1]), kind: "scatter", c: 2, r: 3 }, ...(L0 ? [{ name: "ağırlıksız", ...L0, c: 5, dash: true }] : []), ...(Lw ? [{ name: `w = ${fmt(w, 3)}${q.corr ? ", düzeltilmiş" : ""}`, ...Lw, c: 3 }] : [])] },
        { title: "Kalibrasyon (seçili model, test verisi)", xlabel: "tahmin edilen olasılık (kutu ortalaması)", ylabel: "gerçek azınlık oranı", xlim: [0, 1], ylim: [0, 1], size: { w: 360, h: 340 }, series: [{ name: "mükemmel kalibrasyon", x: [0, 1], y: [0, 1], c: 5, dash: true }, { name: "model", x: cb.map((v) => v[0] / v[2]), y: cb.map((v) => v[1] / v[2]), c: 3, dots: true }] },
      ],
      stats: [
        { label: "Azınlık recall: ağırlıksız → seçili", value: `${fmt(E0.rec, 3)} → ${fmt(Ew.rec, 3)}` },
        { label: "Azınlık precision: ağırlıksız → seçili", value: `${fmt(E0.prec, 3)} → ${fmt(Ew.prec, 3)}` },
        { label: "Dengeli doğruluk: ağırlıksız → seçili", value: `${fmt(E0.bal, 3)} → ${fmt(Ew.bal, 3)}` },
        { label: "Doğruluk: ağırlıksız → seçili", value: `${fmt(E0.acc, 3)} → ${fmt(Ew.acc, 3)}` },
        { label: "Ortalama tahmin olasılığı (gerçek π)", value: `${fmt(Ew.mp, 3)} (${fmt(q.pi, 3)})` },
        { label: "Sabit terimin kayması b₀(w) − b₀(1)  vs  log w", value: `${fmt(bw[0] - b0[0], 3)}  vs  ${fmt(Math.log(w), 3)}` },
      ],
      note: `Ağırlıklı çapraz entropinin popülasyon minimumu p_w(1|x) = w·p(1|x) / (w·p(1|x) + p(0|x)), yani <b>logit + log w</b>. Model doğru belirlenmiş olduğundan (eşit kovaryanslı Gauss sınıfları → lojistik) öğrenilen sabit terim yaklaşık log w kadar kayar, eğimler değişmez: karar sınırı paralel olarak çoğunluk tarafına kayar. Sonuç: azınlık recall'u artar, precision ve ham doğruluk düşer; olasılıklar artık gerçek oranları göstermez (kalibrasyon eğrisi köşegenin altında). ${q.corr ? "Çıkarımda log w çıkarılınca kalibrasyon geri gelir ama karar da ağırlıksız modelinkine döner: ağırlıklandırmanın kalıcı faydası bu basit modelde eşik seçimiyle aynıdır. " : "Ağırlıklandırma burada eşiği değiştirmekle eşdeğerdir; asıl faydası yanlış belirlenmiş ya da kapasitesi sınırlı modellerde (azınlık bölgesine daha çok kapasite ayırmak) ve SGD'de azınlık gradyanlarının seyrekliğini azaltmaktadır. "}`,
    };
  },
};
