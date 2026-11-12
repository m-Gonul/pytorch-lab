// M1.2 — Batch, mini-batch ve stokastik GD: aynı epoch bütçesinde ilerleme, gürültü topu ve azalan öğrenme oranı.
// Adım bazlı kayıp, birinci mertebe teorik taban (η/4)·tr Σg ve karıştırma/sıralı seçeneği dahil.
import { fmt, range } from "./plot.js";

export function data(n, r) { const x = [], y = []; for (let i = 0; i < n; i++) { const xi = 1 + 2 * r(); x.push(xi); y.push(2 * xi - 1 + 0.5 * r.normal()); } return { x, y }; }
const mse = (x, y, w, b) => x.reduce((a, xi, i) => a + (w * xi + b - y[i]) ** 2, 0) / x.length;
function ols(x, y) { const n = x.length, mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n; let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; } const w = sxy / sxx; return { w, b: my - w * mx }; }
export function train(x, y, B, lr, epochs, sched, r, order = 0) {
  const n = x.length; let w = -2, b = 3; const pw = [w], pb = [b], le = [mse(x, y, w, b)], ls = [];
  let steps = 0;
  for (let ep = 0; ep < epochs; ep++) {
    let idx = range(n);
    if (order === 1) idx.sort((i, j) => x[i] - x[j]);   // sıralı veri: her batch tek x bölgesi
    else for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const eta = sched === 1 ? lr / (1 + ep) : lr;
    for (let s = 0; s < n; s += B) {
      const bi = idx.slice(s, s + B); let gw = 0, gb = 0;
      for (const i of bi) { const e = w * x[i] + b - y[i]; gw += e * x[i]; gb += e; }
      w -= (eta * 2 * gw) / bi.length; b -= (eta * 2 * gb) / bi.length; steps++;
      if (!Number.isFinite(w) || Math.abs(w) > 1e8) return { pw, pb, le, ls, steps, div: true };
      pw.push(w); pb.push(b); ls.push(mse(x, y, w, b));
    }
    le.push(mse(x, y, w, b));
  }
  return { pw, pb, le, ls, steps, div: false };
}

export default {
  params: [
    { id: "B", label: "Mini-batch boyu B", min: 1, max: 200, value: 10, int: true, log: true, help: "B = 1: saf SGD. B = 200 (= n): tam batch GD (epoch başına 1 adım)." },
    { id: "lr", label: "Öğrenme oranı η", min: 0.001, max: 0.2, value: 0.05, log: true, hardMin: 1e-6 },
    { id: "epochs", label: "Epoch sayısı", min: 1, max: 60, value: 10, int: true },
    { id: "sched", label: "Öğrenme oranı takvimi", options: ["sabit η", "azalan ηₑ = η/(1 + e)"], value: 0 },
    { id: "order", label: "Veri sırası", options: ["karıştırılmış (shuffle)", "x'e göre sıralı"], value: 0, help: "Sıralı: her batch tek x bölgesi görür; batch gradyanı yansızlığını yitirir (tanım kutusundaki tehlike)." },
  ],
  seed: true,
  run(q, r) {
    const n = 200, { x, y } = data(n, r), B = Math.min(Math.round(q.B), n), E = Math.round(q.epochs);
    const opt = ols(x, y), Lopt = mse(x, y, opt.w, opt.b);
    const S = train(x, y, B, q.lr, E, q.sched, r, q.order), G = train(x, y, n, q.lr, E, q.sched, r, q.order);
    // her adımda tek örnek gradyanının varyansı (optimumda): gürültü düzeyi tahmini
    let v = 0, s1 = 0; for (let i = 0; i < n; i++) { const e = opt.w * x[i] + opt.b - y[i]; v += (2 * e * x[i]) ** 2 + (2 * e) ** 2; s1 += e * e * (x[i] * x[i] + 1); } v /= n;
    // birinci mertebe denge: E[L − L*] ≈ (η/4)·tr Σg = (η/4)·(4/(nB))·Σ e²(x²+1) = η·Σ e²(x²+1)/(nB)
    const theory = q.sched === 0 && !S.div ? (q.lr * s1) / (n * B) : null;
    const wl = [-2.6, 3.4], bl = [-3.5, 7], nx = 46, ny = 46, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) z.push(mse(x, y, wl[0] + ((i + 0.5) * (wl[1] - wl[0])) / nx, bl[0] + ((j + 0.5) * (bl[1] - bl[0])) / ny));
    const thin = (a) => { const k = Math.max(1, Math.ceil(a.length / 1500)); return a.filter((_, i) => i % k === 0 || i === a.length - 1); };
    const clip = (a, lo, hi) => a.map((v) => Math.max(lo, Math.min(hi, v)));
    const tail = Math.max(1, Math.floor(S.pw.length / 4)), dist = S.pw.slice(-tail).map((w, i) => Math.hypot(w - opt.w, S.pb[S.pb.length - tail + i] - opt.b));
    const meanDist = dist.reduce((a, b) => a + b, 0) / dist.length;
    const lossPlot = (title, X, Y, c) => ({
      title, xlabel: title.startsWith("Epoch") ? "epoch" : "adım", ylabel: "L − L*", ylog: true, series: [
        { name: `B = ${B}`, x: X, y: Y.map((v) => Math.max(v - Lopt, 1e-12)), c, dots: true },
        ...(theory != null ? [{ name: "teorik taban ≈ (η/4)·tr Σg", x: [X[0], X[X.length - 1]], y: [theory, theory], c: 1, dash: true }] : []),
      ],
    });
    return {
      plots: [
        { title: `Parametre yolu (B = ${B}, ${S.steps} adım) ve tam batch GD (${G.steps} adım)`, xlabel: "w", ylabel: "b", size: { w: 380, h: 370 }, xlim: wl, ylim: bl,
          field: { x0: wl[0], x1: wl[1], y0: bl[0], y1: bl[1], nx, ny, z, log: true },
          series: [
            { name: `mini-batch B = ${B}${q.order === 1 ? ", sıralı" : ""}`, x: clip(thin(S.pw), ...wl), y: clip(thin(S.pb), ...bl), c: 2 },
            { name: "tam batch GD", x: clip(G.pw, ...wl), y: clip(G.pb, ...bl), c: 0, dots: true },
            { name: "optimum", x: [opt.w], y: [opt.b], kind: "scatter", c: 1, r: 5 },
          ] },
        lossPlot("Epoch sonunda kayıp − en küçük kayıp", range(S.le.length), S.le, 2),
        lossPlot("Adım bazında kayıp − en küçük kayıp (gürültü topu)", range(S.ls.length).filter((_, i) => i % Math.max(1, Math.ceil(S.ls.length / 1500)) === 0 || i === S.ls.length - 1),
                 S.ls.filter((_, i) => i % Math.max(1, Math.ceil(S.ls.length / 1500)) === 0 || i === S.ls.length - 1), 2),
      ],
      stats: [
        { label: "Epoch başına adım ⌈n/B⌉", value: Math.ceil(n / B) },
        { label: `Son kayıp − L* (B = ${B})`, value: S.div ? "ıraksadı" : S.le.at(-1) - Lopt },
        { label: "Son kayıp − L* (tam batch)", value: G.div ? "ıraksadı" : G.le.at(-1) - Lopt },
        { label: "Son çeyrekte optimuma ortalama uzaklık", value: meanDist },
        { label: "Mini-batch gradyan gürültüsü ≈ tr Σ / B (optimumda)", value: v / B },
        ...(theory != null ? [{ label: "Teorik denge tabanı (η/4)·tr Σg", value: theory }] : []),
      ],
      note: `Aynı epoch sayısında (aynı hesap bütçesi) mini-batch, tam batch'ten ${fmt(Math.ceil(n / B), 3)} kat fazla adım atar; gradyanları gürültülü ama ucuzdur. Sabit η ile yol optimuma gelip etrafında bir <b>gürültü topu</b> içinde dolaşır: topun büyüklüğü kabaca η·(gürültü)/(eğrilik), yani η ve 1/B ile büyür. Adım bazlı grafik bu topu gösterir; noktalı çizgi birinci mertebe denge tabanı (η/4)·tr Σg'dir (ηa ≪ 1 için; ölçümlenen tabanla karşılaştır). ${q.sched === 1 ? "Azalan takvimde adımlar küçüldükçe top da küçülür ve yol optimuma yaklaşır (Robbins–Monro koşulları: Σηₜ = ∞, Σηₜ² < ∞)." : "'Azalan' takvimi seç: gürültü topu küçülür."} ${q.order === 1 ? "Sıralı veri: her batch tek x bölgesinden geldiği için batch gradyanı hep aynı tarafta yanlı; yol b yönünde salınıp doğrudan ilerleyemez — karıştırmanın amacı budur." : ""}`,
    };
  },
};
