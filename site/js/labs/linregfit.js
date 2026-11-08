// M1.1 — Tek özellikli doğrusal regresyon: kapalı form vs gradyan inişi; kayıp yüzeyi, koşul sayısı ve öznitelik ölçekleme.
import { fmt, linspace, range } from "./plot.js";

const W0 = 1.5, B0 = -1; // gerçek parametreler
export function makeData(q, r) {
  const n = Math.round(q.n), x = [], y = [];
  for (let i = 0; i < n; i++) { const xi = q.c + 2 * r() - 1; x.push(xi); y.push(W0 * xi + B0 + q.noise * r.normal()); }
  return { x, y };
}
export function ols(x, y) {
  const n = x.length, mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const w = sxy / sxx; return { w, b: my - w * mx };
}
export const mse = (x, y, w, b) => x.reduce((a, xi, i) => a + (w * xi + b - y[i]) ** 2, 0) / x.length;
export function hessEig(x) { // H = (2/n) [[Σx², Σx], [Σx, n]]
  const n = x.length, a = (2 / n) * x.reduce((s, v) => s + v * v, 0), c = (2 / n) * x.reduce((s, v) => s + v, 0), d = 2;
  const tr = a + d, det = a * d - c * c, disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  return [tr / 2 + disc, Math.max(tr / 2 - disc, 0)];
}

export default {
  params: [
    { id: "n", label: "Örnek sayısı n", min: 3, max: 200, value: 30, int: true },
    { id: "noise", label: "Gürültü σ", min: 0, max: 3, step: 0.05, value: 0.5 },
    { id: "c", label: "x'lerin merkezi (x ∈ [c−1, c+1])", min: 0, max: 10, step: 0.5, value: 3, help: "Merkez sıfırdan uzaklaştıkça w ve b'nin etkileri birbirine karışır: kayıp vadisi uzar, koşul sayısı büyür." },
    { id: "std", label: "Öznitelik", options: ["ham x", "standartlaştırılmış x' = (x − x̄)/s"], value: 0 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.0001, max: 2, value: 0.05, log: true, hardMin: 1e-8, why: "η ≤ 0 inişi durdurur ya da tersine çevirir." },
    { id: "steps", label: "GD adım sayısı", min: 1, max: 1000, value: 100, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const { x: xr, y } = makeData(q, r), n = xr.length;
    const mx = xr.reduce((a, b) => a + b, 0) / n, sx = Math.sqrt(xr.reduce((a, b) => a + (b - mx) ** 2, 0) / n) || 1;
    const x = q.std === 1 ? xr.map((v) => (v - mx) / sx) : xr;
    const opt = ols(x, y), Lopt = mse(x, y, opt.w, opt.b);
    const [lmax, lmin] = hessEig(x), kappa = lmax / Math.max(lmin, 1e-12);
    // GD
    let w = 0, b = 0; const pw = [w], pb = [b], loss = [mse(x, y, w, b)];
    const T = Math.round(q.steps);
    for (let t = 0; t < T; t++) {
      let gw = 0, gb = 0; for (let i = 0; i < n; i++) { const e = w * x[i] + b - y[i]; gw += e * x[i]; gb += e; }
      w -= (q.lr * 2 * gw) / n; b -= (q.lr * 2 * gb) / n;
      if (!Number.isFinite(w) || Math.abs(w) > 1e12) break;
      pw.push(w); pb.push(b); loss.push(mse(x, y, w, b));
    }
    const diverged = pw.length < T + 1 || loss.at(-1) > 1e6 * (loss[0] + 1);
    // orijinal koordinatlara dönüş
    const back = (ww, bb) => (q.std === 1 ? { w: ww / sx, b: bb - (ww * mx) / sx } : { w: ww, b: bb });
    const gdO = back(w, b), olsO = back(opt.w, opt.b);
    const xl = [Math.min(...xr) - 0.3, Math.max(...xr) + 0.3];
    const snaps = [1, 5, 20, T].filter((s, i, a) => s <= pw.length - 1 && a.indexOf(s) === i);
    // kayıp yüzeyi penceresi: başlangıç (0, 0) ve optimum görünsün
    const wl = [Math.min(0, opt.w) - 1.2, Math.max(0, opt.w) + 1.2], bl = [Math.min(0, opt.b) - 2.5, Math.max(0, opt.b) + 2.5];
    const nx = 50, ny = 50, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) z.push(mse(x, y, wl[0] + ((i + 0.5) * (wl[1] - wl[0])) / nx, bl[0] + ((j + 0.5) * (bl[1] - bl[0])) / ny));
    const inWin = (a, lo, hi) => Math.max(lo - (hi - lo), Math.min(hi + (hi - lo), a));
    return {
      plots: [
        { title: "Veri ve doğrular (orijinal x ekseni)", xlabel: "x", ylabel: "y", series: [
          { name: "veri", x: xr, y, kind: "scatter", c: 0, r: 3 },
          ...snaps.slice(0, -1).map((s, k) => { const o = back(pw[s], pb[s]); return { name: k === 0 ? `GD ara adımlar (${snaps.slice(0, -1).join(", ")})` : "", x: xl, y: xl.map((v) => o.w * v + o.b), c: 4, dash: true }; }),
          { name: `GD, ${pw.length - 1}. adım`, x: xl, y: xl.map((v) => gdO.w * v + gdO.b), c: 2 },
          { name: "kapalı form (en küçük kareler)", x: xl, y: xl.map((v) => olsO.w * v + olsO.b), c: 1, dash: true },
        ] },
        { title: `Kayıp yüzeyi ${q.std === 1 ? "(w′, b′) — standartlaştırılmış" : "(w, b)"} ve GD yolu`, xlabel: q.std === 1 ? "w′" : "w", ylabel: q.std === 1 ? "b′" : "b", size: { w: 380, h: 370 }, xlim: wl, ylim: bl,
          field: { x0: wl[0], x1: wl[1], y0: bl[0], y1: bl[1], nx, ny, z, log: true },
          series: [{ name: "GD yolu", x: pw.map((v) => inWin(v, ...wl)), y: pb.map((v) => inWin(v, ...bl)), c: 2, dots: pw.length < 60 }, { name: "optimum", x: [opt.w], y: [opt.b], kind: "scatter", c: 1, r: 5 }] },
        { title: "Kayıp − en küçük kayıp (log)", xlabel: "adım", ylabel: "L − L*", ylog: true, series: [{ name: "GD", x: range(loss.length), y: loss.map((v) => Math.max(v - Lopt, 1e-16)), c: 2 }] },
      ],
      stats: [
        { label: "Kapalı form w, b (orijinal x)", value: `${fmt(olsO.w)}, ${fmt(olsO.b)}` },
        { label: `GD w, b (${pw.length - 1} adım)`, value: diverged ? "ıraksadı" : `${fmt(gdO.w)}, ${fmt(gdO.b)}` },
        { label: "Gerçek (veriyi üreten) w, b", value: `${W0}, ${B0}` },
        { label: "Hessian özdeğerleri λ_max, λ_min", value: `${fmt(lmax, 3)}, ${fmt(lmin, 3)}` },
        { label: "Koşul sayısı κ = λ_max/λ_min", value: kappa },
        { label: "Kararlılık sınırı η < 2/λ_max", value: 2 / lmax },
      ],
      note: `${diverged ? `<b>Iraksama:</b> η = ${fmt(q.lr)} > 2/λ_max = ${fmt(2 / lmax)}. ` : ""}MSE'nin Hessian'ı sabittir: H = (2/n)·X̃ᵀX̃. GD, en dik yönde (λ_max) kararlı kalacak kadar küçük adım atmak zorundadır; en yatık yönde (λ_min) ise her adımda yalnız ~η·λ_min oranında ilerler. Koşul sayısı κ büyükse vadi uzun ve dardır, GD zikzak yapar ya da sürünür. ${q.std === 1 ? "Standartlaştırmada x̄ = 0 ve s = 1 olduğu için H = 2I: κ = 1, tüm yönler eşit hızda yakınsar." : `Merkez c = ${fmt(q.c)}: w ve b aynı işi yapmaya çalışıyor (x hep pozitif), bu yüzden κ ≈ ${fmt(kappa, 3)}. 'Standartlaştırılmış x'i seç.`}`,
    };
  },
};
