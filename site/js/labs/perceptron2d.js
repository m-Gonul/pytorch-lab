// M3.8 — Rosenblatt perceptronu: hata yapınca w ← w + y·φ(x). Karar bölgesi, hata sayısı ve Novikoff sınırı (R/γ)²;
// doğrusal ayrılamayan veride (XOR, halka) sonsuza kadar dolaşma ve öznitelik dönüşümüyle (x₁x₂ ya da x₁² + x₂²) ayrılabilirlik.
import { fmt, range } from "./plot.js";

export const FEATS = ["x₁, x₂ (doğrusal)", "x₁, x₂, x₁·x₂", "x₁, x₂, x₁² + x₂²"];
export const phi = (x, f) => (f === 0 ? [x[0], x[1], 1] : f === 1 ? [x[0], x[1], x[0] * x[1], 1] : [x[0], x[1], x[0] * x[0] + x[1] * x[1], 1]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

export function makeData(kind, n, gam, r) {
  const X = [], Y = []; let ref = null; // ref: veriyi üreten ayırıcı (öznitelik uzayında) — Novikoff sınırı için
  const th = 2 * Math.PI * r(), w = [Math.cos(th), Math.sin(th)], b = 0.3 * (r() - 0.5);
  let guard = 0;
  while (X.length < n && guard++ < 200000) {
    const x = [2 * r() - 1, 2 * r() - 1];
    if (kind <= 1) { const s = w[0] * x[0] + w[1] * x[1] + b; if (Math.abs(s) < gam) continue; X.push(x); Y.push(s > 0 ? 1 : -1); }
    else if (kind === 2) { if (Math.abs(x[0]) < gam || Math.abs(x[1]) < gam) continue; X.push(x); Y.push(x[0] * x[1] > 0 ? 1 : -1); }
    else { const rr = Math.hypot(...x); if (Math.abs(rr - 0.6) < gam) continue; X.push(x); Y.push(rr < 0.6 ? 1 : -1); }
  }
  if (kind === 1) for (let i = 0; i < X.length; i++) if (r() < 0.06) Y[i] = -Y[i];
  if (kind === 0) ref = { f: 0, w: [w[0], w[1], b] };
  if (kind === 2) ref = { f: 1, w: [0, 0, 1, 0] };
  if (kind === 3) ref = { f: 2, w: [0, 0, -1, 0.36] };
  return { X, Y, ref };
}
export function train(X, Y, f, E, r) {
  const d = phi([0, 0], f).length, P = X.map((x) => phi(x, f)); let w = new Array(d).fill(0), k = 0, epochs = 0, conv = false;
  const hist = [], cum = [];
  for (let e = 0; e < E; e++) {
    const idx = range(X.length); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    let m = 0;
    for (const i of idx) if (Y[i] * dot(w, P[i]) <= 0) { w = w.map((v, j) => v + Y[i] * P[i][j]); m++; k++; }
    hist.push(m); cum.push(k); epochs = e + 1;
    if (m === 0) { conv = true; break; }
  }
  return { w, k, epochs, conv, hist, cum };
}

export default {
  params: [
    { id: "data", label: "Veri", options: ["doğrusal ayrılabilir (marjlı)", "ayrılabilir + %6 ters etiket", "XOR", "halka (içeride +, dışarıda −)"], value: 0 },
    { id: "feat", label: "Öznitelikler φ(x) (+ sabit 1 = bias nöronu)", options: FEATS, value: 0 },
    { id: "gam", label: "Boşluk (veri üretilirken sınırın iki yanında boş bırakılan genişlik)", min: 0.01, max: 0.3, step: 0.01, value: 0.1 },
    { id: "n", label: "Örnek sayısı", min: 10, max: 300, value: 80, int: true, log: true },
    { id: "E", label: "En fazla epoch", min: 1, max: 100, value: 50, int: true },
  ],
  seed: true,
  run(q, r) {
    const D = makeData(q.data, Math.round(q.n), q.gam, r), f = q.feat, T = train(D.X, D.Y, f, Math.round(q.E), r);
    const P = D.X.map((x) => phi(x, f)), R = Math.max(...P.map((v) => Math.hypot(...v)));
    let bound = null, gamma = null;
    const refw = D.ref && (D.ref.f === f ? D.ref.w : D.ref.f === 0 ? [D.ref.w[0], D.ref.w[1], 0, D.ref.w[2]] : null);
    if (refw && q.data !== 1) { const nw = Math.hypot(...refw); gamma = Math.min(...P.map((v, i) => (D.Y[i] * dot(refw, v)) / nw)); if (gamma > 0) bound = (R / gamma) ** 2; }
    const acc = P.filter((v, i) => D.Y[i] * dot(T.w, v) > 0).length / P.length;
    const N = 60, z = []; for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const x = [-1.1 + (2.2 * (i + 0.5)) / N, -1.1 + (2.2 * (j + 0.5)) / N]; z.push(dot(T.w, phi(x, f)) > 0 ? 1 : 0); }
    const pos = D.X.filter((_, i) => D.Y[i] > 0), neg = D.X.filter((_, i) => D.Y[i] < 0);
    return {
      plots: [
        { title: "Veri ve perceptronun karar bölgeleri (koyu: −, açık: +)", xlabel: "x₁", ylabel: "x₂", size: { w: 400, h: 380 }, xlim: [-1.1, 1.1], ylim: [-1.1, 1.1], field: { x0: -1.1, x1: 1.1, y0: -1.1, y1: 1.1, nx: N, ny: N, z },
          series: [{ name: "y = +1", x: pos.map((p) => p[0]), y: pos.map((p) => p[1]), kind: "scatter", c: 0, r: 3.5 }, { name: "y = −1", x: neg.map((p) => p[0]), y: neg.map((p) => p[1]), kind: "scatter", c: 2, r: 3.5 }] },
        { title: "Toplam hata (güncelleme) sayısı, epoch'a göre", xlabel: "epoch", ylabel: "toplam hata", yzero: true, series: [{ name: "toplam güncelleme", x: T.cum.map((_, i) => i + 1), y: T.cum, c: 0, dots: true }, { name: "o epoch'taki hata", x: T.hist.map((_, i) => i + 1), y: T.hist, kind: "bar", c: 4 }], marks: bound != null && bound < 4 * Math.max(...T.cum, 1) ? [{ y: bound, label: "Novikoff sınırı (R/γ)²" }] : [] },
      ],
      stats: [
        { label: "Yakınsadı mı?", value: T.conv ? `evet, ${T.epochs}. epoch hatasız` : `hayır (${T.epochs} epoch)` },
        { label: "Toplam güncelleme k", value: T.k },
        { label: "Novikoff sınırı (R/γ)²", value: bound != null ? `${fmt(bound, 4)} (R = ${fmt(R, 3)}, γ ≥ ${fmt(gamma, 3)})` : "— (bu öznitelikle ayrılabilirlik garantisi yok)" },
        { label: "Eğitim doğruluğu", value: `%${fmt(acc * 100, 4)}` },
      ],
      note: `${bound != null ? `Veri bu öznitelik uzayında γ marjıyla ayrılabilir: teorem, sıfırdan başlayan perceptronun en fazla (R/γ)² ≈ ${fmt(bound, 3)} hata yapacağını garanti eder; gerçekleşen ${T.k}. Sınır kötü durum içindir, çoğu zaman çok gevşektir. ` : q.data === 1 ? "Birkaç ters etiket veriyi ayrılamaz kılar: perceptron hiçbir epoch'u hatasız bitiremez, ağırlıklar dolaşmaya devam eder ve son ağırlıklar rastgele bir anın ağırlıklarıdır (cep algoritması ya da ortalama perceptron bunu düzeltir). " : `Bu veri ${FEATS[f]} ile doğrusal ayrılamaz: hiçbir doğru (düzlem) tüm noktaları ayıramaz; perceptron hiç yakınsamaz. ${q.data === 2 ? "XOR için x₁·x₂ özniteliğini dene." : q.data === 3 ? "Halka için x₁² + x₂² özniteliğini dene." : ""} `}Boşluğu daraltınca γ küçülür, sınır (∝ 1/γ²) ve genellikle gerçek hata sayısı büyür. Öğrenme oranı yok: w₀ = 0 iken η yalnız w'yi ölçekler, tahminleri değiştirmez.`,
    };
  },
};
