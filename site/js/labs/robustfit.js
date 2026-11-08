// M1.1 — Kayıp seçimi = gürültü varsayımı: MSE (Gauss), MAE (Laplace), Huber. Aykırı değerler doğruyu ne kadar çeker?
// Çözüm: yinelemeli ağırlıklı en küçük kareler (IRLS) — her adım ağırlıklı bir kapalı form.
import { fmt, linspace } from "./plot.js";

export function wls(x, y, wt) {
  let S = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
  for (let i = 0; i < x.length; i++) { S += wt[i]; Sx += wt[i] * x[i]; Sy += wt[i] * y[i]; Sxx += wt[i] * x[i] * x[i]; Sxy += wt[i] * x[i] * y[i]; }
  const w = (S * Sxy - Sx * Sy) / (S * Sxx - Sx * Sx); return { w, b: (Sy - w * Sx) / S };
}
export function fit(x, y, kind, delta = 1) {
  let p = wls(x, y, x.map(() => 1));
  if (kind === 0) return p;
  for (let it = 0; it < 100; it++) {
    const wt = x.map((xi, i) => { const r = Math.abs(p.w * xi + p.b - y[i]); return kind === 1 ? 1 / Math.max(r, 1e-6) : r <= delta ? 1 : delta / r; });
    p = wls(x, y, wt);
  }
  return p;
}
const rho = (r, k, d) => (k === 0 ? r * r : k === 1 ? Math.abs(r) : Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - 0.5 * d));
const psi = (r, k, d) => (k === 0 ? 2 * r : k === 1 ? Math.sign(r) : Math.abs(r) <= d ? r : d * Math.sign(r));

export default {
  params: [
    { id: "n", label: "Örnek sayısı", min: 10, max: 200, value: 40, int: true },
    { id: "frac", label: "Aykırı değer oranı", min: 0, max: 0.45, step: 0.01, value: 0.1 },
    { id: "mag", label: "Aykırı büyüklüğü", min: 1, max: 30, step: 0.5, value: 12 },
    { id: "loss", label: "Kayıp", options: ["MSE (Gauss gürültüsü)", "MAE (Laplace gürültüsü)", "Huber"], value: 2 },
    { id: "delta", label: "Huber eşiği δ", min: 0.1, max: 5, step: 0.1, value: 1, on: { loss: [2] }, help: "|r| ≤ δ: kare gibi; |r| > δ: mutlak değer gibi (doğrusal büyür)." },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), x = [], y = [], out = [];
    for (let i = 0; i < n; i++) { const xi = 4 * r() - 2, o = r() < q.frac; x.push(xi); out.push(o); y.push(1.5 * xi - 1 + 0.4 * r.normal() + (o ? q.mag * (0.6 + 0.4 * r()) : 0)); }
    const P = fit(x, y, q.loss, q.delta), O = fit(x, y, 0), xl = [-2.3, 2.3];
    const rs = linspace(-4, 4, 161);
    const inl = x.map((_, i) => !out[i]);
    return {
      plots: [
        { title: "Veri (turuncu = aykırı) ve uydurulan doğrular", xlabel: "x", ylabel: "y", series: [
          { name: "normal örnekler", x: x.filter((_, i) => inl[i]), y: y.filter((_, i) => inl[i]), kind: "scatter", c: 0, r: 3 },
          { name: "aykırı (yalnız yukarı kaydırılmış)", x: x.filter((_, i) => !inl[i]), y: y.filter((_, i) => !inl[i]), kind: "scatter", c: 2, r: 3.5 },
          { name: "gerçek doğru y = 1.5x − 1", x: xl, y: xl.map((v) => 1.5 * v - 1), c: 4, dash: true },
          ...(q.loss ? [{ name: "MSE (karşılaştırma)", x: xl, y: xl.map((v) => O.w * v + O.b), c: 5, dash: true }] : []),
          { name: `${["MSE", "MAE", "Huber"][q.loss]} uydurması`, x: xl, y: xl.map((v) => P.w * v + P.b), c: 1 },
        ] },
        { title: "Kayıp ρ(r) ve etkisi ψ(r) = ρ′(r)", xlabel: "artık r = tahmin − y", ylabel: "değer", ylim: [-4, 8], series: [
          { name: "ρ(r) seçili kayıp", x: rs, y: rs.map((v) => rho(v, q.loss, q.delta)), c: 1 },
          { name: "ψ(r): bir örneğin gradyana katkısı", x: rs, y: rs.map((v) => psi(v, q.loss, q.delta)), c: 2, dash: true },
        ], marks: [{ y: 0, label: "" }] },
      ],
      stats: [
        { label: "Seçili kayıpla w, b", value: `${fmt(P.w)}, ${fmt(P.b)}` },
        { label: "MSE ile w, b", value: `${fmt(O.w)}, ${fmt(O.b)}` },
        { label: "Gerçek w, b", value: "1.5, −1" },
        { label: "Seçili kayıpta b hatası", value: Math.abs(P.b + 1) },
        { label: "Aykırı sayısı", value: out.filter(Boolean).length },
      ],
      note: `ψ(r), tek bir örneğin gradyana ne kadar "itme" uyguladığıdır. MSE'de ψ = 2r sınırsız büyür: artığı 12 olan tek bir örnek, artığı 0.5 olan 24 örnek kadar çeker. MAE'de ψ = ±1: her örnek eşit oy verir, sonuç koşullu <b>medyan</b>dır (M0.4: medyan aykırılara dayanıklı). Huber ikisinin arası: küçük artıklarda MSE'nin verimliliği, büyüklerde MAE'nin dayanıklılığı. Bedeli: MAE'nin r = 0'da türevi yoktur ve çözümü tek olmayabilir; Huber'de δ seçilmelidir. Aykırı oranı %50'ye yaklaşınca hiçbir yöntem "doğru" doğruyu bulamaz: çoğunluk artık aykırılardır.`,
    };
  },
};
