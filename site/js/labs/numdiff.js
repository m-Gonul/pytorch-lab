// M0.3 — Sekanttan teğete: ileri ve merkezi fark, kesme hatası (h büyük) ile yuvarlama hatası (h küçük) arasındaki denge.
import { fmt, linspace } from "./plot.js";

const sig = (x) => 1 / (1 + Math.exp(-x));
export const FUNS = [
  { name: "x³ − 2x", f: (x) => x ** 3 - 2 * x, d: (x) => 3 * x * x - 2 },
  { name: "sin x", f: Math.sin, d: Math.cos },
  { name: "eˣ", f: Math.exp, d: Math.exp },
  { name: "σ(x) (sigmoid)", f: sig, d: (x) => sig(x) * (1 - sig(x)) },
  { name: "ReLU = max(0, x)", f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), kink: true },
  { name: "|x|", f: Math.abs, d: (x) => Math.sign(x), kink: true },
];
// float32 benzetimi: her ara sonuç en yakın float32'ye yuvarlanır
export function diffs(F, x, h, f32) {
  const r = f32 ? Math.fround : (v) => v;
  const fx = r(F.f(r(x))), fp = r(F.f(r(x + h))), fm = r(F.f(r(x - h)));
  return { fwd: r(fp - fx) / h, ctr: r(fp - fm) / (2 * h) };
}

export default {
  params: [
    { id: "fn", label: "Fonksiyon", options: FUNS.map((F) => F.name), value: 0 },
    { id: "x0", label: "Nokta x₀", min: -3, max: 3, step: 0.05, value: 1, help: "Türevi aldığımız nokta. ReLU ve |x|'te 0 türevin tanımsız olduğu kırılma noktası." },
    { id: "h", label: "Adım h", min: 1e-12, max: 1, step: 0.05, value: 0.3, log: true, hardMin: 1e-15, hardMax: 2, why: "h ≤ 0 ile fark oranı tanımsız; çok küçük h'de f(x+h) = f(x) olur.", help: "Sürgü log₁₀ ölçeğindedir. Büyük h: sekant teğetten uzak (kesme hatası). Çok küçük h: f(x+h) − f(x) farkı yuvarlama gürültüsünde kaybolur." },
    { id: "prec", label: "Sayı biçimi", options: ["float64 (ε ≈ 2.2·10⁻¹⁶)", "float32 (ε ≈ 1.2·10⁻⁷)"], value: 0 },
  ],
  validate(p) {
    const F = FUNS[p.fn], out = [];
    if (F.kink && Math.abs(p.x0) <= p.h) out.push({ level: "warn", msg: `Kırılma noktası x = 0, [x₀ − h, x₀ + h] aralığının içinde: fark oranı iki yanın eğimlerini karıştırır.${p.x0 === 0 ? " x₀ = 0'da türev tanımlı değil (sol eğim ≠ sağ eğim); çerçeveler burada bir alt-gradyan seçer (PyTorch ReLU için 0)." : ""}` });
    return out;
  },
  run(p) {
    const F = FUNS[p.fn], f32 = p.prec === 1, x0 = p.x0, h = p.h;
    const exact = F.d(x0), y0 = F.f(x0);
    const xs = linspace(x0 - 2, x0 + 2, 161);
    const D = diffs(F, x0, h, f32);
    const tangent = xs.map((x) => y0 + exact * (x - x0));
    const secant = xs.map((x) => y0 + D.fwd * (x - x0));
    const ys = xs.map(F.f), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.25 + 0.2;
    const lh = linspace(-14, 0, 85), ef = [], ec = [];
    for (const e of lh) { const d = diffs(F, x0, 10 ** e, f32); ef.push(Math.max(Math.abs(d.fwd - exact), 1e-17)); ec.push(Math.max(Math.abs(d.ctr - exact), 1e-17)); }
    const eps = f32 ? 2 ** -23 : 2 ** -52;
    const ok = !(F.kink && x0 === 0);
    return {
      plots: [
        { title: `${F.name}: teğet ve ileri fark sekantı (h = ${fmt(h, 3)})`, xlabel: "x", ylabel: "f(x)", ylim: [lo - pad, hi + pad], series: [
          { name: "f(x)", x: xs, y: ys, c: 0 },
          { name: ok ? "teğet: eğim f′(x₀)" : "x₀ = 0'da teğet yok", x: xs, y: ok ? tangent : xs.map(() => NaN), c: 1 },
          { name: "sekant: (x₀, x₀+h) noktalarından", x: xs, y: secant, c: 2, dash: true },
          { name: "x₀ ve x₀+h", x: [x0, x0 + h], y: [y0, F.f(x0 + h)], kind: "scatter", c: 2, r: 4 },
        ], marks: [{ x: x0, label: "x₀" }] },
        { title: "Mutlak hata |yaklaşık − gerçek türev| (log ölçek)", xlabel: "log₁₀ h", ylabel: "hata", ylog: true, series: [
          { name: "ileri fark (f(x+h) − f(x))/h", x: lh, y: ef, c: 2 },
          { name: "merkezi fark (f(x+h) − f(x−h))/2h", x: lh, y: ec, c: 0 },
        ], marks: [{ x: Math.log10(h), label: "seçili h" }, { x: Math.log10(Math.sqrt(eps)), label: "√ε" }] },
      ],
      stats: [
        { label: "Gerçek türev f′(x₀)", value: ok ? exact : "tanımsız (kırılma)" },
        { label: "İleri fark", value: D.fwd },
        { label: "Merkezi fark", value: D.ctr },
        { label: "İleri fark hatası", value: ok ? Math.abs(D.fwd - exact) : "—" },
        { label: "Merkezi fark hatası", value: ok ? Math.abs(D.ctr - exact) : "—" },
        { label: "Kabaca en iyi h — ileri fark: √ε", value: Math.sqrt(eps) }, { label: "Kabaca en iyi h — merkezi fark: ∛ε", value: Math.cbrt(eps) },
      ],
      note: ok
        ? `Sağ grafikte hata eğrisi <b>V</b> biçimindedir. Sağ kol (büyük h) <b>kesme hatası</b>: Taylor açılımından ileri farkta ≈ h·|f″|/2 (eğim 1), merkezi farkta ≈ h²·|f‴|/6 (eğim 2) — merkezi fark aynı h'de çok daha doğrudur. Sol kol (küçük h) <b>yuvarlama hatası</b>: f(x+h) ile f(x) neredeyse eşit, farkları ε·|f|/h kadar gürültülü; h küçüldükçe büyür. ${f32 ? "float32'de ε büyük olduğu için V'nin dibi çok daha yüksektir: gradyan kontrolünü float64'te yap." : "float64'te en iyi ileri fark ≈ 10⁻⁸, merkezi fark ≈ 10⁻¹⁰–10⁻¹¹ hata verir."}`
        : "x₀ = 0'da sol eğim ile sağ eğim farklı: türev yok. İleri fark sağ eğimi, merkezi fark iki eğimin ortalamasını (ReLU'da 0.5, |x|'te 0) verir — h ne kadar küçülürse küçülsün. Gradyan kontrolünde kırılma noktalarına yakın örnekler bu yüzden yanlış alarm üretir.",
    };
  },
};
