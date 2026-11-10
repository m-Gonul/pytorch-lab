// M6.2 — gradcheck neden float64 ister? Sonlu farkın hatası = kesme hatası (h ile azalır) + yuvarlama hatası (u/h ile artar).
// float32 benzetimi: her ara sonuç Math.fround ile 24 bitlik mantise yuvarlanır (x ± h dahil). İleri fark: hata ≈ |f''|h/2 + 2u|f|/h,
// en iyi h ≈ √u; merkezi fark: ≈ |f'''|h²/6 + u|f|/h, en iyi h ≈ u^(1/3). PyTorch gradcheck: eps = 1e-6, atol = 1e-5, rtol = 1e-3.
import { fmt, range } from "./plot.js";

const F = [
  { n: "exp(x)", f: Math.exp, d: Math.exp },
  { n: "sin(x)", f: Math.sin, d: Math.cos },
  { n: "tanh(x)", f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2 },
  { n: "x³", f: (x) => x ** 3, d: (x) => 3 * x * x },
  { n: "log(1 + eˣ) (softplus)", f: (x) => Math.log1p(Math.exp(x)), d: (x) => 1 / (1 + Math.exp(-x)) },
];
export function fd(fn, x0, h, central, single) {
  const r = single ? Math.fround : (v) => v, f = (x) => r(fn.f(r(x)));
  const xp = r(x0 + h), xm = r(x0 - h), x = r(x0);
  // pay ve payda gerçek (yuvarlanmış) adımla: PyTorch da (f(x+h) − f(x−h)) / (2h) hesaplar, h'yi düzeltmez
  return central ? r(r(f(xp) - f(xm)) / r(2 * h)) : r(r(f(xp) - f(x)) / r(h));
}
export default {
  params: [
    { id: "fn", label: "Fonksiyon", options: F.map((f) => f.n), value: 0 },
    { id: "x0", label: "Nokta x₀", min: -3, max: 3, step: 0.1, value: 1 },
    { id: "kind", label: "Fark formülü", options: ["merkezi: (f(x+h) − f(x−h)) / 2h", "ileri: (f(x+h) − f(x)) / h"], value: 0 },
    { id: "prec", label: "Hassasiyet (istatistikler için)", options: ["float32", "float64"], value: 0 },
    { id: "rtol", label: "gradcheck göreli toleransı rtol", min: 1e-6, max: 1e-1, value: 1e-3, log: true },
  ],
  run(q) {
    const fn = F[q.fn], exact = fn.d(q.x0), c = q.kind === 0, hs = range(57).map((i) => 10 ** (-0.25 * i - 0.5));
    const err = (s) => hs.map((h) => Math.max(1e-17, Math.abs(fd(fn, q.x0, h, c, s) - exact) / Math.max(1, Math.abs(exact))));
    const e32 = err(true), e64 = err(false), sel = q.prec === 0 ? e32 : e64, bi = sel.indexOf(Math.min(...sel));
    const u = q.prec === 0 ? 2 ** -24 : 2 ** -53, hTheory = c ? u ** (1 / 3) : Math.sqrt(u);
    const at6 = Math.abs(fd(fn, q.x0, 1e-6, c, q.prec === 0) - exact), pass = at6 <= 1e-5 + q.rtol * Math.abs(exact);
    const P = [`f = ${fn.n}, x₀ = ${q.x0}: gerçek türev ${fmt(exact, 6)}. ${q.prec === 0 ? "float32" : "float64"}'te en küçük göreli hata ${fmt(sel[bi], 2)}, h = ${fmt(hs[bi], 2)}'de (kuram: h* ≈ ${c ? "u^(1/3)" : "√u"} = ${fmt(hTheory, 2)}; grafikte "eps" PyTorch’un varsayılan adımı).`];
    P.push(`PyTorch'un varsayılan eps = 10⁻⁶ adımında fark ${fmt(at6, 2)}; |fark| ≤ atol + rtol·|türev| = ${fmt(1e-5 + q.rtol * Math.abs(exact), 2)} koşulu ${pass ? "<b>sağlanıyor</b>" : "<b>sağlanmıyor</b>: doğru türev bile reddedilir"}.`);
    P.push(q.prec === 0 ? "float32'de birim yuvarlama u = 2⁻²⁴ ≈ 6·10⁻⁸; h = 10⁻⁶'da yuvarlama hatası u·|f|/h ≈ 0.06 mertebesindedir. Eğrinin sol kolu (küçük h) bu yüzden yükselir." : "float64'te u = 2⁻⁵³ ≈ 1.1·10⁻¹⁶; h = 10⁻⁶ hem kesme hem yuvarlama hatasını 10⁻¹⁰ civarında tutar.");
    if (!c) P.push("İleri farkın kesme hatası h ile doğrusal azalır (merkezi farkta h²): en iyi hata yalnız ≈ √u'dur. gradcheck bu yüzden merkezi fark kullanır.");
    const lx = hs.map(Math.log10);
    return {
      plots: [{ title: "Sonlu fark hatası ve adım h", xlabel: "h (log₁₀)", ylabel: "göreli hata", ylog: true, wide: true, size: { w: 620, h: 260 }, series: [
        { name: "float32", x: lx, y: e32, c: 1 }, { name: "float64", x: lx, y: e64, c: 0 }],
        marks: [{ x: -6, label: "eps" }, { x: Math.log10(hTheory), label: "h*" }], xticks: [-14, -12, -10, -8, -6, -4, -2].map((v) => [v, `10^${v}`]) }],
      stats: [
        { label: "Gerçek türev", value: exact },
        { label: "En küçük göreli hata (seçili)", value: sel[bi] },
        { label: "En iyi h (ölçülen / kuramsal)", value: `${fmt(hs[bi], 2)} / ${fmt(hTheory, 2)}` },
        { label: "eps = 10⁻⁶'da gradcheck", value: pass ? "geçer" : "kalır" },
      ],
      note: P.join(" "),
    };
  },
};
