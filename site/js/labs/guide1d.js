// M5.8 — Yönlendirme 1-B'de kesin hesapla. Veri: p(x) = ½N(−m, σ²) + ½N(m, σ²); sınıf y = 1 sağdaki bileşen. Gürültülü marjinaller de
// karışımdır, bu yüzden ε_u (koşulsuz) ve ε_c (koşullu) kapalı formda bilinir. Deterministik DDIM (T = 1000, doğrusal β) ile:
// CFG ε̂ = ε_u + s(ε_c − ε_u); sınıflandırıcı yönlendirmesi ε̂ = ε_u − s√(1 − ᾱ)∇log p(y|x_t), sınıflandırıcı ya gürültüye duyarlı
// (kesin p(y|x_t)) ya da yalnız temiz veride eğitilmiş p(y|x₀) gürültülü x_t'ye uygulanmış. Hedef: eğik dağılım p(x) p(y|x)^s.
import { fmt, range, linspace } from "./plot.js";

const T = 1000, betas = range(T).map((i) => 1e-4 + ((0.02 - 1e-4) * i) / (T - 1)), AB = [];
{ let a = 1; for (const b of betas) { a *= 1 - b; AB.push(a); } }
const N = (x, m, v) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);
const sg = (t) => 1 / (1 + Math.exp(-t));
export function epsU(x, ab, m, s2) {
  const v = ab * s2 + 1 - ab, a = Math.sqrt(ab), w1 = N(x, a * m, v), w0 = N(x, -a * m, v), mean = (a * m * w1 - a * m * w0) / (w1 + w0);
  return (Math.sqrt(1 - ab) * (x - mean)) / v;
}
export function epsC(x, ab, m, s2) { const v = ab * s2 + 1 - ab; return (Math.sqrt(1 - ab) * (x - Math.sqrt(ab) * m)) / v; }
export function sample(q, x0s) {
  const ts = range(q.steps).map((i) => Math.round(((i + 1) * (T - 1)) / q.steps)), s2 = q.sig ** 2;
  return x0s.map((xT) => {
    let x = xT;
    for (let i = ts.length - 1; i >= 0; i--) {
      const ab = AB[ts[i]], abp = i ? AB[ts[i - 1]] : 1, eu = epsU(x, ab, q.m, s2);
      let e;
      if (q.mode === 0) e = eu + q.s * (epsC(x, ab, q.m, s2) - eu);
      else if (q.mode === 1) { const g = -(epsC(x, ab, q.m, s2) - eu) / Math.sqrt(1 - ab); e = eu - q.s * Math.sqrt(1 - ab) * g; }
      else { const k = (2 * q.m) / s2, g = (1 - sg(k * x)) * k; e = eu - q.s * Math.sqrt(1 - ab) * g; }
      const x0 = (x - Math.sqrt(1 - ab) * e) / Math.sqrt(ab);
      x = Math.sqrt(abp) * x0 + Math.sqrt(1 - abp) * e;
    }
    return x;
  });
}
const msd = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return [m, Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)]; };

export default {
  params: [
    { id: "s", label: "Yönlendirme ölçeği s", min: 0, max: 10, step: 0.1, value: 3 },
    { id: "mode", label: "Yönlendirme", options: ["sınıflandırıcısız (CFG)", "sınıflandırıcı, gürültüye duyarlı (kesin p(y | x_t))", "sınıflandırıcı, yalnız temiz veride eğitilmiş"], value: 0 },
    { id: "m", label: "Sınıf merkezleri ±m", min: 0.3, max: 3, step: 0.1, value: 1 },
    { id: "sig", label: "Sınıf içi std σ", min: 0.2, max: 1.5, step: 0.05, value: 0.6 },
    { id: "steps", label: "DDIM adımı", min: 5, max: 200, value: 50, int: true, log: true },
    { id: "n", label: "Örnek sayısı", min: 200, max: 3000, value: 1500, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const xs = sample(q, range(q.n).map(() => r.normal())), [mS, sS] = msd(xs), s2 = q.sig ** 2, k = (2 * q.m) / s2;
    // hedef: p(x) p(y=1|x)^s, ızgarada
    const G = linspace(-q.m - 5 * q.sig - 2, q.m + 5 * q.sig + 2, 801), dx = G[1] - G[0];
    const px = G.map((x) => 0.5 * N(x, -q.m, s2) + 0.5 * N(x, q.m, s2)), tilt = G.map((x, i) => px[i] * sg(k * x) ** q.s), Z = tilt.reduce((s, v) => s + v, 0) * dx, tg = tilt.map((v) => v / Z);
    const mT = G.reduce((s, x, i) => s + x * tg[i], 0) * dx, sT = Math.sqrt(G.reduce((s, x, i) => s + (x - mT) ** 2 * tg[i], 0) * dx);
    const cond = G.map((x) => N(x, q.m, s2));
    const acc = xs.filter((x) => x > 0).length / q.n;
    // histogram
    const nb = 60, lo = G[0], hi = G.at(-1), w = (hi - lo) / nb, h = new Array(nb).fill(0);
    xs.forEach((x) => { const b = Math.floor((x - lo) / w); if (b >= 0 && b < nb) h[b]++; });
    const hc = range(nb).map((b) => lo + (b + 0.5) * w), hd = h.map((c) => c / (q.n * w));
    const tv = 0.5 * range(nb).reduce((s, b) => { const t = tg.filter((_, i) => G[i] >= lo + b * w && G[i] < lo + (b + 1) * w).reduce((a, v) => a + v, 0) * dx; return s + Math.abs(h[b] / q.n - t); }, 0);
    const P = [`s = ${q.s}: örneklerin ortalaması ${fmt(mS, 3)}, std'si ${fmt(sS, 3)}; hedef p(x)p(y|x)^s'nin ortalaması ${fmt(mT, 3)}, std'si ${fmt(sT, 3)}. Örneklerin %${fmt(100 * acc, 3)}'ı doğru sınıf tarafında (x > 0). Örnek histogramı ile hedef arasındaki toplam değişim uzaklığı ${fmt(tv, 2)}.`];
    if (q.mode === 1) P.push("Gürültüye duyarlı kesin sınıflandırıcıyla yönlendirme, CFG ile <b>birebir aynı</b> örnekleri verir: Bayes kuralıyla ∇log p(y|x_t) = ∇log p(x_t|y) − ∇log p(x_t), yani ε_u − s√(1−ᾱ)∇log p(y|x_t) = ε_u + s(ε_c − ε_u). İki yöntem yalnız modellerin kusurlarında ayrılır.");
    if (q.mode === 2) P.push("Sınıflandırıcı yalnız temiz veride eğitildi; gürültülü x_t'ye uygulanınca p(y|x₀)'ın keskin sınırı yüksek gürültüde yanlış ve aşırı güvenli bir gradyan verir. Dhariwal ve Nichol bu yüzden sınıflandırıcıyı her gürültü düzeyinde (t'yi girdi alarak) eğitir.");
    if (q.s === 1 && q.mode < 2) P.push("s = 1'de kâhin skorlarla DDIM tam olarak koşullu dağılımı p(x|y) üretir (fark yalnız ayrıklaştırma ve örneklem gürültüsü).");
    else if (q.s > 1 && q.mode < 2) P.push("s > 1'de yönlendirilmiş skor her gürültü düzeyinde p_t(x)p_t(y|x)^s'nin skorudur, ama bu eğik gürültülü dağılımlar, eğik temiz dağılımın gürültülenmiş hâlleri <b>değildir</b>. Bu yüzden örnekler p(x)p(y|x)^s'yi tam vermez; ortalama ve std farkı bunu gösterir.");
    else if (q.s === 0) P.push("s = 0 koşulsuz model: iki tepe de üretilir.");
    return {
      plots: [
        { title: "Örnekler ve hedef dağılım", xlabel: "x", ylabel: "yoğunluk", wide: true, size: { w: 620, h: 240 }, series: [
          { name: "örnek histogramı", x: hc, y: hd, kind: "bar", c: 0 },
          { name: "hedef p(x) p(y|x)^s", x: G, y: tg, c: 1 },
          { name: "koşullu p(x|y)", x: G, y: cond, c: 4, dash: true },
          { name: "p(x)", x: G, y: px, c: 2, dash: true }] },
      ],
      stats: [
        { label: "Örnek ortalama / std", value: `${fmt(mS, 3)} / ${fmt(sS, 3)}` },
        { label: "Hedef ortalama / std", value: `${fmt(mT, 3)} / ${fmt(sT, 3)}` },
        { label: "Doğru sınıf tarafında", value: acc },
        { label: "Toplam değişim uzaklığı (histogram)", value: tv },
      ],
      note: P.join(" "),
    };
  },
};
