// M5.6 — Gürültü giderici otomatik kodlayıcının ne öğrendiği, kesin hesapla: veri 1-B Gauss karışımı, bozma x′ = x + σε.
// MSE ile eğitilmiş sonsuz kapasiteli DAE'nin optimumu r*(x′) = E[x | x′] (kapalı form). Tweedie: r*(x′) − x′ = σ²·d/dx′ log p_σ(x′),
// yani DAE'nin artığı gürültülü veri yoğunluğunun skoruyla orantılı (Alain & Bengio 2014; difüzyonla bağlantı, M5.3).
import { fmt, range, linspace } from "./plot.js";

export function mixture(q) {
  return [{ w: q.w, mu: -q.m, s: q.s }, { w: 1 - q.w, mu: q.m, s: q.s }];
}
const N = (x, m, v) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);
export function denoiser(xt, mix, sig) {
  let num = 0, den = 0, dp = 0;
  for (const c of mix) { const v = c.s * c.s + sig * sig, p = c.w * N(xt, c.mu, v), post = (c.mu * sig * sig + xt * c.s * c.s) / v; num += p * post; den += p; dp += p * (-(xt - c.mu) / v); }
  return { r: num / den, score: dp / den, dens: den };
}

export default {
  params: [
    { id: "sig", label: "Bozma gürültüsü σ", min: 0.05, max: 3, step: 0.05, value: 0.6 },
    { id: "m", label: "Tepeler ±m", min: 0.5, max: 4, step: 0.1, value: 2 },
    { id: "s", label: "Tepe genişliği s", min: 0.05, max: 1.5, step: 0.05, value: 0.3 },
    { id: "w", label: "Sol tepenin ağırlığı", min: 0.1, max: 0.9, step: 0.05, value: 0.5 },
    { id: "n", label: "Örnek sayısı (gösterim)", min: 50, max: 2000, value: 400, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const mix = mixture(q), xs = linspace(-q.m - 3, q.m + 3, 241), D = xs.map((x) => denoiser(x, mix, q.sig));
    const X = range(q.n).map(() => { const c = r() < q.w ? mix[0] : mix[1]; return c.mu + c.s * r.normal(); }), Xt = X.map((x) => x + q.sig * r.normal()), R = Xt.map((x) => denoiser(x, mix, q.sig).r);
    const mseNoisy = X.reduce((s, x, i) => s + (Xt[i] - x) ** 2, 0) / q.n, mseDen = X.reduce((s, x, i) => s + (R[i] - x) ** 2, 0) / q.n;
    const maxDiff = Math.max(...D.map((d, i) => Math.abs(d.r - xs[i] - q.sig * q.sig * d.score)));
    const mid = denoiser(0, mix, q.sig);
    const P = [`σ = ${q.sig}: gürültülü örneklerin temiz veriye ortalama kare uzaklığı ${fmt(mseNoisy, 3)} (≈ σ² = ${fmt(q.sig ** 2, 3)}), en iyi gürültü gidericiden sonra ${fmt(mseDen, 3)}.`];
    P.push(`En iyi DAE r*(x′) = E[x | x′], gürültülü noktayı veri tepelerine doğru çeker. Artığı r*(x′) − x′, σ²·d log p_σ/dx′'ye eşittir (Tweedie; ızgarada en büyük fark ${fmt(maxDiff, 2)}): DAE, gürültülü dağılımın <b>skorunu</b> öğrenir. Difüzyon modellerinin gürültü tahmini (M5.3) aynı nesnedir; ε̂ = −σ·skor.`);
    P.push(Math.abs(mid.r) < 0.3 * q.m && q.m > 2 * q.s + q.sig ? `İki tepenin ortasında (x′ = 0) gürültü giderici hangi tepeye gideceğini bilemez: çıktı ${fmt(mid.r, 3)}, yani iki tepenin ağırlıklı ortalaması. MSE ile eğitilmiş DAE'lerin belirsiz bölgelerde bulanık (ortalama) çıktı vermesinin nedeni budur.` : `σ büyüdükçe r*(x′) tüm noktaları verinin ortalamasına doğru çeker; σ küçükken r*(x′) ≈ x′ (neredeyse özdeşlik) ve DAE yalnız yerel yapıyı öğrenir.`);
    return {
      plots: [
        { title: "En iyi gürültü giderici r*(x′) = E[x | x′]", xlabel: "gürültülü girdi x′", ylabel: "çıktı", series: [
          { name: "r*(x′)", x: xs, y: D.map((d) => d.r), c: 0 },
          { name: "özdeşlik", x: xs, y: xs, c: 4, dash: true },
          { name: "örnekler (x′, r*)", x: Xt.slice(0, 300), y: R.slice(0, 300), kind: "scatter", c: 2, r: 2 }], marks: [{ y: -q.m, label: "−m" }, { y: q.m, label: "+m" }] },
        { title: "Artık ve skor", xlabel: "x′", ylabel: "değer", series: [
          { name: "r*(x′) − x′", x: xs, y: D.map((d, i) => d.r - xs[i]), c: 0 },
          { name: "σ² · d log p_σ / dx′", x: xs, y: D.map((d) => q.sig ** 2 * d.score), c: 2, dash: true },
          { name: "p_σ(x′) (ölçekli)", x: xs, y: D.map((d) => d.dens * 2), c: 4 }] },
      ],
      stats: [
        { label: "MSE: gürültülü / gürültü giderilmiş", value: `${fmt(mseNoisy, 3)} / ${fmt(mseDen, 3)}` },
        { label: "max |artık − σ²·skor| (ızgara)", value: maxDiff },
        { label: "r*(0)", value: mid.r },
      ],
      note: P.join(" "),
    };
  },
};
