// M5.2 — 1-B GAN: veri iki Gauss'un karışımı (±m, std 0.5, sol ağırlık w); üreteç G(z) = μ + s·z (tek tepeli, z ~ N(0,1));
// ayırt edici D(x) = σ(b + Σ_j a_j φ_j(x)), φ_j sabit merkezli RBF'ler. Her yinelemede k ayırt edici adımı (BCE), sonra bir üreteç
// adımı: minimaks log(1 − D(G(z))) ya da doyumsuz −log D(G(z)). Gradyanlar elle; SGD, batch 128.
import { fmt, range } from "./plot.js";

const C = range(25).map((i) => -6 + i * 0.5), H = 0.5, sig = (t) => 1 / (1 + Math.exp(-t));
const phi = (x) => C.map((c) => Math.exp(-((x - c) ** 2) / (2 * H * H)));
const logit = (w, x) => { const f = phi(x); return w[0] + f.reduce((s, v, j) => s + v * w[j + 1], 0); };
const dlogit = (w, x) => phi(x).reduce((s, v, j) => s - v * ((x - C[j]) / (H * H)) * w[j + 1], 0);

export function train(q, r) {
  const w = new Array(C.length + 1).fill(0); let mu = q.mu0, ls = 0; const B = 128, hist = [], gsat = [];
  const real = () => (r() < q.w ? -q.m : q.m) + 0.5 * r.normal();
  for (let it = 0; it < q.T; it++) {
    for (let k = 0; k < q.k; k++) {
      const g = new Array(w.length).fill(0);
      for (let b = 0; b < B; b++) {
        const xr = real(), xf = mu + Math.exp(ls) * r.normal(), fr = phi(xr), ff = phi(xf), dr = 1 - sig(logit(w, xr)), df = -sig(logit(w, xf));
        g[0] += dr + df; for (let j = 0; j < C.length; j++) g[j + 1] += dr * fr[j] + df * ff[j];
      }
      for (let j = 0; j < w.length; j++) w[j] += (q.lrd * g[j]) / B - q.lrd * 1e-3 * w[j];
    }
    let gm = 0, gl = 0, gabs = 0;
    for (let b = 0; b < B; b++) {
      const z = r.normal(), x = mu + Math.exp(ls) * z, a = logit(w, x), dLda = q.loss ? -(1 - sig(a)) : -sig(a), da = dlogit(w, x);
      gm += dLda * da; gl += dLda * da * Math.exp(ls) * z; gabs += Math.abs(dLda);
    }
    mu -= (q.lrg * gm) / B; ls = Math.max(-3, Math.min(2, ls - (q.lrg * gl) / B));
    hist.push([mu, Math.exp(ls)]); gsat.push(gabs / B);
  }
  return { w, mu, s: Math.exp(ls), hist, gsat };
}

export default {
  params: [
    { id: "loss", label: "Üreteç kaybı", options: ["minimaks: log(1 − D(G(z)))", "doyumsuz: −log D(G(z))"], value: 1 },
    { id: "k", label: "Üreteç adımı başına ayırt edici adımı k", min: 1, max: 10, value: 1, int: true },
    { id: "m", label: "Tepelerin yarı uzaklığı m", min: 0.5, max: 4, step: 0.1, value: 2.5 },
    { id: "w", label: "Sol tepenin ağırlığı w", min: 0.1, max: 0.9, step: 0.05, value: 0.5 },
    { id: "lrd", label: "Ayırt edici öğrenme oranı", min: 0.01, max: 2, value: 0.3, log: true },
    { id: "lrg", label: "Üreteç öğrenme oranı", min: 0.005, max: 1, value: 0.1, log: true },
    { id: "T", label: "Yineleme", min: 50, max: 3000, value: 800, int: true, log: true },
    { id: "mu0", label: "Üretecin başlangıç ortalaması μ₀", min: -5, max: 5, step: 0.1, value: 0 },
  ],
  seed: true,
  run(q, r) {
    const R = train(q, r), xs = range(241).map((i) => -6 + i * 0.05), pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const pd = (x) => q.w * pdf(x, -q.m, 0.5) + (1 - q.w) * pdf(x, q.m, 0.5), pg = (x) => pdf(x, R.mu, R.s), D = xs.map((x) => sig(logit(R.w, x)));
    const mus = R.hist.map((h) => h[0]), swings = mus.slice(1).filter((m, i) => Math.sign(m) !== Math.sign(mus[i]) && Math.abs(m) > 0.5).length;
    const near = Math.abs(Math.abs(R.mu) - q.m) < 0.8 && R.s < 1.2, avg = R.gsat.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, R.gsat.length), first = R.gsat.slice(0, 20).reduce((a, b) => a + b, 0) / Math.min(20, R.gsat.length);
    const P = [`${q.T} yinelemeden sonra üreteç N(${fmt(R.mu, 3)}, ${fmt(R.s, 3)}²); veri ±${q.m}'deki iki tepe.`];
    const mp = (1 - 2 * q.w) * q.m, sp = Math.sqrt(0.25 + q.m * q.m - mp * mp), mom = Math.abs(R.mu - mp) < 0.25 * sp && Math.abs(R.s - sp) < 0.25 * sp;
    P.push(near ? `Üreteç <b>tek bir tepeye</b> yerleşti (${R.mu < 0 ? "sol" : "sağ"}). Tek tepeli bir üreteç iki tepeyi birden temsil edemez. Ayırt edicinin gradyanı yalnız üretecin örneklerinin düştüğü yerde hesaplanır: diğer tepedeki "gerçek" işareti üretece ulaşmaz. Mod çöküşünün en basit hâli budur.` : mom ? `Üreteç iki tepeyi tek geniş bir Gauss'la örttü; verinin momentlerine yakın (veri: ortalama ${fmt(mp, 3)}, std ${fmt(sp, 3)}). Arada veri olmayan bölgeye de olasılık veriyor.` : `Üreteç ne bir tepeye ne momentlere yerleşti (veri: ortalama ${fmt(mp, 3)}, std ${fmt(sp, 3)}); daha uzun eğitimde değişebilir.`);
    if (swings > 2) P.push(`Ortalama eğitim boyunca ${swings} kez işaret değiştirdi: üreteç tepeler arasında <b>salınıyor</b>. Ayırt edici o an üretecin bulunduğu tepeyi "sahte" diye öğrenince üreteç diğerine kaçıyor; bu, mod çöküşünün dinamik yüzüdür.`);
    P.push(`Üretecin kayıp gradyanının ortalama büyüklüğü |∂L/∂logit|: ilk 20 yinelemede ${fmt(first, 3)}, son 50'de ${fmt(avg, 3)}. ${q.loss === 0 ? "Minimaks kayıpta bu terim D(G(z))'dir: ayırt edici sahteleri kolayca ayırdığında (D ≈ 0) gradyan söner (doyum)." : "Doyumsuz kayıpta bu terim 1 − D(G(z))'dir: ayırt edici sahteleri kolayca ayırdığında bile güçlü kalır."}`);
    return {
      plots: [
        { title: "Veri, üreteç ve ayırt edici", xlabel: "x", ylabel: "yoğunluk / D(x)", ylim: [0, Math.max(1.05, ...xs.map(pg), ...xs.map(pd)) * 1.05], series: [
          { name: "p_veri", x: xs, y: xs.map(pd), c: 4 },
          { name: "p_üreteç", x: xs, y: xs.map(pg), c: 1 },
          { name: "D(x)", x: xs, y: D, c: 0, dash: true }] },
        { title: "Üretecin ortalaması ve std'si", xlabel: "yineleme", ylabel: "değer", series: [
          { name: "μ", x: range(R.hist.length), y: mus, c: 2 },
          { name: "s", x: range(R.hist.length), y: R.hist.map((h) => h[1]), c: 3, dash: true }], marks: [{ y: q.m, label: "+m" }, { y: -q.m, label: "−m" }] },
      ],
      stats: [
        { label: "Üreteç μ", value: R.mu },
        { label: "Üreteç s", value: R.s },
        { label: "μ'nin işaret değiştirme sayısı", value: swings },
        { label: "Üreteç gradyanı |∂L/∂logit|: ilk 20 / son 50", value: `${fmt(first, 3)} / ${fmt(avg, 3)}` },
      ],
      note: P.join(" "),
    };
  },
};
