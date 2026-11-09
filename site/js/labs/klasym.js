// M5.1 — KL'nin asimetrisi: iki tepeli p = w·N(−m, s²) + (1 − w)·N(m, s²)'ye tek bir Gauss q = N(μ, σ²) uydurmak.
// İleri KL(p‖q) en küçük → moment eşleme (kapalı form: μ = E_p[x], σ² = Var_p[x]); ters KL(q‖p) en küçük → ızgara araması
// (sayısal integral), tepe arayan çözüm. VAE'nin ELBO'su ters KL'yi (q‖p) kullanır.
import { fmt, range } from "./plot.js";

const N = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
export function mix(x, q) { return q.w * N(x, -q.m, q.s) + (1 - q.w) * N(x, q.m, q.s); }
export function klGrid(fa, fb, xs, dx) { let k = 0; for (const x of xs) { const a = fa(x), b = fb(x); if (a > 1e-300) k += a * Math.log(a / Math.max(b, 1e-300)) * dx; } return k; }

export default {
  params: [
    { id: "m", label: "Tepelerin yarı uzaklığı m", min: 0, max: 5, step: 0.1, value: 2.5 },
    { id: "s", label: "Tepe genişliği s", min: 0.2, max: 2, step: 0.05, value: 0.6 },
    { id: "w", label: "Sol tepenin ağırlığı w", min: 0.05, max: 0.95, step: 0.05, value: 0.5 },
    { id: "start", label: "Ters KL aramasının başladığı yer", options: ["global ızgara (en iyi)", "sağ tepeden yerel iniş", "sol tepeden yerel iniş"], value: 0 },
  ],
  run(q) {
    const xs = range(601).map((i) => -10 + i / 30), dx = 1 / 30, p = (x) => mix(x, q);
    const mu_f = (1 - 2 * q.w) * q.m, var_f = q.s ** 2 + q.m ** 2 - mu_f ** 2, sf = Math.sqrt(var_f);
    const rev = (mu, sg) => klGrid((x) => N(x, mu, sg), p, xs, dx);
    let best = { mu: 0, sg: 1, k: Infinity };
    if (q.start === 0) { for (let i = 0; i <= 120; i++) for (let j = 0; j < 40; j++) { const mu = -6 + i * 0.1, sg = 0.15 + j * 0.1, k = rev(mu, sg); if (k < best.k) best = { mu, sg, k }; } }
    else { let mu = q.start === 1 ? q.m : -q.m, sg = q.s; for (let it = 0; it < 120; it++) { const h = 1e-3, k0 = rev(mu, sg), gm = (rev(mu + h, sg) - k0) / h, gs = (rev(mu, sg + h) - k0) / h; mu -= 0.2 * gm; sg = Math.max(0.1, sg - 0.2 * gs); } best = { mu, sg, k: rev(mu, sg) }; }
    // en iyi σ ile ters KL'nin μ'ye göre profili
    const prof = range(121).map((i) => -6 + i * 0.1), kp = prof.map((mu) => Math.min(...range(30).map((j) => rev(mu, 0.15 + j * 0.13))));
    const kf_f = klGrid(p, (x) => N(x, mu_f, sf), xs, dx), kr_f = rev(mu_f, sf), kf_r = klGrid(p, (x) => N(x, best.mu, best.sg), xs, dx);
    const P = [`İleri KL(p‖q)'yi en küçükleyen Gauss, p'nin ortalamasını ve varyansını eşler: μ = ${fmt(mu_f, 3)}, σ = ${fmt(sf, 3)} (kapalı form). İki tepeyi birden örter ve aradaki boşluğa da olasılık verir; KL(p‖q) = ${fmt(kf_f, 3)}, ama ters yönde KL(q‖p) = ${fmt(kr_f, 3)}.`];
    P.push(`Ters KL(q‖p)'yi en küçükleyen Gauss (${["ızgara araması", "sağ tepeden gradyan inişi", "sol tepeden gradyan inişi"][q.start]}): μ = ${fmt(best.mu, 3)}, σ = ${fmt(best.sg, 3)}, KL(q‖p) = ${fmt(best.k, 3)} (ileri yönde ${fmt(kf_r, 3)}). q'nun olasılık verdiği yerde p küçükse ceza büyüktür: ters KL <b>tepe arar</b> ve tek bir tepeye yerleşir.${q.m > 2.5 * q.s ? ` Tepeler iyi ayrıkken tek bir tepeye oturan q için KL(q‖p) ≈ log(1/tepenin ağırlığı): ${best.mu < 0 ? `sol tepe, log(1/${q.w}) = ${fmt(Math.log(1 / q.w), 3)}` : `sağ tepe, log(1/${fmt(1 - q.w, 2)}) = ${fmt(Math.log(1 / (1 - q.w)), 3)}`}.` : ""}`);
    if (q.m > 1.5 * q.s) P.push(`İkinci grafikteki profil iki yerel minimum gösteriyor: ters KL dışbükey değil, sonuç başlangıca bağlı. Bir VAE kodlayıcısı ELBO'daki KL(q(z|x)‖p(z|x)) yüzünden gerçek posterior çok tepeliyse genellikle bir tepeyi seçer.`);
    else P.push("Tepeler örtüşüyor: p tek tepeli gibi davranıyor ve iki yön benzer çözümler veriyor.");
    return {
      plots: [
        { title: "p ve iki farklı en iyi Gauss", xlabel: "x", ylabel: "yoğunluk", yzero: true, series: [
          { name: "p (iki tepeli)", x: xs, y: xs.map(p), c: 4 },
          { name: "en küçük KL(p‖q): moment eşleme", x: xs, y: xs.map((x) => N(x, mu_f, sf)), c: 0, dash: true },
          { name: "en küçük KL(q‖p): tepe arama", x: xs, y: xs.map((x) => N(x, best.mu, best.sg)), c: 2 }] },
        { title: "Ters KL(q‖p), μ'ye göre (her μ'de en iyi σ)", xlabel: "μ", ylabel: "KL(q‖p)", series: [{ name: "profil", x: prof, y: kp, c: 3 }], marks: [{ x: best.mu, label: "bulunan μ" }] },
      ],
      stats: [
        { label: "İleri: μ, σ", value: `${fmt(mu_f, 3)}, ${fmt(sf, 3)}` },
        { label: "Ters: μ, σ", value: `${fmt(best.mu, 3)}, ${fmt(best.sg, 3)}` },
        { label: "KL(p‖q) ileri çözümde / ters çözümde", value: `${fmt(kf_f, 3)} / ${fmt(kf_r, 3)}` },
        { label: "KL(q‖p) ileri çözümde / ters çözümde", value: `${fmt(kr_f, 3)} / ${fmt(best.k, 3)}` },
      ],
      note: P.join(" "),
    };
  },
};
