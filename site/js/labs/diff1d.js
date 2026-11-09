// M5.3 — 1-B difüzyon, kesin "kâhin" gürültü tahminiyle: veri bir Gauss karışımı (iki tepe). q(x_t|x_0) kapalı formda olduğu için
// x_t'nin dağılımı da bir Gauss karışımıdır ve en iyi ε tahmini E[ε | x_t] = Σ_k r_k(x_t)·√(1−ᾱ)(x_t − √ᾱ μ_k)/(ᾱ s_k² + 1 − ᾱ)
// (r_k: bileşen sorumlulukları). Eğitim hatası olmadan DDPM ve DDIM örnekleyicilerini, takvimi ve adım sayısını sına.
import { fmt, range, linspace } from "./plot.js";

export function schedule(kind, T) {
  if (kind === 0) { const b = linspace(1e-4, Math.min(0.5, (0.02 * 1000) / T), T); let a = 1; return b.map((x) => (a *= 1 - x)); }
  const s = 0.008, f = (t) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
  return range(T).map((i) => Math.min(Math.max(f(i + 1) / f(0), 1e-5), 0.9999));
}
export function oracleEps(x, ab, mix) {
  let num = 0, den = 0; const parts = mix.map((c) => { const v = ab * c.s * c.s + 1 - ab, m = Math.sqrt(ab) * c.mu, lp = Math.log(c.w) - 0.5 * Math.log(v) - (x - m) ** 2 / (2 * v); return { lp, e: Math.sqrt(1 - ab) * (x - m) / v }; });
  const mx = Math.max(...parts.map((p) => p.lp)); for (const p of parts) { const w = Math.exp(p.lp - mx); num += w * p.e; den += w; }
  return num / den;
}
export function sample(q, r, mix) {
  const ab = schedule(q.sched, q.T), al = ab.map((a, i) => a / (i ? ab[i - 1] : 1)), n = q.n, X = range(n).map(() => r.normal()), traj = [X.slice(0, 12)];
  if (q.sampler === 0) {
    for (let t = q.T - 1; t >= 0; t--) {
      const b = 1 - al[t], abp = t ? ab[t - 1] : 1, sig = Math.sqrt(((1 - abp) / (1 - ab[t])) * b);
      for (let i = 0; i < n; i++) { const e = oracleEps(X[i], ab[t], mix); X[i] = (X[i] - (b / Math.sqrt(1 - ab[t])) * e) / Math.sqrt(al[t]) + (t ? sig * r.normal() : 0); }
      if (t % Math.max(1, Math.floor(q.T / 50)) === 0) traj.push(X.slice(0, 12));
    }
  } else {
    const ts = [...new Set(linspace(q.T - 1, 0, q.steps).map(Math.round))];
    ts.forEach((t, k) => {
      const abp = k + 1 < ts.length ? ab[ts[k + 1]] : 1;
      for (let i = 0; i < n; i++) { const e = oracleEps(X[i], ab[t], mix), x0 = (X[i] - Math.sqrt(1 - ab[t]) * e) / Math.sqrt(ab[t]); X[i] = Math.sqrt(abp) * x0 + Math.sqrt(1 - abp) * e; }
      traj.push(X.slice(0, 12));
    });
  }
  return { X, traj, abT: ab[q.T - 1] };
}

export default {
  params: [
    { id: "sampler", label: "Örnekleyici", options: ["DDPM (stokastik, T adım)", "DDIM (η = 0, deterministik)"], value: 0 },
    { id: "steps", label: "DDIM adım sayısı", min: 2, max: 200, value: 20, int: true, log: true, on: { sampler: [1] } },
    { id: "T", label: "Difüzyon adımı T", min: 20, max: 1000, value: 200, int: true, log: true },
    { id: "sched", label: "Takvim", options: ["doğrusal β (T'ye göre ölçekli)", "kosinüs ᾱ"], value: 0 },
    { id: "m", label: "Tepe konumu ±m", min: 0.5, max: 3, step: 0.1, value: 2 },
    { id: "w", label: "Sol tepenin ağırlığı", min: 0.1, max: 0.9, step: 0.05, value: 0.3 },
    { id: "n", label: "Örnek sayısı", min: 200, max: 3000, value: 1000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const mix = [{ w: q.w, mu: -q.m, s: 0.3 }, { w: 1 - q.w, mu: q.m, s: 0.3 }], R = sample(q, r, mix);
    const lo = -q.m - 2, hi = q.m + 2, nb = 50, bw = (hi - lo) / nb, h = new Array(nb).fill(0);
    R.X.forEach((x) => { const k = Math.floor((x - lo) / bw); if (k >= 0 && k < nb) h[k]++; });
    const cx = range(nb).map((k) => lo + (k + 0.5) * bw), dens = h.map((c) => c / (q.n * bw)), xs = linspace(lo, hi, 200);
    const pdf = (x) => mix.reduce((s, c) => s + (c.w * Math.exp(-0.5 * ((x - c.mu) / c.s) ** 2)) / (c.s * Math.sqrt(2 * Math.PI)), 0);
    const left = R.X.filter((x) => x < 0).length / q.n, near = R.X.filter((x) => Math.min(Math.abs(x - q.m), Math.abs(x + q.m)) < 0.9).length / q.n;
    const mean = R.X.reduce((a, b) => a + b, 0) / q.n, tgtMean = q.w * -q.m + (1 - q.w) * q.m;
    const evals = q.sampler ? new Set(linspace(q.T - 1, 0, q.steps).map(Math.round)).size : q.T;
    const P = [`${evals} ağ değerlendirmesiyle ${q.n} örnek. Sol tepedeki örnek oranı ${fmt(left, 3)} (hedef ${q.w}); örneklerin %${fmt(100 * near, 3)} kadarı bir tepenin 0.9 yakınında. ᾱ_T = ${fmt(R.abT, 3)}.`];
    if (R.abT > 0.02) P.push(`<b>ᾱ_T büyük</b>: son adımda veri tam gürültüye dönüşmüyor, ama örnekleme saf N(0, 1)'den başlıyor. Bu uyuşmazlık örnekleri kaydırır (ortalama ${fmt(mean, 3)}, hedef ${fmt(tgtMean, 3)}).`);
    if (q.sampler === 1) P.push(q.steps < 10 ? `DDIM ${q.steps} adımla: her adımda x̂₀ tahmini yapılıp yeni gürültü seviyesine deterministik olarak taşınıyor. Çok az adımda kâhin bile karışımı tam çözemiyor; tepeler genişliyor ya da kayıyor.` : `DDIM, kâhin ε ile ${q.steps} adımda DDPM'nin ${q.T} adımına yakın bir dağılım veriyor: rastgelelik yalnız başlangıçta, her x_T tek bir örneğe eşleniyor.`);
    else P.push("DDPM her adımda σ_t = √β̃_t ölçekli yeni gürültü ekler; yörüngeler (ikinci grafik) son adımlara kadar dalgalanır ve tepelere ancak sonda ayrışır.");
    P.push("Kâhin, bu veri için kesin E[ε | x_t]'dir: gerçek bir ağın ulaşabileceği en iyi tahmin. Buradaki hatalar yalnız örnekleyiciden ve takvimden gelir.");
    const tr = R.traj, tt = range(tr.length);
    return {
      plots: [
        { title: "Örneklerin histogramı ve veri yoğunluğu", xlabel: "x", ylabel: "yoğunluk", yzero: true, series: [
          { name: "örnekler", x: cx, y: dens, kind: "bar", c: 0 },
          { name: "p_veri", x: xs, y: xs.map(pdf), c: 2 }] },
        { title: "12 örneğin yörüngesi (soldan sağa: gürültüden veriye)", xlabel: "örnekleme ilerlemesi", ylabel: "x", series: range(12).map((i) => ({ name: i ? "" : "yörüngeler", x: tt, y: tr.map((s) => s[i]), c: i % 6 })), marks: [{ y: q.m, label: "+m" }, { y: -q.m, label: "−m" }] },
      ],
      stats: [
        { label: "Sol tepe oranı (hedef w)", value: left },
        { label: "Bir tepeye 0.9'dan yakın örnek oranı", value: near },
        { label: "Örnek ortalaması / hedef", value: `${fmt(mean, 3)} / ${fmt(tgtMean, 3)}` },
        { label: "Ağ değerlendirmesi", value: evals },
        { label: "ᾱ_T", value: R.abT },
      ],
      note: P.join(" "),
    };
  },
};
