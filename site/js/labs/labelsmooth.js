// M4.14 — Etiket yumuşatma (Szegedy ve ark. 2016; Transformer'da ε = 0.1). Hedef q = (1 − ε)·onehot + ε/V. Yüksek boyutlu,
// az örnekli bir sınıflandırma probleminde (V Gauss kümesi, D boyut) doğrusal softmax regresyonu tam batch Adam ile eğitilir;
// test kümesinde doğruluk, yumuşatılmamış NLL (perpleksitenin logu), beklenen kalibrasyon hatası (ECE, 10 kova) ve ortalama
// logit aralığı ölçülür. Bir ε taraması ve güvenilirlik diyagramı.
import { fmt, range } from "./plot.js";

export function data(n, D, V, sep, r, mu) {
  const X = [], y = [];
  for (let k = 0; k < n; k++) { const c = Math.floor(r() * V); y.push(c); X.push(range(D).map((j) => sep * mu[c][j] + r.normal())); }
  return { X, y };
}
export function fit(tr, D, V, eps, iters, r) {
  const W = range(V).map(() => range(D + 1).map(() => 0.01 * r.normal())), m = W.map((w) => w.map(() => 0)), v = W.map((w) => w.map(() => 0)), n = tr.X.length;
  for (let it = 1; it <= iters; it++) {
    const G = W.map((w) => w.map(() => 0));
    tr.X.forEach((x, k) => {
      const z = W.map((w) => w.reduce((s, wj, j) => s + wj * (j < D ? x[j] : 1), 0)), mx = Math.max(...z), e = z.map((u) => Math.exp(u - mx)), Z = e.reduce((a, b) => a + b, 0);
      for (let c = 0; c < V; c++) { const g = e[c] / Z - ((c === tr.y[k] ? 1 - eps : 0) + eps / V); for (let j = 0; j <= D; j++) G[c][j] += (g * (j < D ? x[j] : 1)) / n; }
    });
    const c1 = 1 - 0.9 ** it, c2 = 1 - 0.999 ** it;
    for (let c = 0; c < V; c++) for (let j = 0; j <= D; j++) { m[c][j] = 0.9 * m[c][j] + 0.1 * G[c][j]; v[c][j] = 0.999 * v[c][j] + 0.001 * G[c][j] ** 2; W[c][j] -= (0.05 * m[c][j]) / c1 / (Math.sqrt(v[c][j] / c2) + 1e-8); }
  }
  return W;
}
export function evaluateModel(W, te, D, V) {
  let acc = 0, nll = 0, gap = 0, mc = 0; const bins = range(10).map(() => ({ n: 0, conf: 0, acc: 0 }));
  te.X.forEach((x, k) => {
    const z = W.map((w) => w.reduce((s, wj, j) => s + wj * (j < D ? x[j] : 1), 0)), mx = Math.max(...z), e = z.map((u) => Math.exp(u - mx)), Z = e.reduce((a, b) => a + b, 0), p = e.map((u) => u / Z);
    const am = p.indexOf(Math.max(...p)), ok = am === te.y[k] ? 1 : 0, conf = p[am];
    acc += ok; mc += conf; nll -= Math.log(p[te.y[k]] + 1e-15); const zs = [...z].sort((a, b) => b - a); gap += zs[0] - zs[1];
    const b = bins[Math.min(9, Math.floor(conf * 10))]; b.n++; b.conf += conf; b.acc += ok;
  });
  const n = te.X.length, ece = bins.reduce((s, b) => s + (b.n ? (b.n / n) * Math.abs(b.acc / b.n - b.conf / b.n) : 0), 0);
  return { acc: acc / n, nll: nll / n, ece, gap: gap / n, bins, conf: mc / n };
}

export default {
  params: [
    { id: "eps", label: "Etiket yumuşatma ε", min: 0, max: 0.5, step: 0.01, value: 0.1 },
    { id: "V", label: "Sınıf sayısı V", min: 2, max: 10, value: 5, int: true },
    { id: "D", label: "Öznitelik boyutu D", min: 2, max: 100, value: 40, int: true },
    { id: "n", label: "Eğitim örneği", min: 30, max: 1000, value: 120, int: true, log: true },
    { id: "sep", label: "Küme ayrışması", min: 0.1, max: 3, step: 0.05, value: 0.6 },
    { id: "iters", label: "Adam adımı (tam batch)", min: 20, max: 1000, value: 300, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const mu = range(q.V).map(() => range(q.D).map(() => r.normal())), tr = data(q.n, q.D, q.V, q.sep, r, mu), te = data(1000, q.D, q.V, q.sep, r, mu);
    const eps = [...new Set([0, 0.05, 0.1, 0.2, 0.3, q.eps])].sort((a, b) => a - b), seed = Math.floor(r() * 1e9);
    const res = eps.map((e) => { let s = seed; const rr = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); rr.normal = () => Math.sqrt(-2 * Math.log(rr() + 1e-12)) * Math.cos(2 * Math.PI * rr()); return evaluateModel(fit(tr, q.D, q.V, e, q.iters, rr), te, q.D, q.V); });
    const cur = res[eps.indexOf(q.eps)], base = res[0], trainAcc = evaluateModel((() => { let s = seed; const rr = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); rr.normal = () => Math.sqrt(-2 * Math.log(rr() + 1e-12)) * Math.cos(2 * Math.PI * rr()); return fit(tr, q.D, q.V, q.eps, q.iters, rr); })(), tr, q.D, q.V).acc;
    const rel = (R) => R.bins.map((b, i) => (b.n ? [(i + 0.5) / 10, b.acc / b.n] : null)).filter(Boolean);
    const optGap = q.eps > 0 ? Math.log((1 - q.eps + q.eps / q.V) / (q.eps / q.V)) : Infinity;
    const P = [`Eğitim ${q.n} örnek, D = ${q.D} boyut, V = ${q.V} sınıf; eğitim doğruluğu ${fmt(trainAcc, 3)}, test doğruluğu ${fmt(cur.acc, 3)}.`];
    P.push(`ε = ${q.eps}: test NLL ${fmt(cur.nll, 3)} (ε = 0'da ${fmt(base.nll, 3)}), ECE ${fmt(cur.ece, 3)} (ε = 0'da ${fmt(base.ece, 3)}), en büyük iki logit arasındaki ortalama fark ${fmt(cur.gap, 3)} (ε = 0'da ${fmt(base.gap, 3)}).`);
    P.push(q.eps > 0 ? `Yumuşatılmış hedefin optimumunda logit farkı log((1 − ε + ε/V)/(ε/V)) = ${fmt(optGap, 3)} ile sınırlıdır; ε = 0'da optimum sonsuzdadır ve ayrılabilir veride logitler büyümeye devam eder.` : "ε = 0: hedef one-hot, optimum logit farkı sonsuz; model aşırı uyduğunda güveni doğruluğundan hızla büyür.");
    const better = cur.nll < base.nll, cal = cur.ece < base.ece;
    const st = (R) => (R.conf > R.acc + 0.02 ? "aşırı güvenli" : R.conf < R.acc - 0.02 ? "eksik güvenli" : "iyi kalibre");
    if (q.eps > 0) P.push(`Ortalama güven / doğruluk: ε = 0'da ${fmt(base.conf, 3)} / ${fmt(base.acc, 3)} (${st(base)}), ε = ${q.eps} için ${fmt(cur.conf, 3)} / ${fmt(cur.acc, 3)} (${st(cur)}). Bu ayarda yumuşatma NLL'yi ${better ? "<b>iyileştirdi</b>" : "<b>kötüleştirdi</b>"}, ECE'yi ${cal ? "iyileştirdi" : "kötüleştirdi"}. ${better ? "Düz CE aşırı güvenliyken ε'nin getirdiği güven sınırı test olabilirliğine yarıyor." : "Düz CE zaten aşırı güvenli değilken yumuşatma doğru sınıfın olasılığını gereksizce düşürüyor: Vaswani ve ark.'nın 'perpleksiteyi kötüleştirir' gözlemine karşılık gelen durum."}`);
    return {
      plots: [
        { title: "Güvenilirlik diyagramı (test)", xlabel: "güven (en büyük olasılık)", ylabel: "doğruluk", xlim: [0, 1], ylim: [0, 1], size: { w: 310, h: 300 }, series: [
          { name: "mükemmel kalibrasyon", x: [0, 1], y: [0, 1], c: 4, dash: true },
          { name: "ε = 0", x: rel(base).map((p) => p[0]), y: rel(base).map((p) => p[1]), c: 5, dots: true },
          ...(q.eps > 0 ? [{ name: `ε = ${q.eps}`, x: rel(cur).map((p) => p[0]), y: rel(cur).map((p) => p[1]), c: 0, dots: true }] : [])] },
        { title: "ε taraması (test)", xlabel: "ε", ylabel: "değer", yzero: true, series: [
          { name: "NLL (nat)", x: eps, y: res.map((x) => x.nll), c: 3, dots: true },
          { name: "ECE", x: eps, y: res.map((x) => x.ece), c: 2, dots: true },
          { name: "doğruluk", x: eps, y: res.map((x) => x.acc), c: 1, dots: true }], marks: [{ x: q.eps, label: `ε = ${q.eps}` }] },
      ],
      stats: [
        { label: "Test doğruluğu", value: cur.acc },
        { label: "Test NLL, ε = seçili / 0", value: `${fmt(cur.nll, 4)} / ${fmt(base.nll, 4)}` },
        { label: "ECE, ε = seçili / 0", value: `${fmt(cur.ece, 3)} / ${fmt(base.ece, 3)}` },
        { label: "Ortalama güven (seçili ε)", value: cur.conf },
        { label: "Ortalama logit farkı (1. − 2.)", value: cur.gap },
        { label: "Optimum logit farkı log((1−ε+ε/V)/(ε/V))", value: Number.isFinite(optGap) ? optGap : "∞" },
      ],
      note: P.join(" "),
    };
  },
};
