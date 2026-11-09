// M3.4 — Öğrenme oranı takvimleri: sabit, power, üstel, parçalı sabit, kosinüs (+ ısınma), 1cycle, performansa göre (plateau).
// Etkisi: gürültülü, kötü koşullu bir kuadratikte SGD + momentum'un son kaybı.
import { fmt, range } from "./plot.js";

export const NAMES = ["sabit", "power: η₀/(1 + t/s)ᶜ", "üstel: η₀·0.1^(t/s)", "parçalı sabit (×0.1, 50% ve 75%'te)", "kosinüs (+ ısınma)", "1cycle", "performansa göre (plato → ×0.5)"];
export function lrAt(kind, t, T, q, plateau) {
  const e0 = q.eta, w = Math.round(q.warm * T);
  if (kind === 4 && t < w) return (e0 * (t + 1)) / w;
  switch (kind) {
    case 0: return e0;
    case 1: return e0 / (1 + t / (T / 10)) ** 1;
    case 2: return e0 * 0.1 ** (t / (T / 2));
    case 3: return e0 * (t < 0.5 * T ? 1 : t < 0.75 * T ? 0.1 : 0.01);
    case 4: return 0.5 * e0 * (1 + Math.cos((Math.PI * (t - w)) / Math.max(1, T - w)));
    case 5: { const up = 0.45 * T, down = 0.9 * T; return t < up ? e0 / 10 + ((e0 - e0 / 10) * t) / up : t < down ? e0 - ((e0 - e0 / 10) * (t - up)) / (down - up) : (e0 / 10) * (1 - (0.99 * (t - down)) / (T - down)); }
    case 6: return plateau.eta;
  }
}
export function run(kind, q, r) {
  const T = Math.round(q.T), lam = [1, 1 / q.kappa], x = [3, 3], v = [0, 0], L = [], E = [], P = { eta: q.eta, best: Infinity, wait: 0 };
  for (let t = 0; t < T; t++) {
    const eta = lrAt(kind, t, T, q, P); E.push(eta);
    const loss = 0.5 * (lam[0] * x[0] ** 2 + lam[1] * x[1] ** 2); L.push(loss);
    if (kind === 6) { if (loss < P.best * 0.99) { P.best = loss; P.wait = 0; } else if (++P.wait > T / 20) { P.eta *= 0.5; P.wait = 0; } }
    for (let i = 0; i < 2; i++) { const g = lam[i] * x[i] + q.noise * r.normal(); v[i] = 0.9 * v[i] + g; x[i] -= eta * v[i]; }
  }
  return { L, E };
}

export default {
  params: [
    { id: "kind", label: "Takvim", options: NAMES, value: 4 },
    { id: "eta", label: "Tepe öğrenme oranı η₀", min: 0.001, max: 0.35, value: 0.1, log: true, hardMin: 1e-6, help: "Momentum 0.9 ile kararlılık sınırı ≈ 2(1 + β)/λ_max = 3.8 — ama gürültüyle ve etkin adım η/(1 − β) ile daha küçük η'lar pratik." },
    { id: "warm", label: "Isınma oranı (yalnız kosinüs)", min: 0, max: 0.3, step: 0.01, value: 0.05, on: { kind: [4] } },
    { id: "T", label: "Toplam adım", min: 100, max: 3000, value: 1000, int: true, log: true },
    { id: "noise", label: "Gradyan gürültüsü σ", min: 0, max: 1, step: 0.02, value: 0.3 },
    { id: "kappa", label: "Koşul sayısı κ", min: 1, max: 1000, value: 100, log: true },
  ],
  seed: true,
  run(q, r) {
    const T = Math.round(q.T), cur = run(q.kind, q, r), all = NAMES.map((_, k) => { const rr = run(k, q, r); return rr.L.slice(-Math.max(1, Math.round(T / 10))).reduce((a, b) => a + b, 0) / Math.max(1, Math.round(T / 10)); });
    const t = range(T), th = (v) => v.filter((_, i) => i % Math.max(1, Math.floor(T / 400)) === 0);
    return {
      plots: [
        { title: `${NAMES[q.kind]}: öğrenme oranı`, xlabel: "adım", ylabel: "η", ylog: q.kind === 2 || q.kind === 3, series: [{ name: "ηₜ", x: th(t), y: th(cur.E), c: 0 }] },
        { title: "Kayıp (gürültülü kuadratik, SGD + momentum 0.9)", xlabel: "adım", ylabel: "kayıp", ylog: true, series: [{ name: "kayıp", x: th(t), y: th(cur.L).map((v) => Math.max(v, 1e-10)), c: 2 }] },
        { title: "Son %10'luk dilimde ortalama kayıp, tüm takvimler (aynı tohum ve η₀)", xlabel: "takvim", ylabel: "son kayıp", ylog: true, ylim: [Math.max(1e-10, Math.min(...all)) / 4, Math.max(...all) * 2], xlim: [-0.5, NAMES.length - 0.5], xticks: NAMES.map((n, i) => [i, ["sabit", "power", "üstel", "parçalı", "kosinüs", "1cycle", "plato"][i]]), series: [{ name: "son kayıp", x: range(NAMES.length), y: all.map((v) => Math.max(v, 1e-10)), kind: "bar", c: 0 }, { name: "seçili", x: [q.kind], y: [Math.max(all[q.kind], 1e-10)], kind: "scatter", c: 2, r: 6 }] },
      ],
      stats: [
        { label: "Son %10'da ortalama kayıp (seçili)", value: all[q.kind] },
        { label: "En iyi takvim (bu ayarlarda)", value: NAMES[all.indexOf(Math.min(...all))] },
        { label: "Sabit η'ya göre iyileşme", value: `${fmt(all[0] / Math.max(all[q.kind], 1e-12), 3)} kat` },
      ],
      note: `Gürültülü gradyanla sabit η, optimumun etrafında η ile orantılı bir <b>gürültü topu</b>nda takılır (M1.2). Takvimlerin ortak fikri: başta büyük adımlarla hızla yaklaş, sonra adımı küçültüp topu daralt. Üstel ve parçalı takvimler sonda çok küçük η'ya iner; kosinüs bunu pürüzsüz yapar; 1cycle önce ısınır (η'yı artırır), sonra düşer. ${q.noise === 0 ? "<b>Gürültü sıfır:</b> takvimin faydası azalır; sabit η da optimuma yakınsar — o zaman yavaşlatmak yalnız zarar verir. " : ""}Dürüst not: bu küçük kuadratik problemde ısınmanın asıl faydası (büyük ağlarda ilk adımlardaki kararsızlık, Adam'ın ilk v kestirimleri) görünmez; sıralama probleme ve η₀'a bağlıdır.`,
    };
  },
};
