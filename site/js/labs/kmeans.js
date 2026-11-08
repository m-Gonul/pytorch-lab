// M2.7 — k-means (Lloyd): adım adım atama/güncelleme, başlatmanın etkisi
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "k", label: "Küme sayısı k", min: 1, max: 8, step: 1, value: 4, int: true },
    { id: "true", label: "Gerçek küme sayısı", min: 1, max: 8, step: 1, value: 4, int: true },
    { id: "spread", label: "Küme yayılımı σ", min: 0.1, max: 2, step: 0.05, value: 0.5 },
    { id: "init", label: "Başlatma", options: ["rastgele k nokta", "k-means++", "hepsi aynı köşede (kötü)"], value: 0 },
    { id: "it", label: "Gösterilen iterasyon", min: 0, max: 20, step: 1, value: 3, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const C = range(p.true).map((i) => [3 * Math.cos((2 * Math.PI * i) / p.true) * (p.true > 1), 3 * Math.sin((2 * Math.PI * i) / p.true) * (p.true > 1)]);
    const X = [];
    for (const c of C) for (let j = 0; j < 60; j++) X.push([c[0] + p.spread * rnd.normal(), c[1] + p.spread * rnd.normal()]);
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    let M;
    if (p.init === 0) M = range(p.k).map(() => [...X[Math.floor(rnd() * X.length)]]);
    else if (p.init === 1) { M = [[...X[Math.floor(rnd() * X.length)]]]; while (M.length < p.k) { const D = X.map((x) => Math.min(...M.map((m) => d2(x, m)))); let r = rnd() * D.reduce((a, b) => a + b, 0), i = 0; while ((r -= D[i]) > 0 && i < X.length - 1) i++; M.push([...X[i]]); } }
    else M = range(p.k).map((i) => [-5 + 0.01 * i, -5]);
    const J = [];
    let lab = X.map(() => 0), Mshow = M.map((m) => [...m]), labShow = lab;
    for (let t = 0; t <= 20; t++) {
      lab = X.map((x) => { let b = 0; for (let j = 1; j < p.k; j++) if (d2(x, M[j]) < d2(x, M[b])) b = j; return b; });
      J.push(X.reduce((a, x, i) => a + d2(x, M[lab[i]]), 0));
      if (t === p.it) { Mshow = M.map((m) => [...m]); labShow = lab; }
      M = M.map((m, j) => { const pts = X.filter((_, i) => lab[i] === j); return pts.length ? [pts.reduce((a, q) => a + q[0], 0) / pts.length, pts.reduce((a, q) => a + q[1], 0) / pts.length] : m; });
    }
    const empty = range(p.k).filter((j) => !labShow.includes(j)).length;
    const ser = range(p.k).map((j) => ({ name: `küme ${j + 1}`, x: X.filter((_, i) => labShow[i] === j).map((q) => q[0]), y: X.filter((_, i) => labShow[i] === j).map((q) => q[1]), kind: "scatter", c: j }));
    ser.push({ name: "merkezler", x: Mshow.map((m) => m[0]), y: Mshow.map((m) => m[1]), kind: "scatter", r: 7, c: 5 });
    return {
      plots: [
        { title: `İterasyon ${p.it}: atama (renk) ve merkezler`, xlabel: "x₁", ylabel: "x₂", xlim: [-6, 6], ylim: [-6, 6], series: ser, side: true },
        { title: "Amaç J = Σ‖x − μ_c(x)‖²", xlabel: "iterasyon", ylabel: "J", series: [{ name: "J", x: range(J.length), y: J, dots: true }], marks: [{ x: p.it, label: "gösterilen" }] },
      ],
      stats: [{ label: "J (gösterilen iterasyon)", value: J[p.it] }, { label: "J (20 iterasyon sonra)", value: J[20] }, { label: "Boş küme sayısı", value: empty }],
      note: `Her iki adım da J'yi artıramaz (atama: her noktayı en yakın merkeze; güncelleme: ortalama, kareler toplamını en küçükleyen nokta) — bu yüzden J monoton iner ve sonlu adımda durur, ama yalnız <b>yerel</b> minimuma. ${empty ? "Boş küme oluştu: merkezi hiçbir noktaya en yakın değil; pratikte en uzak noktaya yeniden atanır." : ""} k-means++ uzak noktaları seçerek kötü yerel minimum olasılığını azaltır.`,
    };
  },
};
