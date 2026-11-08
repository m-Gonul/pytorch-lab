// M5.8 — Sınıflandırıcısız yönlendirme: kâhin skorlu 2B karışımda s'nin kalite–çeşitlilik etkisi
import { fmt, range } from "./plot.js";
const MU = [[-2, 0], [2, 0], [0, 2.5]], SIG = 0.5;
export default {
  params: [
    { id: "s", label: "Yönlendirme ölçeği s", min: 0, max: 8, step: 0.1, value: 1, help: "s = 0 koşulsuz, s = 1 koşullu, s > 1 aşırı yönlendirme." },
    { id: "cls", label: "Hedef sınıf", options: ["sol (−2, 0)", "sağ (2, 0)", "üst (0, 2.5)"], value: 0 },
    { id: "steps", label: "DDIM adımı", min: 5, max: 100, step: 1, value: 40, int: true },
    { id: "n", label: "Örnek sayısı", min: 50, max: 600, step: 50, value: 300, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const T = 200, betas = range(T).map((i) => 1e-4 + ((0.05 - 1e-4) * i) / (T - 1)); const ab = []; let a = 1; for (const b of betas) { a *= 1 - b; ab.push(a); }
    // kâhin ε: karışım Gauss'un kapalı form skoru; koşullu = tek bileşen
    const eps = (x, al, comps) => { const v = al * SIG * SIG + 1 - al; const m = comps.map((c) => [Math.sqrt(al) * MU[c][0], Math.sqrt(al) * MU[c][1]]); const lw = m.map((mm) => -((x[0] - mm[0]) ** 2 + (x[1] - mm[1]) ** 2) / (2 * v)); const mx = Math.max(...lw); const w = lw.map((l) => Math.exp(l - mx)); const S = w.reduce((q, r) => q + r, 0); const mean = [0, 1].map((d) => m.reduce((q, mm, i) => q + (w[i] / S) * mm[d], 0)); return [0, 1].map((d) => (Math.sqrt(1 - al) * (x[d] - mean[d])) / v); };
    const ts = range(p.steps).map((k) => Math.round((T - 1) * (1 - k / (p.steps - 1 || 1))));
    const X = [], Y = [];
    for (let i = 0; i < p.n; i++) {
      let x = [rnd.normal(), rnd.normal()];
      ts.forEach((t, k) => { const al = ab[t]; const eu = eps(x, al, [0, 1, 2]), ec = eps(x, al, [p.cls]); const e = [0, 1].map((d) => eu[d] + p.s * (ec[d] - eu[d])); const x0 = [0, 1].map((d) => (x[d] - Math.sqrt(1 - al) * e[d]) / Math.sqrt(al)); const ap = k + 1 < ts.length ? ab[ts[k + 1]] : 1; x = [0, 1].map((d) => Math.sqrt(ap) * x0[d] + Math.sqrt(1 - ap) * e[d]); });
      X.push(x[0]); Y.push(x[1]);
    }
    const c = MU[p.cls], nearest = X.map((x, i) => range(3).reduce((b, j) => ((x - MU[j][0]) ** 2 + (Y[i] - MU[j][1]) ** 2 < (x - MU[b][0]) ** 2 + (Y[i] - MU[b][1]) ** 2 ? j : b), 0));
    const acc = nearest.filter((j) => j === p.cls).length / p.n;
    const inC = X.map((x, i) => [x, Y[i]]).filter((_, i) => nearest[i] === p.cls);
    const mx = inC.reduce((q, v) => q + v[0], 0) / (inC.length || 1), my = inC.reduce((q, v) => q + v[1], 0) / (inC.length || 1);
    const sd = Math.sqrt(inC.reduce((q, v) => q + (v[0] - mx) ** 2 + (v[1] - my) ** 2, 0) / (2 * (inC.length || 1)));
    const off = Math.hypot(mx - c[0], my - c[1]);
    return {
      plots: [{ title: "Örnekler (gri halkalar: veri bileşenleri)", xlabel: "x₁", ylabel: "x₂", xlim: [-6, 6], ylim: [-3.5, 6], series: [{ name: "örnekler", x: X, y: Y, kind: "scatter", r: 2 }, { name: "bileşen merkezleri", x: MU.map((m) => m[0]), y: MU.map((m) => m[1]), kind: "scatter", r: 6, c: 5 }] }],
      stats: [{ label: "Hedef sınıfa düşen oran", value: acc }, { label: "Hedef kümedeki std (gerçek 0.5)", value: sd }, { label: "Küme merkezinden sapma", value: off }],
      note: p.s === 0 ? "s = 0: koşulsuz model — üç bileşen de örneklenir." : p.s <= 1.05 ? "s = 1: koşullu dağılım (kâhin modelde tam doğru: std ≈ 0.5)." : `s = ${fmt(p.s)} > 1: p(x)·p(c|x)^s'ten örnekleme benzeri — örnekler hedefe yığılır, öteki bileşenlerden uzağa itilir; std 0.5'in altına iner (çeşitlilik ↓) ve merkez kayar. Gerçek modellerde bu "aşırı doymuş" görüntüler olarak görünür.`,
    };
  },
};
