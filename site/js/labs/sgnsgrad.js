// M4.12 — Tek bir (merkez, bağlam) çifti için skip-gram + negatif örnekleme (SGNS) kaybı ve gradyanı, 2-B vektörlerle.
// J = −log σ(u_o·v) − Σ_k log σ(−u_k·v); ∂J/∂v = (σ(u_o·v) − 1) u_o + Σ_k σ(u_k·v) u_k; ∂J/∂u_o = (σ(u_o·v) − 1) v; ∂J/∂u_k = σ(u_k·v) v.
// Gradyan inişi adımları ya yalnız merkez vektörünü ya da (word2vec'te olduğu gibi) üç vektörü birden günceller.
import { fmt, range } from "./plot.js";

const sig = (x) => 1 / (1 + Math.exp(-x)), dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const sp = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x))); // softplus: −log σ(−x)

export function sgns(v, uo, neg) {
  const so = sig(dot(uo, v)), pos = sp(-dot(uo, v)), negs = neg.map((u) => sp(dot(u, v)));
  const gv = [(so - 1) * uo[0], (so - 1) * uo[1]];
  neg.forEach((u) => { const s = sig(dot(u, v)); gv[0] += s * u[0]; gv[1] += s * u[1]; });
  return { J: pos + negs.reduce((a, b) => a + b, 0), pos, neg: negs.reduce((a, b) => a + b, 0), gv, guo: [(so - 1) * v[0], (so - 1) * v[1]], gneg: neg.map((u) => { const s = sig(dot(u, v)); return [s * v[0], s * v[1]]; }) };
}

export function descend(v0, uo0, neg0, lr, steps, all) {
  let v = [...v0], uo = [...uo0], neg = neg0.map((u) => [...u]);
  const tr = { v: [v], uo: [uo], neg: neg.map((u) => [u]), J: [], pos: [], negL: [] };
  for (let t = 0; t <= steps; t++) {
    const g = sgns(v, uo, neg);
    tr.J.push(g.J); tr.pos.push(g.pos); tr.negL.push(g.neg);
    if (t === steps) break;
    v = [v[0] - lr * g.gv[0], v[1] - lr * g.gv[1]];
    if (all) { uo = [uo[0] - lr * g.guo[0], uo[1] - lr * g.guo[1]]; neg = neg.map((u, i) => [u[0] - lr * g.gneg[i][0], u[1] - lr * g.gneg[i][1]]); }
    tr.v.push(v); tr.uo.push(uo); neg.forEach((u, i) => tr.neg[i].push(u));
  }
  return tr;
}

export default {
  params: [
    { id: "vx", label: "Merkez vektörü v_c: x", min: -2, max: 2, step: 0.1, value: -0.5 },
    { id: "vy", label: "Merkez vektörü v_c: y", min: -2, max: 2, step: 0.1, value: 1 },
    { id: "ox", label: "Pozitif bağlam u_o: x", min: -2, max: 2, step: 0.1, value: 1 },
    { id: "oy", label: "Pozitif bağlam u_o: y", min: -2, max: 2, step: 0.1, value: 0.5 },
    { id: "k", label: "Negatif örnek sayısı k", min: 0, max: 2, value: 2, int: true },
    { id: "n1x", label: "1. negatif u_1: x", min: -2, max: 2, step: 0.1, value: -1, on: { k: [1, 2] } },
    { id: "n1y", label: "1. negatif u_1: y", min: -2, max: 2, step: 0.1, value: 0.5, on: { k: [1, 2] } },
    { id: "n2x", label: "2. negatif u_2: x", min: -2, max: 2, step: 0.1, value: 0.5, on: { k: [2] } },
    { id: "n2y", label: "2. negatif u_2: y", min: -2, max: 2, step: 0.1, value: -1, on: { k: [2] } },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 3, value: 0.5, log: true },
    { id: "steps", label: "Gradyan inişi adım sayısı", min: 0, max: 200, value: 30, int: true },
    { id: "upd", label: "Güncellenen vektörler", options: ["yalnız merkez v_c", "üçü birden (word2vec gibi)"], value: 1 },
  ],
  validate(q) {
    const out = [];
    if (q.k >= 1 && q.n1x === q.ox && q.n1y === q.oy) out.push({ level: "warn", msg: "1. negatif pozitif bağlamla aynı vektör: aynı kelime hem 'gerçek' hem 'gürültü' etiketli; gradyanlar birbirini kısmen götürür (gerçek eğitimde de bu ara sıra olur)." });
    if (q.lr > 2) out.push({ level: "warn", msg: "η > 2: ilk adımlar çok büyük; kayıp salınabilir." });
    return out;
  },
  run(q) {
    const v0 = [q.vx, q.vy], uo0 = [q.ox, q.oy], neg0 = [[q.n1x, q.n1y], [q.n2x, q.n2y]].slice(0, q.k);
    const g0 = sgns(v0, uo0, neg0), tr = descend(v0, uo0, neg0, q.lr, q.steps, q.upd === 1), T = tr.J.length - 1;
    const vT = tr.v[T], uoT = tr.uo[T], nT = tr.neg.map((p) => p[T]);
    const pts = [...tr.v, ...tr.uo, ...tr.neg.flat(), [0, 0]], m = Math.max(1.2, ...pts.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.1;
    const gl = Math.hypot(...g0.gv), arrowS = gl > 1e-9 ? Math.min(1, (0.6 * m) / gl) : 0;
    const cosT = dot(vT, uoT) / (Math.hypot(...vT) * Math.hypot(...uoT) + 1e-12);
    const series = [
      { name: "v_c yolu", x: tr.v.map((p) => p[0]), y: tr.v.map((p) => p[1]), c: 2, dots: tr.v.length <= 40 },
      { name: "−∂J/∂v_c (başlangıç, ölçekli)", x: [v0[0], v0[0] - arrowS * g0.gv[0]], y: [v0[1], v0[1] - arrowS * g0.gv[1]], c: 2, dash: true },
      { name: "u_o (pozitif)", x: tr.uo.map((p) => p[0]), y: tr.uo.map((p) => p[1]), c: 0 },
      ...tr.neg.map((p, i) => ({ name: `u_${i + 1} (negatif)`, x: p.map((a) => a[0]), y: p.map((a) => a[1]), c: 5, dash: i === 1 })),
      { name: "son konumlar", x: [vT[0], uoT[0], ...nT.map((u) => u[0])], y: [vT[1], uoT[1], ...nT.map((u) => u[1])], kind: "scatter", c: 4, r: 4 },
    ];
    const big = tr.J.some((x, i) => i && x > tr.J[i - 1] + 1e-9);
    const parts = [];
    parts.push(`Başlangıçta u<sub>o</sub>·v<sub>c</sub> = ${fmt(dot(uo0, v0), 3)}, σ = ${fmt(sig(dot(uo0, v0)), 3)}: pozitif terim ${fmt(g0.pos, 4)}${q.k ? `, negatif terimler toplamı ${fmt(g0.neg, 4)} (${neg0.map((u, i) => `u<sub>${i + 1}</sub>·v<sub>c</sub> = ${fmt(dot(u, v0), 3)}`).join(", ")})` : ""}.`);
    parts.push(`Gradyan ∂J/∂v<sub>c</sub> = (σ(u<sub>o</sub>·v<sub>c</sub>) − 1) u<sub>o</sub> + Σ<sub>k</sub> σ(u<sub>k</sub>·v<sub>c</sub>) u<sub>k</sub> = (${fmt(g0.gv[0], 4)}, ${fmt(g0.gv[1], 4)}): iniş yönü v<sub>c</sub>'yi u<sub>o</sub>'ya doğru çeker${q.k ? ", negatiflerden iter; her negatifin itme gücü kendi σ'sıyla (modelin o kelimeye verdiği yanlış 'gerçek' olasılığıyla) orantılıdır" : ""}.`);
    if (q.steps) parts.push(`${q.steps} adım sonra J = ${fmt(tr.J[T], 4)} (başta ${fmt(tr.J[0], 4)}), ‖v<sub>c</sub>‖ = ${fmt(Math.hypot(...vT), 3)}, cos(v<sub>c</sub>, u<sub>o</sub>) = ${fmt(cosT, 3)}.${big ? " Kayıp bazı adımlarda <b>arttı</b>: η bu eğrilik için büyük." : ""}${q.k === 0 ? " Negatif yokken kaybın alt sınırı 0'a ancak u<sub>o</sub>·v<sub>c</sub> → ∞ ile yaklaşılır: vektörler sınırsız büyür ve <b>her</b> kelime her kelimeye benzer olur. Negatifler bu yozlaşmayı önler." : ""}${q.upd === 1 ? " Üç vektör birlikte güncellendiğinde u<sub>o</sub> da v<sub>c</sub>'ye doğru, negatifler v<sub>c</sub>'nin tersine kayar (∂J/∂u_k = σ(u_k·v_c) v_c)." : ""}`);
    const zero = neg0.filter((u) => u[0] === 0 && u[1] === 0).length;
    if (zero && q.upd === 0 && q.steps) parts.push(`${zero} negatif <b>sıfır vektör</b> ve yalnız v<sub>c</sub> güncelleniyor: u<sub>k</sub>·v<sub>c</sub> = 0 kalır, bu terim sabit log 2 = 0.693 katkı verir ve gradyan üretmez — kaybın tabanı ${fmt(zero * Math.LN2, 4)}. Üç vektör birlikte güncellenirse u<sub>k</sub> v<sub>c</sub>'nin tersine kayar ve taban kalkar.`);
    return {
      plots: [
        { title: "Vektörler ve gradyan inişi yolu (2-B)", xlabel: "x", ylabel: "y", xlim: [-m, m], ylim: [-m, m], size: { w: 310, h: 300 }, series },
        { title: "Kayıp terimleri", xlabel: "adım", ylabel: "kayıp", yzero: true, series: [
          { name: "J (toplam)", x: range(T + 1), y: tr.J, c: 3 },
          { name: "−log σ(u_o·v_c)", x: range(T + 1), y: tr.pos, c: 0, dash: true },
          ...(q.k ? [{ name: "Σ −log σ(−u_k·v_c)", x: range(T + 1), y: tr.negL, c: 5, dash: true }] : [])] },
      ],
      stats: [
        { label: "J (başlangıç)", value: g0.J },
        { label: "∂J/∂v_c: x", value: g0.gv[0] },
        { label: "∂J/∂v_c: y", value: g0.gv[1] },
        { label: "∂J/∂u_o: x", value: g0.guo[0] },
        { label: `J (${q.steps} adım sonra)`, value: tr.J[T] },
        { label: "cos(v_c, u_o) sonda", value: cosT },
      ],
      note: parts.join(" "),
    };
  },
};
