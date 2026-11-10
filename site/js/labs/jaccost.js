// M6.3 — Jacobian'ı iki yoldan, gerçekten hesaplayarak: f: Rⁿ → Rᵐ, iki gizli katmanlı tanh MLP (genişlik h). jacfwd = bir ileri geçiş +
// n JVP (her girdi yönü için teğet vektörünü ileri taşı: ż = W ż); jacrev = bir ileri geçiş (aktivasyonları sakla) + m VJP
// (ḡ = Wᵀ ḡ). Çarpma–toplama sayısı sayılır, iki Jacobian karşılaştırılır. Saklanan aktivasyon sayısı ters modun bellek bedelidir.
import { fmt, range } from "./plot.js";

export function mlp(r, sizes) {
  return sizes.slice(1).map((o, l) => { const i = sizes[l], s = 1 / Math.sqrt(i); return { W: range(o).map(() => range(i).map(() => s * r.normal() * 1.5)), b: range(o).map(() => 0.1 * r.normal()) }; });
}
export function forward(net, x, keep) {
  let a = x, macs = 0; const acts = [a];
  net.forEach((L, l) => { const z = L.W.map((row, k) => row.reduce((s, w, j) => s + w * a[j], L.b[k])); macs += L.W.length * a.length; a = l < net.length - 1 ? z.map(Math.tanh) : z; if (keep) acts.push(a); });
  return { y: a, acts, macs };
}
export function jacfwd(net, x) {
  const { macs: m0 } = forward(net, x, false), n = x.length; let macs = m0;
  const cols = range(n).map((j) => {
    let a = x, t = range(n).map((i) => (i === j ? 1 : 0));
    net.forEach((L, l) => { const z = L.W.map((row, k) => row.reduce((s, w, i) => s + w * a[i], L.b[k])), zd = L.W.map((row) => row.reduce((s, w, i) => s + w * t[i], 0)); macs += L.W.length * a.length;
      if (l < net.length - 1) { a = z.map(Math.tanh); t = zd.map((v, k) => v * (1 - a[k] ** 2)); } else { a = z; t = zd; } });
    return t;
  });
  // ilk ileri geçiş dışında birincil değerler yeniden hesaplanır ama vmap'li jacfwd bunu bir kez yapar: yalnız teğet çarpımlarını say
  return { J: range(cols[0].length).map((i) => cols.map((c) => c[i])), macs: m0 + n * m0 };
}
export function jacrev(net, x) {
  const f = forward(net, x, true), mOut = f.y.length; let macs = f.macs;
  const rows = range(mOut).map((i) => {
    let g = range(mOut).map((k) => (k === i ? 1 : 0));
    for (let l = net.length - 1; l >= 0; l--) {
      const L = net[l]; if (l < net.length - 1) g = g.map((v, k) => v * (1 - f.acts[l + 1][k] ** 2));
      g = range(L.W[0].length).map((j) => L.W.reduce((s, row, k) => s + row[j] * g[k], 0)); macs += L.W.length * L.W[0].length;
    }
    return g;
  });
  return { J: rows, macs, stored: f.acts.slice(1, -1).reduce((s, a) => s + a.length, 0) };
}
export default {
  params: [
    { id: "n", label: "Girdi boyutu n", min: 1, max: 128, value: 8, int: true, log: true },
    { id: "m", label: "Çıktı boyutu m", min: 1, max: 128, value: 64, int: true, log: true },
    { id: "h", label: "Gizli genişlik h", min: 4, max: 128, value: 32, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const net = mlp(r, [q.n, q.h, q.h, q.m]), x = range(q.n).map(() => r.normal());
    const F = jacfwd(net, x), R = jacrev(net, x);
    const diff = Math.max(...F.J.flatMap((row, i) => row.map((v, j) => Math.abs(v - R.J[i][j]))));
    const F0 = q.n * q.h + q.h * q.h + q.h * q.m, ms = [1, 2, 4, 8, 16, 32, 64, 128];
    const cf = ms.map((m) => { const f0 = q.n * q.h + q.h * q.h + q.h * m; return f0 * (1 + q.n); }), cr = ms.map((m) => { const f0 = q.n * q.h + q.h * q.h + q.h * m; return f0 * (1 + m); });
    const win = F.macs < R.macs ? "jacfwd" : F.macs > R.macs ? "jacrev" : "eşit";
    const P = [`f: R^${q.n} → R^${q.m}. İki yol aynı ${q.m} × ${q.n} Jacobian'ı veriyor (en büyük fark ${fmt(diff, 2)}). Bir ileri geçiş ${F0} çarpma–toplama.`];
    P.push(`jacfwd: 1 + n = ${1 + q.n} geçiş eşdeğeri, ${F.macs} işlem; jacrev: 1 + m = ${1 + q.m} geçiş eşdeğeri, ${R.macs} işlem. Daha ucuz: <b>${win}</b>. Kural: girdi azsa (n < m) ileri mod, çıktı azsa (m < n) ters mod. Skaler kayıpta (m = 1) ters mod tek geri geçişle tüm gradyanı verir.`);
    P.push(`Ters mod ${R.stored} ara aktivasyonu saklamak zorunda (VJP'ler tanh'ın türevini onlardan okur); ileri mod hiçbir şey saklamaz, teğeti değerle birlikte taşır.`);
    return {
      plots: [{ title: `İşlem sayısı ve çıktı boyutu m (n = ${q.n}, h = ${q.h})`, xlabel: "m (log₂)", ylabel: "çarpma–toplama", ylog: true, series: [
        { name: "jacfwd: (1 + n)·F", x: ms.map(Math.log2), y: cf, c: 0 }, { name: "jacrev: (1 + m)·F", x: ms.map(Math.log2), y: cr, c: 1 }],
        marks: [{ x: Math.log2(q.m), label: "m" }, { x: Math.log2(q.n), label: "m = n" }], xticks: ms.map((m) => [Math.log2(m), String(m)]) },
        { title: "Jacobian (ilk 16 × 16), |J|", xlabel: "girdi j", ylabel: "çıktı i", size: { w: 300, h: 280 }, side: true, img: (() => { const a = Math.min(16, q.m), b = Math.min(16, q.n), M = Math.max(1e-9, ...R.J.slice(0, a).flatMap((row) => row.slice(0, b).map(Math.abs))); return { w: b, h: a, data: R.J.slice(0, a).flatMap((row) => row.slice(0, b).map((v) => Math.abs(v) / M)) }; })(), series: [] }],
      stats: [
        { label: "max |J_fwd − J_rev|", value: diff },
        { label: "İşlem: jacfwd / jacrev", value: `${F.macs} / ${R.macs}` },
        { label: "Ters modun sakladığı aktivasyon", value: R.stored },
      ],
      note: P.join(" "),
    };
  },
};
