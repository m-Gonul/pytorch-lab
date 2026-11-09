// M4.12 — SGNS örtük olarak kaydırılmış PMI'yi çarpanlarına ayırır (Levy & Goldberg 2014). w2vlab'daki sentetik derlemden
// skip-gram çift sayımları #(w,c) çıkarılır; SGNS'in BEKLENEN amacı
//   L = Σ_{w,c} #(w,c)·softplus(−x_wc) + k·#(w)·P_n(c)·softplus(x_wc),  x_wc = W_w · C_c
// tam batch Adam ile d boyutta en aza indirilir. Her çift için ayrı optimum x* = log(#(w,c) / (k·#(w)·P_n(c))); α = 1'de bu
// PMI(w,c) − log k'dır. Ayrıca SPPMI = max(PMI − log k, 0) matrisinin tekil değerleri (Jacobi özdeğer) ve rank-d SVD hatası.
import { fmt, range } from "./plot.js";
import { corpus } from "./w2vlab.js";

const sp = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x))), sig = (x) => 1 / (1 + Math.exp(-x));

export function counts(S, win) {
  const vocab = [...new Set(S.flat())].sort(), id = Object.fromEntries(vocab.map((w, i) => [w, i])), V = vocab.length;
  const N = range(V).map(() => new Float64Array(V));
  for (const s of S) s.forEach((w, i) => { for (let j = Math.max(0, i - win); j <= Math.min(s.length - 1, i + win); j++) if (j !== i) N[id[w]][id[s[j]]]++; });
  return { vocab, V, N };
}
// simetrik matrisin özdeğerleri (döngüsel Jacobi)
export function symEig(A0) {
  const n = A0.length, A = A0.map((r) => Float64Array.from(r));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-15) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
    }
  }
  return range(n).map((i) => A[i][i]).sort((a, b) => b - a);
}
export function singular(M) { const n = M.length, G = range(n).map((i) => range(n).map((j) => { let s = 0; for (let k = 0; k < n; k++) s += M[k][i] * M[k][j]; return s; })); return symEig(G).map((x) => Math.sqrt(Math.max(0, x))); }

export function fitSGNS(N, k, alpha, d, iters, r) {
  const V = N.length, nw = N.map((row) => row.reduce((a, b) => a + b, 0)), nc = range(V).map((j) => N.reduce((a, row) => a + row[j], 0)), D = nw.reduce((a, b) => a + b, 0);
  const pa = nc.map((c) => c ** alpha), za = pa.reduce((a, b) => a + b, 0), Pn = pa.map((x) => x / za);
  const neg = N.map((_, i) => Pn.map((p) => (k * nw[i] * p) / D)), pos = N.map((row) => Array.from(row, (x) => x / D));
  const W = range(V).map(() => range(d).map(() => 0.1 * r.normal())), C = range(V).map(() => range(d).map(() => 0.1 * r.normal()));
  const mW = W.map((w) => w.map(() => 0)), vW = W.map((w) => w.map(() => 0)), mC = C.map((w) => w.map(() => 0)), vC = C.map((w) => w.map(() => 0));
  const lr = 0.05, b1 = 0.9, b2 = 0.999, loss = [];
  const X = range(V).map(() => new Float64Array(V));
  for (let it = 1; it <= iters; it++) {
    let L = 0; const G = range(V).map(() => new Float64Array(V));
    for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) { let x = 0; for (let t = 0; t < d; t++) x += W[i][t] * C[j][t]; X[i][j] = x; L += pos[i][j] * sp(-x) + neg[i][j] * sp(x); G[i][j] = -pos[i][j] * sig(-x) + neg[i][j] * sig(x); }
    if (it === 1 || it % Math.max(1, Math.floor(iters / 60)) === 0 || it === iters) loss.push([it, L]);
    const gW = W.map((_, i) => range(d).map((t) => { let s = 0; for (let j = 0; j < V; j++) s += G[i][j] * C[j][t]; return s; }));
    const gC = C.map((_, j) => range(d).map((t) => { let s = 0; for (let i = 0; i < V; i++) s += G[i][j] * W[i][t]; return s; }));
    const c1 = 1 - b1 ** it, c2 = 1 - b2 ** it;
    for (const [P, g, m, v] of [[W, gW, mW, vW], [C, gC, mC, vC]]) for (let i = 0; i < V; i++) for (let t = 0; t < d; t++) {
      m[i][t] = b1 * m[i][t] + (1 - b1) * g[i][t]; v[i][t] = b2 * v[i][t] + (1 - b2) * g[i][t] ** 2; P[i][t] -= (lr * m[i][t]) / c1 / (Math.sqrt(v[i][t] / c2) + 1e-8);
    }
  }
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) { let x = 0; for (let t = 0; t < d; t++) x += W[i][t] * C[j][t]; X[i][j] = x; }
  return { X, loss, nw, nc, D, Pn };
}
const corr = (a, b) => { const n = a.length, ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb + 1e-30); };

export default {
  params: [
    { id: "d", label: "Gömme boyutu d", min: 1, max: 48, value: 8, int: true },
    { id: "k", label: "Negatif sayısı k", min: 1, max: 15, value: 5, int: true },
    { id: "alpha", label: "Gürültü dağılımı", options: ["α = 1: P_n = bağlam frekansı", "α = 0.75: düzleştirilmiş"], value: 0 },
    { id: "win", label: "Pencere (her yana)", min: 1, max: 4, value: 2, int: true },
    { id: "iters", label: "Adam adımı (tam batch)", min: 20, max: 1500, value: 400, int: true, log: true },
    { id: "n", label: "Derlemdeki cümle sayısı", min: 200, max: 4000, value: 1500, int: true, log: true },
  ],
  seed: true,
  validate(q) { return q.d * q.iters > 30000 ? [{ level: "warn", msg: "d × adım büyük: hesap birkaç saniye sürebilir." }] : []; },
  run(q, r) {
    const { vocab, V, N } = counts(corpus(r, q.n), q.win), R = fitSGNS(N, q.k, q.alpha ? 0.75 : 1, q.d, q.iters, r);
    const obs = [], xs = [], ts = [], ps = [], unobs = [];
    for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) {
      if (N[i][j] > 0) { obs.push([i, j]); xs.push(R.X[i][j]); ts.push(Math.log(N[i][j] / (q.k * R.nw[i] * R.Pn[j]))); ps.push(Math.log((N[i][j] * R.D) / (R.nw[i] * R.nc[j])) - Math.log(q.k)); }
      else unobs.push(R.X[i][j]);
    }
    const SP = N.map((row, i) => Array.from(row, (x, j) => (x > 0 ? Math.max(0, Math.log((x * R.D) / (R.nw[i] * R.nc[j])) - Math.log(q.k)) : 0)));
    const sv = singular(SP), tot = sv.reduce((s, x) => s + x * x, 0), resid = sv.slice(q.d).reduce((s, x) => s + x * x, 0);
    const Lopt = obs.reduce((s, [i, j], n) => s + (N[i][j] / R.D) * sp(-ts[n]) + ((q.k * R.nw[i] * R.Pn[j]) / R.D) * sp(ts[n]), 0);
    const cT = corr(xs, ts), cP = corr(xs, ps), rmse = Math.sqrt(xs.reduce((s, x, n) => s + (x - ts[n]) ** 2, 0) / xs.length), Lend = R.loss.at(-1)[1];
    const lo = Math.min(...ts, ...xs) - 0.3, hi = Math.max(...ts, ...xs) + 0.3, rank = sv.filter((x) => x > 1e-8 * sv[0]).length;
    const P = [`Sözlük ${V} kelime, ${obs.length} gözlenen (w, c) çifti (${V * V} hücrenin %${fmt((100 * obs.length) / (V * V), 3)} kadarı). Gözlenen çiftlerde öğrenilen x = w·c ile hedef x* = log(#(w,c)/(k·#(w)·P_n(c))) arasındaki korelasyon ${fmt(cT, 3)}, kök ortalama kare fark ${fmt(rmse, 3)}.`];
    P.push(q.alpha ? `α = 0.75'te hedef artık PMI − log k değil: bağlam olasılığı düzleştirilmiş P_n ile değiştirildiği için nadir bağlamların PMI'si aşağı çekilir (Levy, Goldberg ve Dagan 2015 "bağlam dağılımı düzleştirme"). Öğrenilen x'in ham PMI − log k ile korelasyonu ${fmt(cP, 3)}.` : `α = 1'de P_n(c) = #(c)/|D| olduğundan hedef tam olarak PMI(w,c) − log k'dır.`);
    P.push(`Beklenen kayıp ${fmt(Lend, 4)}; her hücre bağımsız ayarlanabilseydi alt sınır ${fmt(Lopt, 4)} olurdu (gözlenmeyen çiftlerde x → −∞, ortalama öğrenilen x = ${fmt(unobs.reduce((s, x) => s + x, 0) / Math.max(1, unobs.length), 3)}). ${q.d >= rank ? `d = ${q.d}, SPPMI matrisinin rankına (${rank}) yetiyor: kısıt, boyuttan çok optimizasyon süresinden geliyor.` : `d = ${q.d} < rank(SPPMI) = ${rank}: W·Cᵀ en fazla rank ${q.d} olabildiği için hücreler bağımsız ayarlanamaz; SGNS ağırlıklı bir düşük-rank yaklaşım yapar (sık çiftlere daha çok önem verir). Karşılaştırma: SPPMI'nin rank-${q.d} SVD'si, matrisin enerjisinin (tekil değer kareleri toplamının) %${fmt(100 * (1 - resid / tot), 3)} kadarını tutuyor.`}`);
    return {
      plots: [
        { title: "Gözlenen çiftler: öğrenilen w·c ve hedef", xlabel: "hedef x* = log(#(w,c) / (k·#(w)·P_n(c)))", ylabel: "öğrenilen w·c", xlim: [lo, hi], ylim: [lo, hi], size: { w: 310, h: 300 }, series: [
          { name: "y = x", x: [lo, hi], y: [lo, hi], c: 4, dash: true },
          { name: "çiftler", x: ts, y: xs, kind: "scatter", c: 0, r: 2.5 }] },
        { title: "SPPMI tekil değerleri", xlabel: "sıra", ylabel: "tekil değer", yzero: true, series: [{ name: "σ_i(SPPMI)", x: range(sv.length).map((i) => i + 1), y: sv, kind: "bar", c: 2 }], marks: [{ x: q.d + 0.5, label: `d = ${q.d}` }] },
      ],
      stats: [
        { label: "corr(w·c, x*) — gözlenen çiftler", value: cT },
        { label: "corr(w·c, PMI − log k)", value: cP },
        { label: "Kök ortalama kare fark (w·c − x*)", value: rmse },
        { label: "Beklenen SGNS kaybı / alt sınır", value: `${fmt(Lend, 4)} / ${fmt(Lopt, 4)}` },
        { label: `SPPMI enerjisi, rank-${q.d} SVD'nin tuttuğu`, value: 1 - resid / tot },
        { label: "rank(SPPMI)", value: rank },
      ],
      note: P.join(" "),
    };
  },
};
