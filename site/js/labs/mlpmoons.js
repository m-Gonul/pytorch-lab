// M3.1 / M3.2 — Küçük bir MLP'yi iki ay verisinde eğit (matris biçiminde ileri/geri geçiş, elle türetilmiş).
// Karar sınırı, kayıp eğrisi ve aynı ağın micrograd'da kaç skaler düğüm üreteceği.
import { fmt, range } from "./plot.js";

export function moons(n, noise, r) { const X = [], Y = []; for (let i = 0; i < n; i++) { const y = i % 2, t = Math.PI * r(); const p = y ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)]; X.push([p[0] - 0.5 + noise * r.normal(), p[1] - 0.25 + noise * r.normal()]); Y.push(y); } return { X, Y }; }
const ACT = [
  { name: "tanh", f: Math.tanh, d: (z, a) => 1 - a * a },
  { name: "ReLU", f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0) },
  { name: "sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
];
export function init(sizes, r, scheme) { return sizes.slice(1).map((m, l) => { const n = sizes[l], s = scheme === 0 ? Math.sqrt(1 / n) : scheme === 1 ? Math.sqrt(2 / n) : 1; return { W: range(m).map(() => range(n).map(() => s * r.normal())), b: new Array(m).fill(0) }; }); }
// ileri: her katman z = W a + b, a = φ(z); son katman doğrusal (logit)
export function forward(P, x, act) { const as = [x], zs = []; let a = x; P.forEach((L, l) => { const z = L.W.map((w, i) => w.reduce((s, wij, j) => s + wij * a[j], L.b[i])); zs.push(z); a = l < P.length - 1 ? z.map(ACT[act].f) : z; as.push(a); }); return { as, zs }; }
export function step(P, X, Y, act, lr, lam, V, mom) {
  const G = P.map((L) => ({ W: L.W.map((w) => w.map(() => 0)), b: L.b.map(() => 0) })); let loss = 0, acc = 0;
  for (let i = 0; i < X.length; i++) {
    const { as, zs } = forward(P, X[i], act), z = as.at(-1)[0];
    loss += Math.max(z, 0) - Y[i] * z + Math.log1p(Math.exp(-Math.abs(z))); acc += (z > 0 ? 1 : 0) === Y[i] ? 1 : 0;
    let delta = [1 / (1 + Math.exp(-z)) - Y[i]]; // BP1: δᴸ = ∂C/∂z (BCE + sigmoid → p − y)
    for (let l = P.length - 1; l >= 0; l--) {
      for (let j = 0; j < delta.length; j++) { G[l].b[j] += delta[j]; for (let k = 0; k < as[l].length; k++) G[l].W[j][k] += delta[j] * as[l][k]; } // BP3, BP4
      if (l > 0) delta = as[l].map((_, k) => P[l].W.reduce((s, w, j) => s + w[k] * delta[j], 0) * ACT[act].d(zs[l - 1][k], as[l][k])); // BP2: δˡ = (Wˡ⁺¹ᵀ δˡ⁺¹) ⊙ φ′(zˡ)
    }
  }
  const n = X.length;
  P.forEach((L, l) => { L.W.forEach((w, j) => w.forEach((_, k) => { const g = G[l].W[j][k] / n + lam * w[k]; V[l].W[j][k] = mom * V[l].W[j][k] + g; w[k] -= lr * V[l].W[j][k]; })); L.b.forEach((_, j) => { const g = G[l].b[j] / n; V[l].b[j] = mom * V[l].b[j] + g; L.b[j] -= lr * V[l].b[j]; }); });
  return { loss: loss / n, acc: acc / n };
}

export default {
  params: [
    { id: "width", label: "Gizli katman genişliği", min: 1, max: 24, value: 12, int: true },
    { id: "depth", label: "Gizli katman sayısı", min: 0, max: 3, value: 2, int: true, help: "0: gizli katman yok — lojistik regresyon (doğrusal sınır)." },
    { id: "act", label: "Aktivasyon", options: ACT.map((a) => a.name), value: 0 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.005, max: 2, value: 0.1, log: true, hardMin: 1e-6 },
    { id: "steps", label: "Adım (tam batch, momentum 0.9)", min: 1, max: 600, value: 300, int: true, log: true },
    { id: "noise", label: "Veri gürültüsü", min: 0, max: 0.4, step: 0.01, value: 0.12 },
  ],
  seed: true,
  run(q, r) {
    const { X, Y } = moons(100, q.noise, r), V = moons(400, q.noise, r), sizes = [2, ...new Array(Math.round(q.depth)).fill(Math.round(q.width)), 1];
    const P = init(sizes, r, q.act === 1 ? 1 : 0), Vel = P.map((L) => ({ W: L.W.map((w) => w.map(() => 0)), b: L.b.map(() => 0) }));
    const T = Math.round(q.steps), hist = [];
    for (let t = 0; t < T; t++) { const s = step(P, X, Y, q.act, q.lr, 1e-4, Vel, 0.9); hist.push(s.loss); if (!Number.isFinite(s.loss)) break; }
    const R = 2, G = 60, z = []; for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const o = forward(P, [-R + ((i + 0.5) * 2 * R) / G, -1.6 + ((j + 0.5) * 3.2) / G], q.act).as.at(-1)[0]; z.push(1 / (1 + Math.exp(-o))); }
    const accOf = (S) => S.X.reduce((s, x, i) => s + ((forward(P, x, q.act).as.at(-1)[0] > 0 ? 1 : 0) === S.Y[i] ? 1 : 0), 0) / S.X.length;
    const nParam = P.reduce((s, L) => s + L.W.length * L.W[0].length + L.b.length, 0), nodes = 100 * P.reduce((s, L) => s + L.W.length * (2 * L.W[0].length + 1) + L.W.length, 0);
    return {
      plots: [
        { title: `Ağ ${sizes.join("–")}: P(y = 1 | x) ve eğitim verisi`, xlabel: "x₁", ylabel: "x₂", size: { w: 400, h: 330 }, xlim: [-R, R], ylim: [-1.6, 1.6], field: { x0: -R, x1: R, y0: -1.6, y1: 1.6, nx: G, ny: G, z },
          series: [0, 1].map((c) => ({ name: `sınıf ${c}`, x: X.filter((_, i) => Y[i] === c).map((p) => p[0]), y: X.filter((_, i) => Y[i] === c).map((p) => p[1]), kind: "scatter", c: c ? 2 : 0, r: 2.8 })) },
        { title: "Eğitim kaybı (BCE)", xlabel: "adım", ylabel: "kayıp", ylog: true, series: [{ name: "kayıp", x: range(hist.length), y: hist.map((v) => (Number.isFinite(v) ? Math.max(v, 1e-6) : NaN)), c: 0 }] },
      ],
      stats: [
        { label: "Eğitim / test doğruluğu", value: `%${fmt(accOf({ X, Y }) * 100, 3)} / %${fmt(accOf(V) * 100, 3)}` },
        { label: "Parametre sayısı", value: nParam },
        { label: "Aynı ileri geçiş micrograd'da (100 örnek): ~skaler düğüm", value: nodes },
        { label: "Son kayıp", value: hist.at(-1) },
      ],
      note: `${Math.round(q.depth) === 0 ? "Gizli katman yok: model lojistik regresyon, sınır bir doğru — ayları ayıramaz. " : ""}Her gizli katman girdiyi doğrusal olmayan yeni özniteliklere dönüştürür; son katman bu öznitelikler üzerinde bir lojistik regresyondur. Geri geçiş matris biçimindeki dört denklemi uygular: δᴸ = p − y; δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ φ′(zˡ); ∂C/∂bˡ = δˡ; ∂C/∂Wˡ = δˡ(aˡ⁻¹)ᵀ. ${q.act === 2 ? "Sigmoid: φ′ ≤ 1/4 ve çıktılar sıfır merkezli değil — derin ağda öğrenme belirgin yavaş (M3.5). " : ""}Aynı hesabı skaler bir motorda (micrograd) yapmak her çarpma ve toplama için bir Python nesnesi demektir: ~${fmt(nodes, 3)} düğüm, her adımda. Tensör motorları bunu birkaç matris çarpımına indirir (M3.3).`,
    };
  },
};
