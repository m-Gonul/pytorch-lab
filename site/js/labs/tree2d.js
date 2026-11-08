// M2.4 — CART karar ağacı, 2B ikili sınıflandırma: derinlik, yaprak boyu, safsızlık ölçütü; eksene paralel bölgeler ve eğitim/test doğruluğu.
import { fmt, range } from "./plot.js";

export const IMP = [(p) => 1 - p * p - (1 - p) * (1 - p), (p) => (p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p))), (p) => Math.min(p, 1 - p)];
export function dataset(kind, n, noise, r) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const x = [4 * r() - 2, 4 * r() - 2]; let y;
    if (kind === 0) y = x[0] + 0.3 * x[1] > 0.2 ? 1 : 0; // eğik doğru
    else if (kind === 1) y = x[0] * x[1] > 0 ? 1 : 0; // XOR
    else if (kind === 2) y = Math.hypot(...x) < 1.2 ? 1 : 0; // halka
    else y = x[0] > -0.5 && x[1] > -0.8 ? 1 : 0; // eksene paralel köşe
    if (r() < noise) y = 1 - y;
    X.push(x); Y.push(y);
  }
  return { X, Y };
}
export function build(X, Y, idx, depth, q) {
  const n = idx.length, pos = idx.reduce((s, i) => s + Y[i], 0), p = pos / n;
  const leaf = { leaf: true, p, n };
  if (depth >= q.depth || n < 2 * q.minLeaf || pos === 0 || pos === n) return leaf;
  const imp = IMP[q.crit]; let best = null;
  for (let j = 0; j < 2; j++) {
    const s = [...idx].sort((a, b) => X[a][j] - X[b][j]); let lp = 0;
    for (let k = 0; k < n - 1; k++) {
      lp += Y[s[k]]; const nl = k + 1, nr = n - nl;
      if (nl < q.minLeaf || nr < q.minLeaf || X[s[k]][j] === X[s[k + 1]][j]) continue;
      const sc = (nl / n) * imp(lp / nl) + (nr / n) * imp((pos - lp) / nr);
      if (!best || sc < best.sc - 1e-12) best = { sc, j, t: (X[s[k]][j] + X[s[k + 1]][j]) / 2 };
    }
  }
  if (!best || best.sc >= imp(p) - 1e-12) return leaf; // kazanç yok
  const L = idx.filter((i) => X[i][best.j] <= best.t), R = idx.filter((i) => X[i][best.j] > best.t);
  return { j: best.j, t: best.t, gain: imp(p) - best.sc, n, L: build(X, Y, L, depth + 1, q), R: build(X, Y, R, depth + 1, q) };
}
export const pred = (T, x) => (T.leaf ? T.p : pred(x[T.j] <= T.t ? T.L : T.R, x));
const count = (T) => (T.leaf ? 1 : count(T.L) + count(T.R));
const deep = (T) => (T.leaf ? 0 : 1 + Math.max(deep(T.L), deep(T.R)));
function rects(T, box, out) { if (T.leaf) { out.push(box); return; } const [x0, x1, y0, y1] = box; if (T.j === 0) { rects(T.L, [x0, T.t, y0, y1], out); rects(T.R, [T.t, x1, y0, y1], out); } else { rects(T.L, [x0, x1, y0, T.t], out); rects(T.R, [x0, x1, T.t, y1], out); } }

export default {
  params: [
    { id: "data", label: "Veri", options: ["eğik doğru sınır", "XOR", "daire", "eksene paralel köşe"], value: 0 },
    { id: "depth", label: "En büyük derinlik", min: 1, max: 12, value: 4, int: true },
    { id: "minLeaf", label: "Yaprakta en az örnek", min: 1, max: 30, value: 1, int: true },
    { id: "crit", label: "Safsızlık ölçütü", options: ["Gini", "entropi", "sınıflandırma hatası"], value: 0 },
    { id: "noise", label: "Etiket gürültüsü (ters çevrilen oran)", min: 0, max: 0.3, step: 0.01, value: 0.05 },
    { id: "n", label: "Eğitim örneği", min: 20, max: 400, value: 200, int: true },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), { X, Y } = dataset(q.data, n, q.noise, r), V = dataset(q.data, 1000, q.noise, r), qq = { ...q, depth: Math.round(q.depth), minLeaf: Math.round(q.minLeaf) };
    const T = build(X, Y, range(n), 0, qq);
    const G = 80, z = []; for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) z.push(pred(T, [-2 + ((i + 0.5) * 4) / G, -2 + ((j + 0.5) * 4) / G]));
    const B = []; rects(T, [-2, 2, -2, 2], B);
    const bx = [], by = []; for (const [x0, x1, y0, y1] of B) { bx.push(x0, x1, x1, x0, x0, NaN); by.push(y0, y0, y1, y1, y0, NaN); }
    const acc = (S, TT) => S.X.reduce((s, x, i) => s + ((pred(TT, x) > 0.5 ? 1 : 0) === S.Y[i] ? 1 : 0), 0) / S.X.length;
    const ds = range(12).map((d) => d + 1), tr = [], te = [];
    for (const d of ds) { const Td = build(X, Y, range(n), 0, { ...qq, depth: d }); tr.push(acc({ X, Y }, Td)); te.push(acc(V, Td)); }
    return {
      plots: [
        { title: `Ağacın bölgeleri (${count(T)} yaprak) — renk: yapraktaki sınıf-1 oranı`, xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-2, 2], ylim: [-2, 2], field: { x0: -2, x1: 2, y0: -2, y1: 2, nx: G, ny: G, z },
          series: [
            { name: "yaprak sınırları", x: bx, y: by, c: 4 },
            { name: "sınıf 0", x: X.filter((_, i) => !Y[i]).map((p) => p[0]), y: X.filter((_, i) => !Y[i]).map((p) => p[1]), kind: "scatter", c: 0, r: 2.4 },
            { name: "sınıf 1", x: X.filter((_, i) => Y[i]).map((p) => p[0]), y: X.filter((_, i) => Y[i]).map((p) => p[1]), kind: "scatter", c: 2, r: 2.4 },
          ] },
        { title: "Doğruluk, en büyük derinliğe göre", xlabel: "derinlik", ylabel: "doğruluk", series: [
          { name: "eğitim", x: ds, y: tr, c: 0, dots: true }, { name: "test (1000 yeni örnek)", x: ds, y: te, c: 2, dots: true },
        ], marks: [{ x: qq.depth, label: "seçili" }, { y: 1 - q.noise, label: "gürültü tavanı" }] },
      ],
      stats: [
        { label: "Yaprak sayısı / gerçek derinlik", value: `${count(T)} / ${deep(T)}` },
        { label: "Eğitim doğruluğu", value: `${fmt(acc({ X, Y }, T) * 100, 3)} %` },
        { label: "Test doğruluğu", value: `${fmt(acc(V, T) * 100, 3)} %` },
        { label: "Kök bölünmesi", value: T.leaf ? "yok" : `x${T.j + 1} ≤ ${fmt(T.t, 3)} (kazanç ${fmt(T.gain, 3)})` },
      ],
      note: `Her bölünme bir eksene paraleldir; bölgeler dikdörtgendir. ${q.data === 0 ? "Eğik bir sınırı yaklaştırmak için ağaç <b>merdiven</b> çizer: derinlik arttıkça basamaklar incelir ama her basamak örnek harcar. " : q.data === 1 ? "XOR'da ilk bölünmenin kazancı neredeyse sıfırdır (her yarı hâlâ %50–50): açgözlü algoritma ileriyi göremez. Yine de derinlik 2'de doğru yapıyı bulabilir — gürültü sayesinde ya da şansla. " : q.data === 3 ? "Eksene paralel bir sınır ağacın en kolay problemidir: iki bölünme yeter. " : ""}Derinlik arttıkça eğitim doğruluğu 1'e gider (her örnek kendi yaprağında), test doğruluğu ise bir noktadan sonra düşer: gürültüyü ezberleyen yapraklar. ${q.crit === 2 ? "<b>Sınıflandırma hatası</b> ölçütü p'ye göre kesin içbükey (strictly concave) değildir: birçok bölünmeye aynı skoru verir ve büyümeyi erken durdurabilir — CART bu yüzden Gini ya da entropi kullanır." : ""}`,
    };
  },
};
