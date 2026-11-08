// M2.7 — k nasıl seçilir, k-means ne zaman başarısız olur: dirsek (inertia) ve siluet; küresel olmayan, farklı boyutlu, uzamış kümeler.
import { fmt, range } from "./plot.js";

const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
export function data(kind, r) {
  const X = [], T = [];
  const blob = (c, s, n, t, ax = [1, 1]) => { for (let i = 0; i < n; i++) { X.push([c[0] + s * ax[0] * r.normal(), c[1] + s * ax[1] * r.normal()]); T.push(t); } };
  if (kind === 0) { [[-2, -2], [2, -2], [-2, 2], [2, 2]].forEach((c, t) => blob(c, 0.6, 60, t)); }
  else if (kind === 1) { blob([-1.5, 0], 1.3, 200, 0); blob([2.6, 1.8], 0.25, 30, 1); blob([2.6, -1.8], 0.25, 30, 2); }
  else if (kind === 2) { blob([0, -1.2], 0.35, 100, 0, [6, 1]); blob([0, 1.2], 0.35, 100, 1, [6, 1]); }
  else { for (let i = 0; i < 240; i++) { const y = i % 2, t = Math.PI * r(), p = y ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25] : [-Math.cos(t) + 0.5, -Math.sin(t) + 0.25]; X.push([2.2 * p[0] + 0.12 * r.normal(), 2.2 * p[1] + 0.12 * r.normal()]); T.push(y); } }
  return { X, T };
}
export function kmeans(X, k, r, restarts = 5) {
  let best = null;
  for (let s = 0; s < restarts; s++) {
    let M = [[...X[Math.floor(r() * X.length)]]];
    while (M.length < k) { const D = X.map((x) => Math.min(...M.map((m) => d2(x, m)))); let u = r() * D.reduce((a, b) => a + b, 0), i = 0; while ((u -= D[i]) > 0 && i < X.length - 1) i++; M.push([...X[i]]); }
    let lab = [];
    for (let it = 0; it < 50; it++) {
      const nl = X.map((x) => { let b = 0; for (let j = 1; j < k; j++) if (d2(x, M[j]) < d2(x, M[b])) b = j; return b; });
      const same = nl.every((v, i) => v === lab[i]); lab = nl;
      M = M.map((m, j) => { const P = X.filter((_, i) => lab[i] === j); return P.length ? [P.reduce((a, p) => a + p[0], 0) / P.length, P.reduce((a, p) => a + p[1], 0) / P.length] : m; });
      if (same) break;
    }
    const J = X.reduce((a, x, i) => a + d2(x, M[lab[i]]), 0);
    if (!best || J < best.J) best = { J, M, lab };
  }
  return best;
}
export function silhouette(X, lab, k) {
  if (k < 2) return 0;
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
    for (let j = 0; j < X.length; j++) if (j !== i) { const d = Math.sqrt(d2(X[i], X[j])); sum[lab[j]] += d; cnt[lab[j]]++; }
    const a = cnt[lab[i]] ? sum[lab[i]] / cnt[lab[i]] : 0; let b = Infinity;
    for (let c = 0; c < k; c++) if (c !== lab[i] && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
    s += cnt[lab[i]] && Number.isFinite(b) ? (b - a) / Math.max(a, b) : 0;
  }
  return s / X.length;
}

export default {
  params: [
    { id: "data", label: "Veri", options: ["4 eşit küresel küme", "farklı boyut ve yoğunluk", "uzamış (anizotrop) kümeler", "iki ay (küresel değil)"], value: 0 },
    { id: "k", label: "Gösterilen k", min: 1, max: 8, value: 4, int: true },
  ],
  seed: true,
  run(q, r) {
    const { X, T } = data(q.data, r), ks = [1, 2, 3, 4, 5, 6, 7, 8];
    const res = ks.map((k) => kmeans(X, k, r)), J = res.map((s) => s.J), S = res.map((s, i) => silhouette(X, s.lab, ks[i]));
    const R = res[Math.round(q.k) - 1], k = Math.round(q.k), trueK = new Set(T).size;
    // gerçek etiketlerle uyum (eşleştirmesiz saflık)
    let pur = 0; for (let c = 0; c < k; c++) { const cnt = {}; R.lab.forEach((l, i) => { if (l === c) cnt[T[i]] = (cnt[T[i]] || 0) + 1; }); pur += Math.max(0, ...Object.values(cnt)); } pur /= X.length;
    const cols = [0, 2, 1, 3, 5, 4, 0, 2];
    return {
      plots: [
        { title: `k = ${k}: kümeler ve merkezler`, xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-5, 5], ylim: [-5, 5], series: [
          ...range(k).map((c) => ({ name: c < 4 ? `küme ${c + 1}` : "", x: X.filter((_, i) => R.lab[i] === c).map((p) => p[0]), y: X.filter((_, i) => R.lab[i] === c).map((p) => p[1]), kind: "scatter", c: cols[c], r: 2.4 })),
          { name: "merkezler", x: R.M.map((m) => m[0]), y: R.M.map((m) => m[1]), kind: "scatter", c: 4, r: 7 },
        ] },
        { title: "Dirsek: inertia J(k) (5 başlatmanın en iyisi)", xlabel: "k", ylabel: "J", yzero: true, series: [{ name: "J(k)", x: ks, y: J, c: 0, dots: true }], marks: [{ x: trueK, label: "gerçek grup sayısı" }] },
        { title: "Ortalama siluet (büyük daha iyi)", xlabel: "k", ylabel: "siluet", series: [{ name: "s̄(k)", x: ks.slice(1), y: S.slice(1), c: 2, dots: true }], marks: [{ x: trueK, label: "gerçek" }] },
      ],
      stats: [
        { label: `Inertia J (k = ${k})`, value: R.J },
        { label: "Siluet en büyük k", value: ks[1 + S.slice(1).indexOf(Math.max(...S.slice(1)))] },
        { label: "Gerçek gruplarla saflık (her küme en kalabalık gerçek gruba)", value: `%${fmt(pur * 100, 3)}` },
      ],
      note: `J(k) k arttıkça <b>hep</b> düşer (k = n'de 0): tek başına k seçemez; "dirsek", eğimin belirgin azaldığı yerdir. Siluet her nokta için (b − a)/max(a, b): a kendi kümesine ortalama uzaklık, b en yakın başka kümeye; 1'e yakın = iyi ayrılmış. ${q.data === 1 ? "<b>Farklı boyutlar:</b> k-means en büyük kümeyi bölmeyi, küçük yoğun kümeleri birleştirmekten 'ucuz' bulur — kare uzaklık toplamı kalabalık kümeleri cezalandırır." : q.data === 2 ? "<b>Uzamış kümeler:</b> k-means her kümeyi küresel varsayar; uzun kümeleri ortadan keser (sınırlar dik açıyla). Her kümeye kendi kovaryansını veren GMM (M2.8) bunu çözer." : q.data === 3 ? "<b>İki ay:</b> merkez + uzaklık ile tanımlanan kümeler dışbükeydir (Voronoi hücreleri); ay biçimini hiçbir k yakalayamaz. Yoğunluk tabanlı (DBSCAN) ya da spektral kümeleme gerekir." : "Eşit, küresel kümeler k-means'in en iyi durumudur: dirsek ve siluet aynı k'yı gösterir."}`,
    };
  },
};
