// M5.4 — Weisfeiler–Lehman (1-WL) renk yenileme ve toplayıcının ayırt etme gücü (Xu ve ark. 2019, GIN). Başlangıçta tüm düğümler
// aynı renkte (özdeş özellik). Her turda yeni renk = (eski renk, komşu renklerinin toplayıcısı): toplam (çoklu küme, enjektif →
// 1-WL / GIN), ortalama (renk dağılımı → GCN/ortalama) ya da maks (renk kümesi). İki graf aynı renk histogramına sahipse bu
// toplayıcıyla hiçbir mesaj geçişli ağ onları ayıramaz.
import { fmt, range } from "./plot.js";

const PAIRS = [
  { name: "iki üçgen / altıgen", a: { n: 6, e: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]] }, b: { n: 6, e: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]] } },
  { name: "yıldız / yol (5 düğüm)", a: { n: 5, e: [[0, 1], [0, 2], [0, 3], [0, 4]] }, b: { n: 5, e: [[0, 1], [1, 2], [2, 3], [3, 4]] } },
  { name: "dal uzunlukları farklı iki ağaç", a: { n: 7, e: [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 6]] }, b: { n: 7, e: [[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [5, 6]] } },
];
const adjOf = (g) => { const a = range(g.n).map(() => []); g.e.forEach(([u, v]) => { a[u].push(v); a[v].push(u); }); return a; };
export function refine(pair, agg, iters) {
  const A = adjOf(pair.a), B = adjOf(pair.b), adj = [...A, ...B.map((l) => l.map((v) => v + pair.a.n))], N = adj.length;
  let col = new Array(N).fill(0); const hist = [];
  for (let it = 0; it <= iters; it++) {
    const K = Math.max(...col) + 1, h = (lo, hi) => { const c = new Array(K).fill(0); for (let i = lo; i < hi; i++) c[col[i]]++; return c.join(","); };
    hist.push({ a: h(0, pair.a.n), b: h(pair.a.n, N), K });
    if (it === iters) break;
    const sig = adj.map((nb, i) => {
      const cs = nb.map((j) => col[j]);
      if (agg === 0) return `${col[i]}|${cs.sort((x, y) => x - y).join(",")}`;
      const v = new Array(K).fill(0); cs.forEach((c) => v[c]++);
      if (agg === 1) return `${col[i]}|${cs.length ? v.map((x) => (x / cs.length).toFixed(6)).join(",") : "-"}`;
      return `${col[i]}|${v.map((x) => (x > 0 ? 1 : 0)).join(",")}`;
    });
    const ids = new Map(); col = sig.map((s) => { if (!ids.has(s)) ids.set(s, ids.size); return ids.get(s); });
  }
  return { col, hist, nA: pair.a.n };
}
const layout = (g, dx) => g.n === 6 && g.e.length === 6 && g.e[2][1] === 0 ? range(6).map((i) => (i < 3 ? [dx + Math.cos((i * 2 * Math.PI) / 3) * 0.8 - 0.9, Math.sin((i * 2 * Math.PI) / 3) * 0.8] : [dx + Math.cos(((i - 3) * 2 * Math.PI) / 3) * 0.8 + 0.9, Math.sin(((i - 3) * 2 * Math.PI) / 3) * 0.8]))
  : range(g.n).map((i) => [dx + Math.cos((i * 2 * Math.PI) / g.n) * 1.4, Math.sin((i * 2 * Math.PI) / g.n) * 1.4]);

export default {
  params: [
    { id: "pair", label: "Graf çifti", options: PAIRS.map((p) => p.name), value: 0 },
    { id: "agg", label: "Toplayıcı", options: ["toplam / çoklu küme (1-WL, GIN)", "ortalama (GCN, GraphSAGE-mean)", "maks (renk kümesi)"], value: 0 },
    { id: "iters", label: "Mesaj geçişi turu", min: 1, max: 5, value: 3, int: true },
  ],
  run(q) {
    const pair = PAIRS[q.pair], R = refine(pair, q.agg, q.iters), last = R.hist.at(-1), same = last.a === last.b;
    const firstDiff = R.hist.findIndex((h) => h.a !== h.b), pa = layout(pair.a, -2.2), pb = layout(pair.b, 2.2), pos = [...pa, ...pb];
    const edges = (g, P) => g.e.flatMap(([u, v]) => [[P[u], P[v]]]);
    const E = [...edges(pair.a, pa), ...edges(pair.b, pb)], ex = E.flatMap(([p1, p2]) => [p1[0], p2[0], NaN]), ey = E.flatMap(([p1, p2]) => [p1[1], p2[1], NaN]);
    const K = Math.max(...R.col) + 1, series = [{ name: "kenarlar", x: ex, y: ey, c: 4 }, ...range(Math.min(K, 6)).map((k) => ({ name: `renk ${k}`, x: pos.filter((_, i) => R.col[i] % 6 === k).map((p) => p[0]), y: pos.filter((_, i) => R.col[i] % 6 === k).map((p) => p[1]), kind: "scatter", c: k, r: 7 }))];
    const name = ["toplam (1-WL)", "ortalama", "maks"][q.agg];
    const P = [`${q.iters} tur ${name} yenilemesinden sonra ${K} renk. Sol graf renk histogramı [${last.a}], sağ graf [${last.b}]: ${same ? "<b>aynı</b> — bu toplayıcıyla hiçbir mesaj geçişli ağ bu iki grafı ayıramaz" : `<b>farklı</b> (ilk ${firstDiff}. turda ayrıştı)`}.`];
    if (q.pair === 0) P.push("İki üçgen ve altıgen: her düğümün derecesi 2, her komşusunun derecesi 2… 1-WL testi her turda aynı renkleri verir. Döngü uzunluğu gibi küresel bir özelliği yerel mesaj geçişi göremez; daha güçlü (ör. k-WL, alt graf sayımı) yöntemler gerekir.");
    if (q.pair === 1 && q.agg !== 0) P.push(`Başlangıçta herkes aynı renkte olduğu için ${q.agg === 1 ? "komşu renklerinin ortalaması" : "komşu renk kümesi"} her düğümde aynıdır: derece bile görünmez. Toplam (çoklu küme) dereceyi sayar ve ilk turda ayırır. Xu ve ark.: en güçlü mesaj geçişli ağ, toplayıcısı çoklu kümelerde enjektif olandır (GIN).`);
    if (q.pair === 2) P.push("İki ağacın derece dizisi aynıdır (bir düğüm 3, diğerleri 2 ya da 1); farkı dal uzunluklarındadır. Toplam toplayıcı bunu birkaç turda (bilgi dal boyunca yayıldıkça) görür.");
    return {
      plots: [{ title: `Renkler ${q.iters}. turdan sonra (sol: A, sağ: B)`, xlabel: "", ylabel: "", xlim: [-4, 4], ylim: [-2, 2], size: { w: 620, h: 300 }, wide: true, series }],
      stats: [
        { label: "Renk sayısı", value: K },
        { label: "Histogramlar aynı mı?", value: same ? "evet (ayırt edilemez)" : "hayır (ayırt edilir)" },
        { label: "İlk ayrıştığı tur", value: firstDiff < 0 ? "—" : firstDiff },
      ],
      note: P.join(" "),
    };
  },
};
