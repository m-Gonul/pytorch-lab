// M4.15 — Daha yüksek çözünürlükte ince ayar: g × g ızgarada öğrenilmiş konum embedding'leri g' × g' ızgaraya taşınmalı.
// "Öğrenilmiş" embedding'ler burada pürüzsüz bir 2B fonksiyonla temsil edilir (rastgele frekanslı sinüs karışımı, D boyut).
// Yöntemler: bilineer (align_corners = False), en yakın komşu, ve yanlış yol: ham sırayla (raster) ilk g'² vektörü kullanmak
// ya da eksikleri tekrarlamak. Ölçüt: aynı fonksiyonun yeni ızgaradaki (görüntüdeki aynı yere karşılık gelen) değerine kosinüs.
import { fmt, range } from "./plot.js";

export function field(D, r) { return range(D).map(() => ({ a: r.normal() * 2.2, b: r.normal() * 2.2, ph: r() * 6.283 })); }
// sürekli koordinat (u, v) ∈ [0, 1]² → D boyutlu vektör
export const evalAt = (F, u, v) => F.map((f) => Math.sin(f.a * u * 3.1416 + f.b * v * 3.1416 + f.ph));
export function grid(F, g) { return range(g * g).map((k) => evalAt(F, ((k % g) + 0.5) / g, (Math.floor(k / g) + 0.5) / g)); }
export function bilinear(P, g, G) {
  const s = (i) => Math.min(g - 1, Math.max(0, ((i + 0.5) * g) / G - 0.5));
  return range(G * G).map((k) => { const y = s(Math.floor(k / G)), x = s(k % G), y0 = Math.floor(y), x0 = Math.floor(x), y1 = Math.min(g - 1, y0 + 1), x1 = Math.min(g - 1, x0 + 1), fy = y - y0, fx = x - x0;
    return P[0].map((_, d) => (1 - fy) * ((1 - fx) * P[y0 * g + x0][d] + fx * P[y0 * g + x1][d]) + fy * ((1 - fx) * P[y1 * g + x0][d] + fx * P[y1 * g + x1][d])); });
}
export function nearest(P, g, G) { const s = (i) => Math.min(g - 1, Math.floor(((i + 0.5) * g) / G)); return range(G * G).map((k) => P[s(Math.floor(k / G)) * g + s(k % G)]); }
export function raster(P, g, G) { return range(G * G).map((k) => P[k % (g * g)]); }
const cos = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b) + 1e-12);

export default {
  params: [
    { id: "g", label: "Ön eğitim ızgarası g (224/16 = 14)", min: 4, max: 32, value: 14, int: true },
    { id: "G", label: "İnce ayar ızgarası g' (384/16 = 24)", min: 4, max: 48, value: 24, int: true },
    { id: "method", label: "Uyarlama yöntemi", options: ["bilineer (2B)", "en yakın komşu", "ham sıra (raster) — yanlış"], value: 0 },
    { id: "D", label: "Embedding boyutu D", min: 4, max: 256, value: 64, int: true, log: true },
    { id: "ref", label: "Referans yama (konum, 0–1 ölçekte satır = sütun)", min: 0, max: 1, step: 0.05, value: 0.3 },
  ],
  seed: true,
  run(q, r) {
    const F = field(q.D, r), P = grid(F, q.g), ideal = grid(F, q.G), Q = [bilinear, nearest, raster][q.method](P, q.g, q.G);
    const sims = Q.map((v, k) => cos(v, ideal[k])), mean = sims.reduce((a, b) => a + b, 0) / sims.length, worst = Math.min(...sims);
    const others = [bilinear, nearest, raster].map((f) => { const Z = f(P, q.g, q.G); return Z.reduce((s, v, k) => s + cos(v, ideal[k]), 0) / Z.length; });
    const ri = Math.min(q.g - 1, Math.floor(q.ref * q.g)), rI = Math.min(q.G - 1, Math.floor(q.ref * q.G));
    const mapOld = P.map((v) => (cos(v, P[ri * q.g + ri]) + 1) / 2), mapNew = Q.map((v) => (cos(v, Q[rI * q.G + rI]) + 1) / 2), mapIdeal = ideal.map((v) => (cos(v, ideal[rI * q.G + rI]) + 1) / 2);
    const name = ["bilineer", "en yakın komşu", "ham sıra"][q.method];
    const Pn = [`${q.g} × ${q.g} = ${q.g * q.g} konum embedding'i, ${q.G} × ${q.G} = ${q.G * q.G} konuma uyarlanıyor (${name}). Görüntüdeki aynı yere karşılık gelen ideal embedding'e ortalama kosinüs ${fmt(mean, 4)}, en kötü konumda ${fmt(worst, 3)}. Karşılaştırma: bilineer ${fmt(others[0], 4)}, en yakın ${fmt(others[1], 4)}, ham sıra ${fmt(others[2], 4)}.`];
    if (q.method === 2) Pn.push(`Ham sırayla kullanmak (yeni k. konuma eski k. vektörü vermek) 2B yapıyı bozar: eski ızgarada satır uzunluğu ${q.g}, yenide ${q.G} olduğu için aynı indis görüntünün başka bir yerine düşer; benzerlik haritası kayar ve parçalanır. ViT makalesinin uyarısı da budur: yalnız bu uyarlama adımında 2B yapı elle eklenir.`);
    else Pn.push(`Benzerlik haritası (seçili yamanın konum embedding'inin diğerlerine kosinüsü) uyarlamadan sonra aynı uzamsal deseni korur. ${q.G > q.g ? "Büyütmede bilineer, en yakın komşuya göre pürüzsüzdür; en yakın komşu bloklu kopyalar üretir." : "Küçültmede bilgi kaybı kaçınılmaz: bazı konumlar ortalanır."}`);
    const im = (m, n) => ({ w: n, h: n, data: m });
    return {
      plots: [
        { title: `Ön eğitim (${q.g} × ${q.g}): benzerlik haritası`, xlabel: "sütun", ylabel: "satır", size: { w: 220, h: 210 }, img: im(mapOld, q.g), side: true, series: [] },
        { title: `Uyarlanmış (${q.G} × ${q.G}, ${name})`, xlabel: "sütun", ylabel: "satır", size: { w: 220, h: 210 }, img: im(mapNew, q.G), side: true, series: [] },
        { title: `İdeal (${q.G} × ${q.G})`, xlabel: "sütun", ylabel: "satır", size: { w: 220, h: 210 }, img: im(mapIdeal, q.G), side: true, series: [] },
      ],
      stats: [
        { label: `Ortalama kosinüs (${name})`, value: mean },
        { label: "En kötü konum", value: worst },
        { label: "Bilineer / en yakın / ham sıra", value: `${fmt(others[0], 4)} / ${fmt(others[1], 4)} / ${fmt(others[2], 4)}` },
        { label: "Yeni dizi uzunluğu ([class] dahil)", value: q.G * q.G + 1 },
      ],
      note: Pn.join(" "),
    };
  },
};
