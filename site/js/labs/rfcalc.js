// M4.2 — Alıcı alan: katman katman r ← r + (k − 1)·j, j ← j·s; ve "etkin" alıcı alan (Luo ve ark. 2016): son katmanın merkez
// nöronunun girdiye göre gradyanı. Teorik alan doğrusal büyür, etkin alan ≈ Gauss ve genişliği ancak √(katman) ile büyür.
import { fmt, range } from "./plot.js";

export function layers(q) {
  const L = [{ name: `stem ${q.k0}×${q.k0}/${q.s0}`, k: q.k0, s: q.s0 }];
  for (let st = 0; st < q.stages; st++) {
    if (st > 0 || q.s0 === 1) L.push({ name: `havuz 2×2/2 (aşama ${st + 1})`, k: 2, s: 2, pool: true });
    for (let c = 0; c < q.convs; c++) L.push({ name: `conv ${q.k}×${q.k} (aşama ${st + 1})`, k: q.k, s: 1, d: q.dil && c === q.convs - 1 ? 2 : 1 });
  }
  return L;
}
export function receptive(L, n0 = 224) {
  let r = 1, j = 1, n = n0; const rows = [];
  for (const l of L) { const ke = (l.d || 1) * (l.k - 1) + 1; r += (ke - 1) * j; j *= l.s; n = Math.floor((n + 2 * Math.floor((ke - 1) / 2) - ke) / l.s) + 1; rows.push({ ...l, r, j, n }); }
  return rows;
}
// 1-B etkin alıcı alan: tek tip ağırlıklı çekirdeklerle geri geçiş (her katmanda gradyan çekirdeğin eşleniğiyle yayılır)
export function erf(L, n0) {
  const sizes = [n0]; for (const l of L) { const ke = (l.d || 1) * (l.k - 1) + 1; sizes.push(Math.floor((sizes[sizes.length - 1] + 2 * Math.floor((ke - 1) / 2) - ke) / l.s) + 1); }
  let g = new Array(sizes[sizes.length - 1]).fill(0); g[Math.floor(g.length / 2)] = 1;
  for (let li = L.length - 1; li >= 0; li--) {
    const l = L[li], d = l.d || 1, ke = d * (l.k - 1) + 1, P = Math.floor((ke - 1) / 2), nin = sizes[li], gin = new Array(nin).fill(0);
    for (let t = 0; t < g.length; t++) { if (!g[t]) continue; for (let u = 0; u < l.k; u++) { const i = t * l.s - P + u * d; if (i >= 0 && i < nin) gin[i] += g[t] / l.k; } }
    g = gin;
  }
  const s = g.reduce((a, b) => a + b, 0) || 1; return g.map((v) => v / s);
}

export default {
  params: [
    { id: "k0", label: "Giriş (stem) çekirdeği", min: 1, max: 11, value: 7, int: true, step: 2 },
    { id: "s0", label: "Giriş adımı", min: 1, max: 4, value: 2, int: true },
    { id: "stages", label: "Aşama sayısı (aralarında 2×2 havuzlama)", min: 1, max: 5, value: 4, int: true },
    { id: "convs", label: "Aşama başına konvolüsyon", min: 1, max: 4, value: 2, int: true },
    { id: "k", label: "Konvolüsyon çekirdeği", options: ["3 × 3", "5 × 5"], value: 0 },
    { id: "dil", label: "Her aşamanın son konvolüsyonu genişlemeli (d = 2)", options: ["hayır", "evet"], value: 0 },
  ],
  validate(q) { return q.k0 % 2 === 0 ? [{ level: "error", msg: "Giriş çekirdeği tek sayı olmalı (simetrik 'same' dolgu için)." }] : []; },
  run(q0) {
    const q = { ...q0, k: q0.k === 0 ? 3 : 5 }, L = layers(q), R = receptive(L, 224), last = R[R.length - 1], g = erf(L, 224);
    const xs = range(224), c = g.reduce((a, v, i) => a + v * i, 0), sd = Math.sqrt(g.reduce((a, v, i) => a + v * (i - c) ** 2, 0)), mx = Math.max(...g);
    const lo = Math.max(0, c - last.r / 2), hi = Math.min(223, c + last.r / 2);
    return {
      plots: [
        { title: "Teorik alıcı alan ve çıktı boyu, katmana göre", xlabel: "katman", ylabel: "piksel", ylog: true, series: [{ name: "alıcı alan r", x: R.map((_, i) => i + 1), y: R.map((v) => v.r), c: 0, dots: true }, { name: "harita boyu (224 girdiden)", x: R.map((_, i) => i + 1), y: R.map((v) => Math.max(1, v.n)), c: 2, dots: true }, { name: "toplam adım j", x: R.map((_, i) => i + 1), y: R.map((v) => v.j), c: 4, dash: true }] },
        { title: "Son katmanın merkez nöronu: girdiye göre gradyan (1-B kesit)", xlabel: "girdi konumu", ylabel: "pay", ylim: [0, mx * 1.15], series: [{ name: "etkin alıcı alan", x: xs, y: g, c: 0 }, { name: "teorik alanın sınırları", x: [lo, lo, NaN, hi, hi], y: [0, mx * 1.1, NaN, mx * 1.1, 0], c: 2, dash: true }] },
      ],
      stats: [
        { label: "Katman sayısı (havuz dahil)", value: L.length },
        { label: "Teorik alıcı alan", value: `${last.r} piksel` },
        { label: "Son harita boyu / toplam adım", value: `${last.n} / ${last.j}` },
        { label: "Etkin alanın std'si (±2σ genişliği)", value: `${fmt(sd, 3)} (${fmt(4 * sd, 3)} piksel)` },
        { label: "±2σ genişliği / teorik alan", value: `%${fmt((400 * sd) / last.r, 3)}` },
      ],
      note: `Teorik alıcı alan (dolgu sınırına çarpmadan) katman katman r ← r + (k − 1)·j ile büyür; j önceki adımların çarpımıdır. Havuzlama ve adımlar j'yi büyüttüğü için sonraki her 3×3 konvolüsyon alanı daha çok genişletir. Ama gradyan bu alana eşit dağılmaz: merkezdeki pikseller çok daha fazla yoldan çıktıya bağlıdır, katkı Gauss benzeri bir tepe oluşturur (merkezi limit teoremi: art arda konvolüsyonlar). Bu basit doğrusal modelde ±2σ genişliği teorik alanın %${fmt((400 * sd) / last.r, 2)}'i; aşama başına konvolüsyon sayısını artır: teorik alan katman sayısıyla doğrusal, etkin genişlik ise yaklaşık karekökle büyüdüğü için oran düşer. Luo ve ark. (2016) gerçek ağlarda etkin alanın teorik alanın küçük bir kısmını kapladığını ve eğitimle büyüdüğünü gösterdi.${last.r > 224 ? " <b>Teorik alan 224 piksellik girdiden büyük:</b> kenarlarda dolguya taşar." : ""} ${q.s0 > 1 || q.stages > 1 ? " Profildeki testere dişi dalgalanma gerçektir: adımlı katmanlarda (havuzlama, adımlı giriş) bazı girdi pikselleri komşularından daha çok yoldan çıktıya bağlanır — 'dama tahtası' etkisinin 1-B karşılığı." : ""}${q.dil ? "Genişlemeli konvolüsyon parametre eklemeden alanı büyütür." : ""}`,
    };
  },
};
