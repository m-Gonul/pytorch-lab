// M4.14 — Sinüzoidal konum kodlaması (Vaswani ve ark. 2017): P[p, 2i] = sin(p·ω_i), P[p, 2i+1] = cos(p·ω_i), ω_i = base^(−2i/d).
// Isı haritası, seçilen boyut çiftinin dalgaları, iç çarpım P_p·P_q'nun yalnız p − q'ya bağlı olması ve "yalnız sinüs" kullanıldığında
// farklı konumların çakışması (en yakın iki farklı konumun uzaklığı).
import { fmt, range } from "./plot.js";

export function pe(n, d, base, sinOnly = false) {
  return range(n).map((p) => range(d).map((j) => { const i = Math.floor(j / 2), w = base ** (-(2 * i) / d); return j % 2 === 0 || sinOnly ? Math.sin(p * w) : Math.cos(p * w); }));
}
const dist = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));

export default {
  params: [
    { id: "d", label: "Boyut d", min: 8, max: 512, value: 128, int: true, log: true, why: "Çift olmalı." },
    { id: "n", label: "Konum sayısı (en büyük uzunluk)", min: 16, max: 512, value: 128, int: true, log: true },
    { id: "base", label: "Taban (makalede 10 000)", min: 10, max: 100000, value: 10000, log: true },
    { id: "pair", label: "İncelenen boyut çifti (2i, 2i+1): i", min: 0, max: 255, value: 25, int: true },
    { id: "p", label: "Konum p", min: 0, max: 511, value: 22, int: true },
    { id: "k", label: "Göreli uzaklık k (q = p + k)", min: 1, max: 200, value: 38, int: true },
    { id: "mode", label: "Kodlama", options: ["sin + cos (makale)", "yalnız sinüs"], value: 0 },
  ],
  validate(q) {
    const e = [];
    if (q.d % 2) e.push({ level: "error", msg: "Boyut d çift olmalı: her frekans bir (sin, cos) çifti kullanır." });
    if (q.pair >= q.d / 2) e.push({ level: "error", msg: `Boyut çifti i, d/2 = ${q.d / 2}'den küçük olmalı.` });
    if (q.p + q.k >= q.n) e.push({ level: "warn", msg: "p + k, konum sayısını aşıyor: ısı haritası dışında hesaplandı." });
    return e;
  },
  run(q) {
    const so = q.mode === 1, P = pe(Math.max(q.n, q.p + q.k + 1), q.d, q.base, so), Pn = P.slice(0, q.n), w = q.base ** (-(2 * q.pair) / q.d), lam = (2 * Math.PI) / w;
    let lo = Infinity, at = [0, 0];
    for (let a = 0; a < q.n; a++) for (let b = a + 1; b < q.n; b++) { const x = dist(Pn[a], Pn[b]); if (x < lo) { lo = x; at = [a, b]; } }
    const flat = Pn.flat(), ks = range(Math.min(q.n, 200)), simAt = (p0) => ks.map((k) => (p0 + k < P.length ? P[p0].reduce((s, x, j) => s + x * P[p0 + k][j], 0) : null));
    const s0 = simAt(0), s1 = simAt(Math.min(q.p, P.length - 1));
    const a = P[q.p], b = P[q.p + q.k], j0 = 2 * q.pair;
    const P1 = [`Boyut çifti i = ${q.pair} (boyut ${j0}, ${j0 + 1}): açısal frekans ω = ${q.base}^(−${2 * q.pair}/${q.d}) = ${fmt(w, 4)}, dalga boyu 2π/ω = ${fmt(lam, 5)} konum. p = ${q.p} ve p + k = ${q.p + q.k} konumlarında bu iki boyutun değerleri (${fmt(a[j0], 4)}, ${fmt(a[j0 + 1], 4)}) ve (${fmt(b[j0], 4)}, ${fmt(b[j0 + 1], 4)}). ${Math.abs(q.k - Math.round(lam)) <= 1 && lam > 2 ? `k dalga boyuna yakın: değerler <b>yaklaşık</b> aynı, ama dalga boyu tam sayı olmadığı için tam aynı değil.` : ""}`];
    let mir = null; for (let m = -3; m <= 3; m++) { const c = Math.round(Math.PI / w - q.p + m * lam); if (c >= 0 && c !== q.p && (mir == null || Math.abs(c - q.p) < Math.abs(mir - q.p))) mir = c; }
    const mtxt = mir == null ? "" : ` Bu frekansta p = ${q.p} konumunun "yansıması" p = ${mir} konumudur: sin değerleri ${fmt(Math.sin(q.p * w), 3)} ve ${fmt(Math.sin(mir * w), 3)} (sin a = sin(π − a)); kosinüsleri ${fmt(Math.cos(q.p * w), 3)} ve ${fmt(Math.cos(mir * w), 3)}.`;
    P1.push(so ? `<b>Yalnız sinüs</b>: tek bir frekansta sin(pω) iki farklı konumda aynı değeri alır.${mtxt} Tüm boyutlar birlikte hâlâ çoğunlukla ayırt edicidir (en yakın iki konumun uzaklığı ${fmt(lo, 3)}). Asıl kayıp göreli konumdadır: iç çarpım Σ sin(pω)sin((p+k)ω) = ½Σ[cos(kω) − cos((2p+k)ω)] artık p'ye de bağlıdır (sağdaki iki eğri ayrılıyor) ve P_(p+k), P_p'nin bir döndürmesi olarak yazılamaz.` : `En yakın iki farklı konum p = ${at[0]} ve ${at[1]}, uzaklık ${fmt(lo, 3)}: her konumun kodu benzersiz. İç çarpım P_p·P_(p+k) = Σ_i cos(k·ω_i) yalnız k'ya bağlıdır; sağdaki grafikte p = 0 ve p = ${q.p} için eğriler üst üste biner.`);
    return {
      plots: [
        { title: "Konum kodlaması (satır: konum p, sütun: boyut)", xlabel: "boyut", ylabel: "konum", size: { w: 310, h: 300 }, img: { w: q.d, h: q.n, data: flat.map((x) => (x + 1) / 2) }, side: true, series: [] },
        { title: `Boyut ${j0} ve ${j0 + 1}`, xlabel: "konum p", ylabel: "değer", ylim: [-1.05, 1.05], series: [
          { name: `boyut ${j0}: sin(pω)`, x: range(P.length), y: P.map((r) => r[j0]), c: 0 },
          { name: `boyut ${j0 + 1}: ${so ? "sin" : "cos"}(pω)`, x: range(P.length), y: P.map((r) => r[j0 + 1]), c: 2, dash: true }],
          marks: [{ x: q.p, label: `p = ${q.p}` }, { x: q.p + q.k, label: `p + k = ${q.p + q.k}` }] },
        { title: "İç çarpım P_p · P_(p+k), k'ya göre", xlabel: "k", ylabel: "iç çarpım", series: [
          { name: "p = 0", x: ks, y: s0, c: 3 },
          { name: `p = ${q.p}`, x: ks, y: s1, c: 1, dash: true }] },
      ],
      stats: [
        { label: `Dalga boyu (çift i = ${q.pair})`, value: lam },
        { label: `‖P_${q.p} − P_${q.p + q.k}‖`, value: dist(a, b) },
        { label: "En yakın iki farklı konumun uzaklığı", value: lo },
        { label: "En kısa / en uzun dalga boyu", value: `${fmt(2 * Math.PI, 4)} / ${fmt(2 * Math.PI * q.base ** ((q.d - 2) / q.d), 4)}` },
      ],
      note: P1.join(" "),
    };
  },
};
