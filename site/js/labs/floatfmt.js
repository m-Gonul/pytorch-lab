// M7.3 — Kayan nokta biçimleri: e bit üs, m bit mantis, bias = 2^(e−1) − 1. Yuvarlama: üs E = ⌊log₂|x|⌋ (alt sınır 1 − bias:
// alt normal sayılar), adım q = 2^(E−m), en yakına (eşitlikte çift). En büyük sonlu değer (2 − 2^−m)·2^bias; fp8 e4m3fn'de 448
// (en üst üs kodu da sayılar için kullanılır, sonsuz yok). Ayrıca: w + Δ işleminin bu biçimde kaybolup kaybolmadığı.
import { fmt, range } from "./plot.js";

export const FMTS = [
  { n: "fp32 (e8, m23)", e: 8, m: 23 }, { n: "tf32 (e8, m10)", e: 8, m: 10 }, { n: "bf16 (e8, m7)", e: 8, m: 7 },
  { n: "fp16 (e5, m10)", e: 5, m: 10 }, { n: "fp8 e5m2", e: 5, m: 2 }, { n: "fp8 e4m3fn", e: 4, m: 3, max: 448 },
];
const rhe = (v) => { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1; };
export function info(F) { const bias = 2 ** (F.e - 1) - 1, emin = 1 - bias, max = F.max ?? (2 - 2 ** -F.m) * 2 ** bias; return { bias, emin, max, minNorm: 2 ** emin, minSub: 2 ** (emin - F.m), eps: 2 ** -F.m }; }
export function roundTo(x, F) {
  if (x === 0 || !isFinite(x)) return x;
  const I = info(F), s = Math.sign(x), a = Math.abs(x), E = Math.max(Math.floor(Math.log2(a)), I.emin), q = 2 ** (E - F.m), r = rhe(a / q) * q;
  return r > I.max ? s * Infinity : s * r;
}
export const ulp = (x, F) => { const I = info(F); return 2 ** (Math.max(Math.floor(Math.log2(Math.abs(x) || I.minNorm)), I.emin) - F.m); };
export default {
  params: [
    { id: "f", label: "Biçim", options: FMTS.map((f) => f.n), value: 3 },
    { id: "lx", label: "Değer x = 10^a, a", min: -12, max: 40, step: 0.01, value: 0.4771 },
    { id: "neg", label: "İşaret", options: ["+", "−"], value: 0 },
    { id: "ld", label: "Güncelleme Δ = 10^b (w = x'e eklenir), b", min: -12, max: 2, step: 0.1, value: -4 },
  ],
  run(q) {
    const F = FMTS[q.f], I = info(F), x = (q.neg ? -1 : 1) * 10 ** q.lx, r = roundTo(x, F), rel = isFinite(r) ? Math.abs(r - x) / Math.abs(x) : Infinity;
    const d = 10 ** q.ld, w = roundTo(x, F), wn = roundTo(w + d, F), lost = wn === w;
    const xs = range(161).map((i) => -12 + i * 0.3), spacing = FMTS.map((G) => xs.map((lx) => { const v = 10 ** lx; return v > info(G).max ? null : ulp(v, G) / v; }));
    const P = [`${F.n}: en büyük sonlu ${fmt(I.max, 6)}, en küçük normal ${fmt(I.minNorm, 3)}, en küçük alt normal ${fmt(I.minSub, 3)}, 1'in üstündeki ilk sayıya uzaklık (eps) 2^−${F.m} = ${fmt(I.eps, 3)}.`];
    P.push(isFinite(r) ? `x = ${fmt(x, 6)} → ${fmt(r, 6)} (göreli hata ${fmt(rel, 2)}; bu büyüklükte iki komşu sayı arası ${fmt(ulp(x, F), 3)}).` : `x = ${fmt(x, 4)} en büyük sonlu değeri aşıyor: <b>taşma</b>${F.max ? " (e4m3fn'de sonsuz yok; dönüşüm ya NaN verir ya doyurulur, kütüphaneye bağlı)" : " → ±∞"}.`);
    if (Math.abs(x) < I.minNorm && x !== 0 && isFinite(r)) P.push(r === 0 ? "x en küçük alt normalin yarısından küçük: <b>sıfıra yuvarlandı</b> (alt taşma). Gradyanlarda bu, bilginin kaybolmasıdır; loss scaling bunu önler." : "x alt normal bölgede: göreli hassasiyet düşüyor, çünkü mantisin baştaki bitleri sıfır.");
    if (isFinite(w)) P.push(lost ? `w = ${fmt(w, 5)} iken w + ${fmt(d, 2)} yine ${fmt(wn, 5)}: güncelleme komşu sayıya olan uzaklığın yarısından (${fmt(ulp(w, F) / 2, 3)}) küçük, <b>kayboldu</b>. Düşük hassasiyetli ağırlıklarla eğitimin takılmasının nedeni budur; fp32 ana kopya ya da stokastik yuvarlama çözümdür.` : `w + Δ = ${fmt(wn, 5)}: güncelleme temsil edilebildi (göreli hata ${fmt(Math.abs(wn - (w + d)) / d, 2)}).`);
    return {
      plots: [{ title: "Göreli aralık (ulp / |x|) ve büyüklük", xlabel: "log₁₀ |x|", ylabel: "ulp / |x|", ylog: true, wide: true, size: { w: 620, h: 250 }, series: FMTS.map((G, i) => ({ name: G.n, x: xs, y: spacing[i], c: i % 5, dash: i >= 5 })), marks: [{ x: q.lx, label: "x" }] }],
      stats: [
        { label: "Yuvarlanmış x", value: r },
        { label: "Göreli hata", value: rel },
        { label: "En büyük sonlu / eps", value: `${fmt(I.max, 6)} / ${fmt(I.eps, 3)}` },
        { label: "w + Δ kayboldu mu?", value: isFinite(w) ? (lost ? "evet" : "hayır") : "—" },
      ],
      note: P.join(" "),
    };
  },
};
