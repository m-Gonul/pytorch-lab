// M9.8 — Gri seviye morfoloji, bir görüntü satırında (1-B). Profil: yavaş değişen arka plan + genişliği 3, 7, 15 olan parlak
// tepeler + genişliği 3, 9 olan koyu çukurlar + Gauss gürültüsü. Düz yapı elemanı (uzunluk L) ya da düz olmayan parabol
// kubbe b_N(s) = h·(1 − (2s/L)²) (merkezde h). Erozyon min{f(x+s) − b_N(s)}, genişleme max{f(x−s) + b_N(s)} (simetrik b için b̂ = b).
import { fmt, range } from "./plot.js";

const N = 220;
const PEAKS = [{ c: 40, w: 3 }, { c: 80, w: 7 }, { c: 125, w: 15 }], DIPS = [{ c: 165, w: 3 }, { c: 195, w: 9 }];
export function profile(r, sn) {
  const bg = range(N).map((i) => 0.3 + 0.15 * Math.sin((2 * Math.PI * i) / N) + 0.001 * i);
  const f = bg.slice();
  for (const p of PEAKS) for (let i = p.c - (p.w >> 1); i <= p.c + (p.w >> 1); i++) f[i] += 0.35;
  for (const d of DIPS) for (let i = d.c - (d.w >> 1); i <= d.c + (d.w >> 1); i++) f[i] -= 0.2;
  return { bg, f: f.map((v) => v + sn * r.normal()) };
}
export function morph(f, L, nonflat, h, dil) {
  const H = (L - 1) / 2, b = range(L).map((j) => (nonflat ? h * (1 - ((2 * (j - H)) / L) ** 2) : 0));
  return f.map((_, x) => {
    let v = dil ? -Infinity : Infinity;
    for (let j = 0; j < L; j++) { const s = j - H, y = dil ? x - s : x + s; if (y < 0 || y >= N) continue; v = dil ? Math.max(v, f[y] + b[j]) : Math.min(v, f[y] - b[j]); }
    return v;
  });
}
const OPS = ["erozyon f ⊖ b", "genişleme f ⊕ b", "açma f ∘ b", "kapama f • b", "morfolojik gradyan (f ⊕ b) − (f ⊖ b)", "top-hat f − (f ∘ b)", "bottom-hat (f • b) − f", "açma, sonra kapama"];
export default {
  params: [
    { id: "op", label: "İşlem", options: OPS, value: 2 },
    { id: "L", label: "Yapı elemanı uzunluğu L (tek)", min: 3, max: 41, step: 2, value: 9, int: true },
    { id: "nf", label: "Yapı elemanı", options: ["düz", "düz olmayan (parabol)"], value: 0 },
    { id: "h", label: "Parabol yüksekliği h", min: 0.02, max: 0.5, step: 0.01, value: 0.1, on: { nf: [1] } },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.05, step: 0.005, value: 0.01 },
  ],
  seed: true,
  run(q, r) {
    const { bg, f } = profile(r, q.noise), E = (a) => morph(a, q.L, q.nf === 1, q.h, false), D = (a) => morph(a, q.L, q.nf === 1, q.h, true);
    const op = (a) => D(E(a)), cl = (a) => E(D(a));
    const out = [E(f), D(f), op(f), cl(f), D(f).map((v, i) => v - E(f)[i]), f.map((v, i) => v - op(f)[i]), cl(f).map((v, i) => v - f[i]), cl(op(f))][q.op];
    const o = op(f), c = cl(f);
    const peak = (p) => { const s = p.c - (p.w >> 1); let m = -Infinity; for (let i = s; i < s + p.w; i++) m = Math.max(m, o[i] - bg[i]); return m / 0.35; };
    const dip = (d) => { const s = d.c - (d.w >> 1); let m = -Infinity; for (let i = s; i < s + d.w; i++) m = Math.max(m, bg[i] - c[i]); return m / 0.2; };
    const P = [`Açmadan sonra parlak tepelerin kalan yüksekliği (genişlik 3, 7, 15): ${PEAKS.map((p) => fmt(Math.max(0, peak(p)), 2)).join(", ")}; kapamadan sonra koyu çukurların kalan derinliği (genişlik 3, 9): ${DIPS.map((d) => fmt(Math.max(0, dip(d)), 2)).join(", ")} (1 = korundu, ≈ 0 = silindi).`];
    P.push([
      "Erozyon, pencere içindeki en küçük değer: parlak ayrıntılar daralır ya da kaybolur, koyu çukurlar genişler.",
      "Genişleme, en büyük değer: parlak tepeler genişler, L'den dar koyu çukurlar dolar.",
      `Açma: yapı elemanı eğrinin altından yukarı itilir. L = ${q.L}'den dar parlak tepeler kırpılır, geniş tepeler ve genel düzey korunur.`,
      `Kapama: yapı elemanı eğrinin üstünden aşağı bastırılır. L = ${q.L}'den dar koyu çukurlar dolar.`,
      "Morfolojik gradyan bölgeler arası geçişleri vurgular; düz bölgelerde (L, çözünürlüğe göre çok büyük değilse) sıfıra yakındır.",
      "Top-hat = f − açma: açmanın kırptığı dar parlak ayrıntılar kalır, yavaş değişen arka plan çıkarılır (düzensiz aydınlatma düzeltmesi).",
      "Bottom-hat = kapama − f: dar koyu ayrıntılar kalır.",
      "Açma, sonra kapama: önce dar parlak, sonra dar koyu ayrıntılar silinir; morfolojik düzleştirme.",
    ][q.op]);
    if (q.nf === 1) { const fm = Math.min(...f), fM = Math.max(...f), om = Math.min(...out), oM = Math.max(...out); P.push(`Düz olmayan kubbe (h = ${q.h}): erozyon f'den b_N çıkarır, genişleme ekler; tek başına erozyon ve genişlemenin sonucu f'nin değer aralığıyla sınırlı değildir: f ∈ [${fmt(fm, 3)}, ${fmt(fM, 3)}], çıkış ∈ [${fmt(om, 3)}, ${fmt(oM, 3)}]${om < fm - 1e-9 || oM > fM + 1e-9 ? " (aralık dışına taştı)" : " (burada aralıkta kaldı: açma ve kapamada çıkarılan ve eklenen kubbe birbirini büyük ölçüde götürür)"}.`); }
    if (q.op === 5) { const err = Math.sqrt(range(N).reduce((s, i) => s + (o[i] - bg[i]) ** 2, 0) / N); P.push(`Arka plan kestirimi olarak açmanın RMS hatası ${fmt(err, 3)}; L en geniş parlak tepeden (15) büyük seçilmeli ama arka planın eğriliğini izleyecek kadar küçük kalmalı.`); }
    const x = range(N);
    return {
      plots: [{ title: `${OPS[q.op]}, L = ${q.L}`, xlabel: "x (piksel)", ylabel: "yoğunluk", wide: true, size: { w: 620, h: 230 }, series: [{ name: "f", x, y: f, c: 4 }, { name: "çıkış", x, y: out, c: 0 }, ...(q.op === 5 ? [{ name: "f ∘ b (arka plan kestirimi)", x, y: o, c: 1, dash: true }] : []), { name: "gerçek arka plan", x, y: bg, c: 2, dash: true }] }],
      stats: [{ label: "Tepe kalanı (w = 3, 7, 15)", value: PEAKS.map((p) => fmt(Math.max(0, peak(p)), 2)).join(", ") }, { label: "Çukur kalanı (w = 3, 9)", value: DIPS.map((d) => fmt(Math.max(0, dip(d)), 2)).join(", ") }, { label: "Çıkış min / max", value: `${fmt(Math.min(...out), 3)} / ${fmt(Math.max(...out), 3)}` }],
      note: P.join(" "),
    };
  },
};
