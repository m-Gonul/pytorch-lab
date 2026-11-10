// M9.16 — Haar benzeri öznitelikler ve integral görüntü. 24 × 24 yapay "yüz" yamaları (koyu göz bandı, açık yanaklar, açık burun
// köprüsü, değişken konum/kontrast/aydınlatma) ve "yüz olmayan" yamalar (rastgele doku, kenarlar, düz bölgeler). Seçilen tek bir
// özniteliğin (tür, konum, boyut) iki sınıftaki dağılımı ve en iyi karar kütüğünün (eşik + kutup) hatası. Varyans normalizasyonu:
// Viola–Jones her pencereyi std'sine böler (integral görüntü + kare integral görüntüyle).
import { fmt, range } from "./plot.js";

export const S = 24;
export function facePatch(r, light) {
  const f = new Float64Array(S * S), dy = Math.round(2 * r.normal()), dx = Math.round(1.2 * r.normal()), c = 0.6 + 0.4 * r(), g = 1 + light * (2 * r() - 1), sl = light * (2 * r() - 1) * 0.02;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let v = 0.55;
    const yy = y - dy, xx = x - dx;
    if (yy >= 7 && yy < 11 && xx >= 3 && xx < 21) v = 0.55 - 0.3 * c;
    if (yy >= 11 && yy < 15 && xx >= 3 && xx < 21) v = 0.55 + 0.15 * c;
    if (yy >= 6 && yy < 13 && xx >= 10 && xx < 14) v = 0.55 + 0.2 * c;
    if (yy >= 17 && yy < 19 && xx >= 7 && xx < 17) v = 0.55 - 0.2 * c;
    f[y * S + x] = Math.max(0, Math.min(1, g * (v + sl * (x - 12)) + 0.04 * r.normal()));
  }
  return f;
}
export function nonFacePatch(r, light) {
  const f = new Float64Array(S * S), kind = Math.floor(r() * 4), g = 1 + light * (2 * r() - 1), a = r() * Math.PI, fr = 0.2 + r() * 0.8, base = 0.3 + 0.4 * r();
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let v = base;
    if (kind === 0) v = base + 0.25 * Math.sin(fr * (x * Math.cos(a) + y * Math.sin(a)));
    else if (kind === 1) v = x * Math.cos(a) + y * Math.sin(a) > 12 ? base + 0.25 : base - 0.1;
    else if (kind === 2) v = base + 0.15 * r.normal();
    f[y * S + x] = Math.max(0, Math.min(1, g * v + 0.04 * r.normal()));
  }
  return f;
}
export function integral(f, n = S) { const ii = new Float64Array((n + 1) * (n + 1)); for (let y = 0; y < n; y++) { let row = 0; for (let x = 0; x < n; x++) { row += f[y * n + x]; ii[(y + 1) * (n + 1) + x + 1] = ii[y * (n + 1) + x + 1] + row; } } return ii; }
export const rect = (ii, x, y, w, h, n = S) => ii[(y + h) * (n + 1) + x + w] - ii[y * (n + 1) + x + w] - ii[(y + h) * (n + 1) + x] + ii[y * (n + 1) + x];
// Öznitelik türleri: 0 yatay iki dikdörtgen (sol − sağ), 1 dikey iki (üst − alt), 2 yatay üç (orta − kenarlar), 3 dikey üç, 4 dört (çapraz).
export function feature(ii, t, x, y, w, h) {
  if (t === 0) return rect(ii, x, y, w, h) - rect(ii, x + w, y, w, h);
  if (t === 1) return rect(ii, x, y, w, h) - rect(ii, x, y + h, w, h);
  if (t === 2) return rect(ii, x + w, y, w, h) - rect(ii, x, y, w, h) - rect(ii, x + 2 * w, y, w, h);
  if (t === 3) return rect(ii, x, y + h, w, h) - rect(ii, x, y, w, h) - rect(ii, x, y + 2 * h, w, h);
  return rect(ii, x, y, w, h) + rect(ii, x + w, y + h, w, h) - rect(ii, x + w, y, w, h) - rect(ii, x, y + h, w, h);
}
export const span = (t) => [[2, 1], [1, 2], [3, 1], [1, 3], [2, 2]][t];
export function dataset(r, n, light) { const X = [], y = []; for (let i = 0; i < n; i++) { X.push(facePatch(r, light)); y.push(1); X.push(nonFacePatch(r, light)); y.push(0); } return { X, y }; }
export function normed(f) { const m = f.reduce((a, b) => a + b, 0) / f.length, s = Math.sqrt(f.reduce((a, b) => a + (b - m) ** 2, 0) / f.length) || 1; return f.map((v) => (v - m) / s); }
export function stump(vals, y, w) {
  const idx = range(vals.length).sort((a, b) => vals[a] - vals[b]); let Tp = 0, Tn = 0; y.forEach((l, i) => (l ? (Tp += w[i]) : (Tn += w[i])));
  let Sp = 0, Sn = 0, best = { err: Infinity };
  for (let k = 0; k < idx.length; k++) {
    const e1 = Sp + (Tn - Sn), e2 = Sn + (Tp - Sp); // e1: eşiğin altı negatif (kutup +1), e2: altı pozitif (kutup −1)
    const th = k === 0 ? vals[idx[0]] - 1e-9 : (vals[idx[k - 1]] + vals[idx[k]]) / 2;
    if (e1 < best.err) best = { err: e1, th, pol: 1 }; if (e2 < best.err) best = { err: e2, th, pol: -1 };
    const i = idx[k]; if (y[i]) Sp += w[i]; else Sn += w[i];
  }
  return best;
}
const TYPES = ["iki dikdörtgen, yatay (sol − sağ)", "iki dikdörtgen, dikey (üst − alt)", "üç dikdörtgen, yatay (orta − kenarlar)", "üç dikdörtgen, dikey (orta − üst/alt)", "dört dikdörtgen (çapraz)"];
export default {
  params: [
    { id: "t", label: "Öznitelik türü", options: TYPES, value: 1 },
    { id: "x", label: "x", min: 0, max: 20, value: 3, int: true },
    { id: "y", label: "y", min: 0, max: 20, value: 7, int: true },
    { id: "w", label: "Dikdörtgen genişliği", min: 1, max: 18, value: 18, int: true },
    { id: "h", label: "Dikdörtgen yüksekliği", min: 1, max: 12, value: 4, int: true },
    { id: "norm", label: "Pencere normalizasyonu", options: ["yok", "varyansa böl (Viola–Jones)"], value: 1 },
    { id: "light", label: "Aydınlatma değişkenliği", min: 0, max: 0.8, step: 0.05, value: 0.4 },
  ],
  seed: true,
  validate(q) { const [sx, sy] = span(q.t); return q.x + sx * q.w > S || q.y + sy * q.h > S ? [{ level: "error", msg: `Öznitelik pencereden taşıyor: x + ${sx}·w ≤ 24 ve y + ${sy}·h ≤ 24 olmalı.` }] : []; },
  run(q, r) {
    const { X, y } = dataset(r, 200, q.light), vals = X.map((f) => feature(integral(q.norm ? normed(f) : f), q.t, q.x, q.y, q.w, q.h));
    const st = stump(vals, y, y.map(() => 1 / y.length));
    const valsO = X.map((f) => feature(integral(q.norm ? f : normed(f)), q.t, q.x, q.y, q.w, q.h)), stO = stump(valsO, y, y.map(() => 1 / y.length));
    const fv = vals.filter((_, i) => y[i]), nv = vals.filter((_, i) => !y[i]), lo = Math.min(...vals), hi = Math.max(...vals), nb = 30, bw = (hi - lo) / nb || 1;
    const hist = (a) => range(nb).map((b) => a.filter((v) => Math.min(nb - 1, Math.floor((v - lo) / bw)) === b).length);
    const ex = X[0], ov = Array.from(ex), [sx, sy] = span(q.t);
    for (let yy = q.y; yy < q.y + sy * q.h; yy++) for (let xx = q.x; xx < q.x + sx * q.w; xx++) { const bx = Math.floor((xx - q.x) / q.w), by = Math.floor((yy - q.y) / q.h), white = q.t === 4 ? (bx + by) % 2 === 0 : q.t === 0 || q.t === 2 ? (q.t === 2 ? bx === 1 : bx === 0) : (q.t === 3 ? by === 1 : by === 0); ov[yy * S + xx] = white ? 0.5 + 0.5 * ov[yy * S + xx] : 0.5 * ov[yy * S + xx]; }
    const nrect = q.t <= 1 ? 2 : q.t <= 3 ? 3 : 4, refs = q.t <= 1 ? 6 : q.t <= 3 ? 8 : 9, naive = sx * q.w * sy * q.h;
    const P = [`En iyi karar kütüğünün hatası ${fmt(100 * st.err, 3)}% (eşik ${fmt(st.th, 3)}, kutup ${st.pol > 0 ? "+" : "−"}). Rastgele tahmin %50.`];
    P.push(`Hesap: ${nrect} dikdörtgen integral görüntüde ${refs} dizi erişimi (komşu dikdörtgenler köşe paylaşır); doğrudan toplamak ${naive} piksel okur. Öznitelik maliyeti boyutundan bağımsızdır.`);
    P.push(`${q.norm ? "Her pencere ortalaması çıkarılıp std'sine bölündü: aydınlatma kazancı özniteliği etkilemez." : "Normalizasyon yok: aydınlatma kazancı öznitelik değerini ölçekler."} Aynı öznitelik ${q.norm ? "normalizasyonsuz" : "varyans normalizasyonuyla"} ${fmt(100 * stO.err, 3)}% hata veriyor.`);
    return {
      plots: [
        { title: "Örnek yüz ve öznitelik (açık: +, koyu: −)", img: { w: S, h: S, data: ov }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Öznitelik değerinin dağılımı: yüz ve yüz olmayan", xlabel: "öznitelik", ylabel: "sayı", side: true, size: { w: 300, h: 290 }, series: [{ name: "yüz", x: range(nb).map((b) => lo + (b + 0.5) * bw), y: hist(fv), kind: "bar", c: 0 }, { name: "yüz olmayan", x: range(nb).map((b) => lo + (b + 0.5) * bw), y: hist(nv), kind: "bar", c: 1 }], marks: [{ x: st.th, label: "eşik" }] },
      ],
      stats: [{ label: "Karar kütüğü hatası", value: `${fmt(100 * st.err, 3)}%` }, { label: "Dizi erişimi / doğrudan piksel", value: `${refs} / ${naive}` }],
      note: P.join(" "),
    };
  },
};
