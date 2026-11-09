// M4.6 — Çevrimiçi (online) softmax: skorları B'lik bloklar hâlinde gör, koşan maksimum m ve koşan payda l'yi güncelle,
// çıktıyı yeniden ölçekle. T×T skor matrisini hiç tutmadan tam softmax ağırlıklı toplamı (FlashAttention'ın çekirdek fikri).
import { fmt, range } from "./plot.js";

export function online(s, v, B) {
  let m = -Infinity, l = 0, acc = 0; const trace = [];
  for (let b = 0; b < s.length; b += B) {
    const blk = s.slice(b, b + B), mb = Math.max(...blk), mn = Math.max(m, mb), c = Math.exp(m - mn);
    l = l * c + blk.reduce((a, x) => a + Math.exp(x - mn), 0);
    acc = acc * c + blk.reduce((a, x, k) => a + Math.exp(x - mn) * v[b + k], 0);
    m = mn; trace.push({ b: b / B, m, l, out: acc / l });
  }
  return { out: acc / l, m, l, trace };
}
export const naive = (s, v) => { const e = s.map(Math.exp), z = e.reduce((a, b) => a + b, 0); return e.reduce((a, x, k) => a + (x / z) * v[k], 0); };
export const stable = (s, v) => { const m = Math.max(...s), e = s.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.reduce((a, x, k) => a + (x / z) * v[k], 0); };

export default {
  params: [
    { id: "T", label: "Anahtar sayısı T (dizi uzunluğu)", min: 16, max: 4096, value: 256, int: true, log: true },
    { id: "B", label: "Blok boyu B", min: 1, max: 512, value: 32, int: true, log: true },
    { id: "sc", label: "Skor ölçeği (logit büyüklüğü)", min: 0.1, max: 1000, value: 3, log: true, help: "Skorlar N(0, ölçek²). Büyük ölçekte exp(s) taşar: max çıkarmayan saf softmax Infinity/NaN verir." },
    { id: "tr", label: "Önemli anahtarın yeri (dizinin yüzdesi)", min: 0, max: 100, step: 1, value: 80 },
  ],
  seed: true,
  run(q, r) {
    const T = Math.round(q.T), B = Math.round(q.B), s = range(T).map(() => q.sc * r.normal()), v = range(T).map(() => r.normal());
    const k = Math.min(T - 1, Math.round((q.tr / 100) * (T - 1))); s[k] = Math.max(...s) + 2 * q.sc; v[k] = 3;
    const O = online(s, v, B), st = stable(s, v), nv = naive(s, v), nb = O.trace.length;
    return {
      plots: [
        { title: "Blok blok: koşan maksimum m ve koşan çıktı", xlabel: "işlenen blok", ylabel: "değer", series: [{ name: "koşan maksimum m", x: O.trace.map((t) => t.b + 1), y: O.trace.map((t) => t.m), c: 2, dots: nb <= 64 }, { name: "koşan çıktı (o ana kadarki blokların softmax ortalaması)", x: O.trace.map((t) => t.b + 1), y: O.trace.map((t) => t.out), c: 0, dots: nb <= 64 }], marks: [{ x: Math.floor(k / B) + 1, label: "baskın skor bu blokta" }, { y: st, label: "tam softmax sonucu" }] },
        { title: "Bellek: tam skor matrisi vs blok tamponu (T sorgu × T anahtar, log)", xlabel: "", ylabel: "saklanan skor", ylog: true, xlim: [-0.6, 1.6], ylim: [Math.max(1, T * B) / 10, T * T * 10], xticks: [[0, "T × T (tam matris)"], [1, "T × B (blok)"]], series: [{ name: "skor sayısı", x: [0, 1], y: [T * T, T * Math.min(B, T)], kind: "bar", c: 0 }] },
      ],
      stats: [
        { label: "Çevrimiçi sonuç", value: O.out },
        { label: "Kararlı tam softmax (max çıkararak)", value: st },
        { label: "|fark|", value: Math.abs(O.out - st) },
        { label: "Saf softmax (max çıkarmadan)", value: Number.isFinite(nv) ? nv : "taştı (exp → ∞, sonuç NaN/∞)" },
        { label: "Blok sayısı", value: nb },
      ],
      note: `Her blokta yeni maksimum m′ = max(m, blok maks.) bulunur; eski payda ve eski birikmiş toplam e^(m − m′) ile <b>yeniden ölçeklenip</b> yeni blokların katkısı eklenir. Sonda çıktı/payda tam softmax ağırlıklı toplamla <b>aynıdır</b> (fark yalnız yuvarlama: ${fmt(Math.abs(O.out - st), 2)}). Skorlar hiçbir zaman tamamı birden tutulmaz: GPU'da her blok hızlı SRAM'e sığar, T × T matris yavaş HBM belleğe hiç yazılmaz — FlashAttention'ın asıl kazancı bellek trafiğidir. ${Number.isFinite(nv) ? "" : "<b>Skor ölçeği büyük:</b> max çıkarmayan saf softmax'ta exp taşıyor; çevrimiçi sürüm her adımda max çıkardığı için kararlı."}`,
    };
  },
};
