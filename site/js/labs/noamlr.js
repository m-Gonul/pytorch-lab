// M4.14 — Öğrenme oranı takvimleri: Noam (Vaswani ve ark. 2017) lr = f·d^−0.5·min(s^−0.5, s·w^−1.5); doğrusal ısınma + kosinüs
// (GPT ailesi); doğrusal ısınma + ters karekök (Noam'ın tepe değeri serbest bırakılmış hâli); sabit. Tepe, ısınma sonundaki değer,
// s adımdaki değer ve toplam "öğrenme oranı alanı" (Σ lr) karşılaştırılır.
import { fmt, range } from "./plot.js";

export const noam = (s, d, w, f = 1) => f * d ** -0.5 * Math.min(s ** -0.5, s * w ** -1.5);
export const cosine = (s, peak, w, S, floor) => (s < w ? (peak * s) / w : floor * peak + 0.5 * (1 - floor) * peak * (1 + Math.cos((Math.PI * Math.min(1, (s - w) / Math.max(1, S - w))))));

export default {
  params: [
    { id: "d", label: "Model boyutu d", min: 64, max: 4096, value: 512, int: true, log: true },
    { id: "w", label: "Isınma adımı w", min: 1, max: 20000, value: 4000, int: true, log: true },
    { id: "f", label: "Noam çarpanı (makalede 1)", min: 0.1, max: 10, value: 1, log: true },
    { id: "S", label: "Toplam adım", min: 1000, max: 500000, value: 100000, int: true, log: true },
    { id: "floor", label: "Kosinüs taban oranı (tepe değerin)", min: 0, max: 0.5, step: 0.01, value: 0.1 },
  ],
  validate(q) { return q.w >= q.S ? [{ level: "warn", msg: "Isınma toplam adımdan uzun: takvim hiç azalma evresine geçmiyor." }] : []; },
  run(q) {
    const N = 300, xs = range(N).map((i) => Math.max(1, Math.round((q.S * (i + 1)) / N))), peak = noam(q.w, q.d, q.w, q.f);
    const yN = xs.map((s) => noam(s, q.d, q.w, q.f)), yC = xs.map((s) => cosine(s, peak, q.w, q.S, q.floor)), area = (fn) => { let a = 0; for (let s = 1; s <= q.S; s += Math.max(1, Math.floor(q.S / 5000))) a += fn(s) * Math.max(1, Math.floor(q.S / 5000)); return a; };
    const aN = area((s) => noam(s, q.d, q.w, q.f)), aC = area((s) => cosine(s, peak, q.w, q.S, q.floor));
    const P = [`Noam tepesi s = w = ${q.w} adımda: ${fmt(q.f, 3)} · ${q.d}^−0.5 · ${q.w}^−0.5 = ${fmt(peak, 4)}. Tepe hem d'ye hem w'ye 1/√ ile bağlıdır: ısınmayı uzatmak tepeyi de düşürür (w'yi 4 katına çıkarmak tepeyi yarıya indirir). Isınmadan sonra lr ∝ 1/√s: ${q.S} adımda ${fmt(noam(q.S, q.d, q.w, q.f), 4)} (tepenin ${fmt(Math.sqrt(q.w / q.S), 3)} katı).`];
    P.push(`Aynı tepe ve ısınmayla kosinüs takvimi ${q.S} adımda tepenin ${q.floor} katına iner (${fmt(q.floor * peak, 4)}). Toplam Σ lr: Noam ${fmt(aN, 4)}, kosinüs ${fmt(aC, 4)}. Noam'ın kuyruğu sonsuza dek yavaşça azalır (eğitimi uzatmak kolay), kosinüs ise toplam adım sayısını baştan bilmeyi gerektirir.`);
    P.push("Neden ısınma? Adam'ın ikinci moment kestirimi ilk adımlarda birkaç gradyana dayanır ve güvenilmezdir. Post-LN Transformer'da çıkışa yakın katmanların gradyanı başlangıçta büyüktür (Xiong ve ark. 2020). Isınmasız büyük bir lr ilk adımlarda ağı kararsız bir bölgeye itebilir (M4.7, lnplace deneyi).");
    return {
      plots: [{ title: "Öğrenme oranı takvimleri", xlabel: "adım", ylabel: "öğrenme oranı", yzero: true, series: [
        { name: "Noam", x: xs, y: yN, c: 0 },
        { name: `ısınma + kosinüs (taban %${Math.round(q.floor * 100)})`, x: xs, y: yC, c: 2, dash: true },
        { name: "sabit (Noam tepesi)", x: [xs[0], xs.at(-1)], y: [peak, peak], c: 4, dash: true }],
        marks: [{ x: q.w, label: `w = ${q.w}` }] }],
      stats: [
        { label: "Noam tepe lr", value: peak },
        { label: `Noam lr, adım ${q.S}`, value: noam(q.S, q.d, q.w, q.f) },
        { label: `Kosinüs lr, adım ${q.S}`, value: cosine(q.S, peak, q.w, q.S, q.floor) },
        { label: "Σ lr: Noam / kosinüs", value: `${fmt(aN, 4)} / ${fmt(aC, 4)}` },
      ],
      note: P.join(" "),
    };
  },
};
