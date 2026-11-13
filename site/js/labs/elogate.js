// M11.11 — Elo ve Bradley–Terry. Gerçek güç farkı d Elo puanı ise güçlü oyuncunun beklenen skoru E = 1/(1 + 10^(−d/400)). n oyunluk
// bir maç simüle edilir (beraberlik olasılığı b; kalan olasılık kazanç/kayıp arasında E'yi koruyacak biçimde bölünür). Skordan Elo
// kestirimi d̂ = −400 log₁₀(1/s − 1) ve %95 aralığı (skorun standart hatasından delta yöntemiyle). AlphaGo Zero'nun "kapısı":
// yeni ağ, en iyi ağa karşı 400 oyunda %55'ten fazla skor alırsa kabul edilir. Kabul olasılığı gerçek farka göre (binom, R tekrar).
import { fmt, range } from "./plot.js";

const E = (d) => 1 / (1 + 10 ** (-d / 400));
const toElo = (s) => -400 * Math.log10(1 / s - 1);
function match(r, d, n, b) { const e = E(d), w = e - b / 2; let s = 0; for (let i = 0; i < n; i++) { const u = r(); s += u < w ? 1 : u < w + b ? 0.5 : 0; } return s / n; }
export default {
  params: [
    { id: "d", label: "Gerçek güç farkı d (Elo)", min: -100, max: 400, step: 5, value: 35 },
    { id: "n", label: "Maçtaki oyun sayısı n", min: 10, max: 1000, step: 10, value: 400, int: true },
    { id: "b", label: "Beraberlik olasılığı", min: 0, max: 0.6, step: 0.05, value: 0 },
    { id: "gate", label: "Kabul eşiği (skor)", min: 0.5, max: 0.7, step: 0.01, value: 0.55 },
  ],
  validate(q) { const e = E(q.d); return q.b / 2 > Math.min(e, 1 - e) ? [{ level: "error", msg: "Beraberlik olasılığı: bu güç farkında kazanç/kayıp olasılıkları negatif olur; beraberliği azalt." }] : []; },
  seed: true,
  run(q, r) {
    const R = 400, e = E(q.d), scores = range(R).map(() => match(r, q.d, q.n, q.b)), acc = scores.filter((s) => s > q.gate).length / R;
    const s = scores[0], se = Math.sqrt((e * (1 - e) - q.b / 4) / q.n), slope = 400 / (Math.LN10 * s * (1 - s));
    const est = Math.abs(s - 0.5) < 0.4999 ? toElo(s) : NaN, ci = 1.96 * se * slope;
    const ds = range(41).map((i) => -50 + i * 5), accCurve = ds.map((d) => { const ee = E(d), sd = Math.sqrt((ee * (1 - ee) - q.b / 4 * (q.b <= 2 * Math.min(ee, 1 - ee) ? 1 : 0)) / q.n); const z = (q.gate - ee) / Math.max(sd, 1e-9); return 0.5 * (1 - erf(z / Math.SQRT2)); });
    const d50 = toElo(q.gate);
    const P = [`Gerçek fark ${q.d} Elo: beklenen skor ${fmt(e, 4)}. Bir ${q.n} oyunluk maçta ölçülen skor ${fmt(s, 3)} → Elo kestirimi ${fmt(est, 3)} ± ${fmt(ci, 3)} (%95).`];
    P.push(`Kapı (skor > ${q.gate}): ${R} tekrarda kabul oranı ${fmt(100 * acc, 3)}%. Kabulün yarı yarıya olduğu gerçek fark ≈ ${fmt(d50, 3)} Elo; ${q.n} oyunda skorun standart hatası ≈ ${fmt(se, 2)}.`);
    if (q.d <= 0) P.push("Gerçek fark sıfır ya da negatifken kabul, gürültünün eseridir: eşik bu yanlış kabulleri sınırlamak için 0.5'in üstünde tutulur.");
    if (q.b > 0) P.push("Beraberlikler skorun varyansını azaltır: aynı n ile aralık daralır.");
    return {
      plots: [
        { title: "Elo farkı ve beklenen skor", xlabel: "d (Elo)", ylabel: "beklenen skor", side: true, size: { w: 300, h: 230 }, ylim: [0, 1], series: [{ name: "E(d)", x: range(81).map((i) => -400 + i * 10), y: range(81).map((i) => E(-400 + i * 10)), c: 0 }], marks: [{ x: q.d, label: "d" }] },
        { title: `Kabul olasılığı (${q.n} oyun, eşik ${q.gate})`, xlabel: "gerçek fark d (Elo)", ylabel: "P(kabul)", side: true, size: { w: 300, h: 230 }, ylim: [0, 1], series: [{ name: "normal yaklaşım", x: ds, y: accCurve, c: 2 }, { name: "simülasyon (seçili d)", x: [q.d], y: [acc], kind: "scatter", c: 0 }], marks: [{ x: d50, label: "%50" }] },
      ],
      stats: [{ label: "Beklenen skor", value: fmt(e, 4) }, { label: "Elo kestirimi (tek maç)", value: `${fmt(est, 3)} ± ${fmt(ci, 3)}` }, { label: "Kabul oranı", value: `${fmt(100 * acc, 3)}%` }],
      note: P.join(" "),
    };
  },
};
function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
