// M1.3 — Ridge'in SVD bakışı: her tekil yön kendi çarpanıyla küçülür (s²/(s²+nλ)),
// df(λ) = Σ s²/(s²+nλ) etkili serbestlik derecesi. c aşamasındaki yol formülünün görsel karşılığı.
import { fmt, linspace } from "./plot.js";

const J = 6; // yön sayısı: s_j, s_büyük'ten s_küçük'e geometrik

export function factors(s, n, lam) { return s.map((v) => (v * v) / (v * v + n * lam)); }
export const dfOf = (f) => f.reduce((a, v) => a + v, 0);
export function spectrum(sb, ss, j) { return Array.from({ length: J }, (_, i) => sb * (ss / sb) ** (i / (J - 1))); }
export function condOf(s, n, lam) { return (s[0] * s[0] + n * lam) / (s[s.length - 1] * s[s.length - 1] + n * lam); }

export default {
  params: [
    { id: "sb", label: "En büyük tekil değer", min: 1, max: 10, step: 0.5, value: 5 },
    { id: "ss", label: "En küçük tekil değer", min: 0.05, max: 2, step: 0.05, value: 0.3 },
    { id: "n", label: "Örnek sayısı n", min: 5, max: 200, step: 5, value: 40, int: true },
    { id: "lam", label: "Ceza λ", min: 0.001, max: 100, value: 0.5, log: true, hardMin: 0 },
  ],
  validate(q) { return q.ss > q.sb ? [{ level: "warn", msg: `En küçük tekil değer (${q.ss}) en büyüğünden (${q.sb}) büyük: ss = sb alındı.` }] : []; },
  run(q) {
    const sb = Math.max(q.sb, q.ss), ss = Math.min(q.sb, q.ss), s = spectrum(sb, ss);
    const f = factors(s, q.n, q.lam), df = dfOf(f);
    const lams = linspace(-3, 2, 61).map((e) => 10 ** e);
    const dfc = lams.map((l) => dfOf(factors(s, q.n, l)));
    return {
      plots: [
        {
          title: "Yön-yön küçültme çarpanı sⱼ²/(sⱼ²+nλ): 1 = dokunulmadı, 0 = tam bastırıldı",
          xlabel: "tekil yön j (büyük → küçük s)", ylabel: "çarpan", ylim: [0, 1.05], yzero: true,
          series: [
            { name: "ridge çarpanı", x: s.map((_, i) => i + 1), y: f, kind: "bar", c: 0 },
            { name: "OLS çarpanı (λ = 0)", x: s.map((_, i) => i + 1), y: s.map(() => 1), kind: "bar", c: 4, off: 0.38, bw: 0.4 },
          ],
        },
        {
          title: `Etkili serbestlik derecesi df(λ) = Σ sⱼ²/(sⱼ²+nλ) — ${J} yön`,
          xlabel: "log₁₀ λ", ylabel: "df(λ)", ylim: [0, J + 0.4],
          series: [{ name: "df(λ)", x: lams.map(Math.log10), y: dfc, c: 1 }],
          marks: [{ x: Math.log10(q.lam), label: "seçili λ" }],
        },
      ],
      stats: [
        { label: "Seçili λ'da df(λ) — etkili serbestlik", value: df },
        { label: "Tam bastırılan yön (çarpan < 0.1)", value: f.filter((v) => v < 0.1).length },
        { label: "Neredeyse dokunulmayan yön (çarpan > 0.9)", value: f.filter((v) => v > 0.9).length },
        { label: "Koşul sayısı: OLS → ridge", value: `${fmt(sb ** 2 / ss ** 2)} → ${fmt(condOf(s, q.n, q.lam))}` },
        { label: "λ → ∞'da df (limit)", value: 0 },
      ],
      note: `Ridge her yönü kendi çarpanıyla küçültür: ${fmt(sb, 1)} gibi büyük tekil değerli (verinin çok değiştiği, bilgi yüklü) yönler ${fmt(f[0], 2)} çarpanıyla neredeyse korunur; ${fmt(ss, 2)} gibi küçük tekil değerli (gürültüye açık) yönler ${fmt(f[J - 1], 3)} çarpanıyla fiilen sıfırlanır. <b>df(λ) = ${fmt(df, 2)}</b>: model, ${J} parametre yerine bu kadar "etkili" parametre kullanıyor. Koşul sayısı da ${fmt(sb ** 2 / ss ** 2)}'den ${fmt(condOf(s, q.n, q.lam))}'ye iner — çözüm kararlılaşır. λ'yı artırdıkça df'in düştüğünü izle: λ = 0'da df = ${J}, λ → ∞'da 0.`,
    };
  },
};
