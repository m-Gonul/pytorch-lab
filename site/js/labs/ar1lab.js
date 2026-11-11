// M11.1 — AR(1) süreci x_t = a·x_{t−1} + w_t, w_t ~ N(0, σ²) bağımsız. Topluluk (ensemble) görünümü: N gerçekleşme aynı anda simüle
// edilir; sabit t'de x_t bir rastgele değişkendir ve topluluk ortalaması/varyansı kuramsal değerlerle karşılaştırılır:
// E[x_t] = aᵗx₀, Var[x_t] = σ²(1 − a^{2t})/(1 − a²) (a = 1'de σ²t: rastgele yürüyüş, Wiener sürecinin ayrık karşılığı).
// |a| < 1 ve x₀ durağan dağılımdan N(0, σ²/(1 − a²)) çekilirse süreç baştan geniş anlamda durağandır; otokovaryans C(τ) = σ²a^|τ|/(1 − a²).
// Zaman ortalaması (tek yolun 0..T ortalaması) ile topluluk ortalamasının karşılaştırılması ergodikliği gösterir.
import { fmt, range } from "./plot.js";

const STARTS = ["x₀ = 0", "x₀ = 5", "x₀ durağan dağılımdan"];
export default {
  params: [
    { id: "a", label: "Katsayı a", min: -0.99, max: 1, step: 0.01, value: 0.8 },
    { id: "st", label: "Başlangıç", options: STARTS, value: 0 },
    { id: "T", label: "Adım sayısı", min: 20, max: 400, step: 10, value: 100, int: true },
    { id: "logN", label: "Gerçekleşme sayısı 10^k", min: 1, max: 3.5, step: 0.5, value: 2.5 },
  ],
  validate(q) { return q.st === 2 && Math.abs(q.a) >= 1 ? [{ level: "error", msg: "Başlangıç: |a| = 1'de durağan dağılım yoktur (varyans sonsuz)." }] : []; },
  seed: true,
  run(q, r) {
    const N = Math.round(10 ** q.logN), a = q.a, T = q.T, s2 = 1, vInf = Math.abs(a) < 1 ? s2 / (1 - a * a) : Infinity;
    const X = range(N).map(() => { const x = new Float64Array(T + 1); x[0] = q.st === 0 ? 0 : q.st === 1 ? 5 : Math.sqrt(vInf) * r.normal(); for (let t = 1; t <= T; t++) x[t] = a * x[t - 1] + r.normal(); return x; });
    const ts = range(T + 1), mean = ts.map((t) => X.reduce((s, x) => s + x[t], 0) / N), vr = ts.map((t, i) => X.reduce((s, x) => s + (x[t] - mean[i]) ** 2, 0) / (N - 1));
    const x0 = q.st === 1 ? 5 : 0, mTh = ts.map((t) => x0 * a ** t);
    const vTh = ts.map((t) => (q.st === 2 ? vInf : a === 1 ? t : a === -1 ? t : (s2 * (1 - a ** (2 * t))) / (1 - a * a)));
    const tav = X.map((x) => x.reduce((s, v) => s + v, 0) / (T + 1)), tavMean = tav.reduce((s, v) => s + v, 0) / N, tavSd = Math.sqrt(tav.reduce((s, v) => s + (v - tavMean) ** 2, 0) / (N - 1));
    const P = [];
    if (Math.abs(a) < 1) {
      const tau = Math.log(0.05) / (2 * Math.log(Math.abs(a) || 1e-9));
      P.push(`|a| < 1: süreç kararlı, durağan varyans σ²/(1 − a²) = ${fmt(vInf, 4)}. Son adımda topluluk varyansı ${fmt(vr[T], 4)} (kuramsal ${fmt(vTh[T], 4)}).`);
      if (q.st === 2) P.push("x₀ durağan dağılımdan çekildi: ortalama ve varyans en baştan sabit, süreç geniş anlamda durağan (Gauss olduğu için katı anlamda da).");
      else P.push(`Başlangıç durağan değil: ${q.st === 1 ? "ortalama aᵗ·5 ile, " : ""}varyans σ²(1 − a²ᵗ)/(1 − a²) ile durağan değere yaklaşıyor (≈ ${fmt(Math.max(1, tau), 2)} adımda farkın %95'i kapanır). Bu geçici evre bitince süreç pratikte durağandır.`);
      P.push(`Ergodiklik: tek bir yolun zaman ortalamasının yollar arası standart sapması ${fmt(tavSd, 3)}; T büyüdükçe 0'a iner ve zaman ortalaması topluluk ortalamasına (0) yakınsar.`);
    } else {
      P.push(`|a| = 1: ${a === 1 ? "rastgele yürüyüş (Wiener sürecinin ayrık karşılığı)" : "işaret değiştiren rastgele yürüyüş"}. Varyans t ile doğrusal büyür (son adımda ${fmt(vr[T], 4)}, kuramsal ${T}): süreç durağan değildir. Artımları ise bağımsız ve durağandır.`);
      P.push(`Zaman ortalaması yakınsamaz: yollar arası standart sapması ${fmt(tavSd, 3)} ve T ile büyür (≈ √(T/3)). Tek bir yoldan topluluk istatistiği kestirilemez.`);
    }
    if (a < 0) P.push("a < 0: komşu adımlar ters işaretli ilişkili, yol testere dişi gibi salınır; C(1) = a·σ²/(1 − a²) negatiftir.");
    const k = Math.min(6, N), band = ts.map((t, i) => 2 * Math.sqrt(vTh[i]));
    return {
      plots: [
        { title: `${k} gerçekleşme ve kuramsal ortalama ± 2σ`, xlabel: "t", ylabel: "x_t", wide: true, size: { w: 620, h: 220 },
          series: [...range(k).map((i) => ({ name: i === 0 ? "gerçekleşmeler" : "", x: ts, y: Array.from(X[i]), c: 0 })), { name: "ortalama", x: ts, y: mTh, c: 2 }, { name: "± 2σ", x: ts, y: mTh.map((m, i) => m + band[i]), c: 2, dash: true }, { name: "", x: ts, y: mTh.map((m, i) => m - band[i]), c: 2, dash: true }] },
        { title: `Topluluk varyansı (N = ${N})`, xlabel: "t", ylabel: "Var[x_t]", wide: true, size: { w: 620, h: 190 }, series: [{ name: "simülasyon", x: ts, y: vr, c: 0 }, { name: "kuramsal", x: ts, y: vTh, c: 2, dash: true }] },
      ],
      stats: [{ label: "Durağan varyans", value: Number.isFinite(vInf) ? fmt(vInf, 4) : "yok" }, { label: `Var[x_${T}] (sim. / kuram)`, value: `${fmt(vr[T], 3)} / ${fmt(vTh[T], 3)}` }, { label: "Zaman ortalamalarının std'si", value: fmt(tavSd, 3) }],
      note: P.join(" "),
    };
  },
};
