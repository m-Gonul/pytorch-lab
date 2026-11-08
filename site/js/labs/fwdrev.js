// M3.3 — Tam Jacobian: n tane JVP (ileri mod) mu, m tane VJP (ters mod) mu? f(x) = W₂ tanh(W₁x), x ∈ ℝⁿ, f ∈ ℝᵐ, gizli h.
// İki yol da gerçekten hesaplanır, sonuçlar karşılaştırılır; maliyet (çarpma sayısı) n ve m'ye göre.
import { fmt, range } from "./plot.js";

export function model(n, m, h, r) { const W1 = range(h).map(() => range(n).map(() => r.normal() / Math.sqrt(n))), W2 = range(m).map(() => range(h).map(() => r.normal() / Math.sqrt(h))), x = range(n).map(() => r.normal()); return { W1, W2, x }; }
export function jacobians(M) {
  const { W1, W2, x } = M, h = W1.length, n = x.length, m = W2.length;
  const z = W1.map((w) => w.reduce((s, wi, i) => s + wi * x[i], 0)), d = z.map((v) => 1 - Math.tanh(v) ** 2); // ileri geçiş (bir kez)
  let mulF = h * n + h, mulR = mulF; // ileri geçişin maliyeti ikisinde de
  // ileri mod: her giriş yönü eᵢ için J eᵢ = W₂ (d ⊙ (W₁ eᵢ)) — W₁eᵢ, W₁'in i. sütunu ama genel v için h·n çarpma sayılır
  const JF = range(m).map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) { const u = W1.map((w, k) => w.reduce((s, wk, j) => s + wk * (j === i ? 1 : 0), 0) * d[k]); for (let o = 0; o < m; o++) JF[o][i] = W2[o].reduce((s, w, k) => s + w * u[k], 0); mulF += h * n + h + m * h; }
  // ters mod: her çıktı eₒ için eₒᵀ J = ((W₂ᵀ eₒ) ⊙ d)ᵀ W₁
  const JR = range(m).map(() => new Array(n).fill(0));
  for (let o = 0; o < m; o++) { const g = range(h).map((k) => W2[o][k] * d[k]); for (let i = 0; i < n; i++) JR[o][i] = g.reduce((s, gk, k) => s + gk * W1[k][i], 0); mulR += m * h + h + h * n; }
  let err = 0; for (let o = 0; o < m; o++) for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(JF[o][i] - JR[o][i]));
  return { JF, JR, mulF, mulR, err };
}

export default {
  params: [
    { id: "n", label: "Girdi boyutu n", min: 1, max: 200, value: 100, int: true, log: true },
    { id: "m", label: "Çıktı boyutu m", min: 1, max: 200, value: 1, int: true, log: true, help: "m = 1: skaler kayıp — eğitimdeki durum." },
    { id: "h", label: "Gizli boyut h", min: 4, max: 64, value: 32, int: true },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), m = Math.round(q.m), h = Math.round(q.h), J = jacobians(model(n, m, h, r));
    const ns = [1, 2, 5, 10, 20, 50, 100, 200], cost = (nn, mm) => ({ f: h * nn + h + nn * (h * nn + h + mm * h), r: h * nn + h + mm * (mm * h + h + h * nn) });
    return {
      plots: [{ title: `Tam Jacobian'ın maliyeti (m = ${m}, h = ${h}), n'ye göre`, xlabel: "log₁₀ n", ylabel: "çarpma sayısı", ylog: true, series: [
        { name: "ileri mod: n tane JVP", x: ns.map(Math.log10), y: ns.map((v) => cost(v, m).f), c: 2, dots: true },
        { name: "ters mod: m tane VJP", x: ns.map(Math.log10), y: ns.map((v) => cost(v, m).r), c: 0, dots: true },
      ], marks: [{ x: Math.log10(n), label: "seçili n" }] }],
      stats: [
        { label: "Jacobian shape", value: `(${m} × ${n})` },
        { label: "İleri mod: geçiş sayısı / çarpma", value: `${n} / ${fmt(J.mulF, 4)}` },
        { label: "Ters mod: geçiş sayısı / çarpma", value: `${m} / ${fmt(J.mulR, 4)}` },
        { label: "İki yolun en büyük farkı", value: J.err },
        { label: "Daha ucuz olan", value: J.mulF < J.mulR ? "ileri mod (JVP)" : J.mulF > J.mulR ? "ters mod (VJP)" : "eşit" },
      ],
      note: `Bir JVP, Jacobian'ın bir <b>sütununu</b> (bir girdi yönünün tüm çıktılara etkisi), bir VJP bir <b>satırını</b> (bir çıktının tüm girdilere göre gradyanı) verir; her biri kabaca bir ileri geçiş kadar sürer. Tam Jacobian için ileri mod n, ters mod m geçiş ister. Eğitimde m = 1 (skaler kayıp) ve n = milyonlarca parametre: ters mod (geri yayılım) tartışmasız kazanır. Az girdili, çok çıktılı fonksiyonlarda (ör. bir parametrenin çok sayıda çıktıya etkisi, duyarlılık analizi) ileri mod kazanır. İki yol da aynı sayıları verir (fark yalnız yuvarlama). Ters modun ayrıca bir <b>bellek</b> bedeli vardır: ara değerleri saklar (burada d vektörü); ileri mod saklamaz.`,
    };
  },
};
