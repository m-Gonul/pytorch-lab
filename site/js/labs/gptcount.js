// M4.7 — GPT'nin parametre ve hesap sayımı: gömme/dikkat/MLP/LN/çıkış dağılımı, ağırlık bağlama, token başına ileri geçiş FLOP'u
// (parametreli matris çarpımları 2N + nedensel dikkat 2·L·T·d) ve bağlam uzunluğuyla değişimi.
import { fmt } from "./plot.js";

export function count(q) {
  const { V, T, d, L, r } = q, tie = q.tie === 1;
  const wte = V * d, wpe = T * d, attn = L * (4 * d * d + 4 * d), mlp = L * (2 * r * d * d + r * d + d), ln = L * 4 * d + 2 * d, head = tie ? 0 : V * d;
  const total = wte + wpe + attn + mlp + ln + head, nonEmb = attn + mlp + ln;
  // token başına ileri FLOP: parametreli matris çarpımları (çıkış izdüşümü dahil, gömme araması hariç) + nedensel dikkat (ortalama bağlam T/2, QKᵀ ve αV)
  const fMat = 2 * (L * (4 * d * d + 2 * r * d * d) + V * d), fAtt = (T) => 2 * L * T * d;
  return { wte, wpe, attn, mlp, ln, head, total, nonEmb, fMat, fAtt };
}
const big = (n) => (n >= 1e9 ? `${fmt(n / 1e9, 4)} milyar` : n >= 1e6 ? `${fmt(n / 1e6, 4)} milyon` : n.toLocaleString("tr-TR"));

export default {
  params: [
    { id: "d", label: "Model boyu d (d_model)", min: 64, max: 16384, value: 768, int: true, log: true },
    { id: "L", label: "Blok sayısı L", min: 1, max: 128, value: 12, int: true },
    { id: "V", label: "Sözlük boyu V", min: 256, max: 262144, value: 50257, int: true, log: true },
    { id: "T", label: "Bağlam uzunluğu T (block_size)", min: 64, max: 131072, value: 1024, int: true, log: true },
    { id: "r", label: "MLP genişlik oranı (d_ff / d)", min: 1, max: 8, step: 0.5, value: 4 },
    { id: "tie", label: "Çıkış katmanı", options: ["ayrı matris", "wte ile bağlı (weight tying)"], value: 1 },
  ],
  validate(q) { return q.d % 64 !== 0 ? [{ level: "warn", msg: `d = ${q.d}, 64'ün katı değil: gerçek modellerde d = baş sayısı × baş boyu (çoğu kez 64 ya da 128) seçilir. Sayım yine doğru.` }] : []; },
  run(q) {
    const c = count(q), M = 1e6, Ts = [], fm = [], fa = [];
    for (let t = 64; t <= 131072; t *= 2) { Ts.push(Math.log2(t)); fm.push(c.fMat); fa.push(c.fAtt(t)); }
    const xt = [6, 8, 10, 12, 14, 16].map((e) => [e, e < 10 ? String(2 ** e) : `${2 ** (e - 10)}K`]);
    const parts = [["token gömme wte", c.wte], ["konum gömme wpe", c.wpe], ["dikkat (L blok)", c.attn], ["MLP (L blok)", c.mlp], ["LayerNorm'lar", c.ln], ["çıkış (bağlı değilse)", c.head]];
    const attShare = c.fAtt(q.T) / (c.fMat + c.fAtt(q.T));
    return {
      plots: [
        { title: "Parametrelerin dağılımı (milyon)", xlabel: "", ylabel: "milyon parametre", yzero: true, xlim: [-0.6, 5.6], xticks: parts.map((p, i) => [i, p[0]]), series: [{ name: "parametre", x: parts.map((_, i) => i), y: parts.map((p) => p[1] / M), kind: "bar", c: 0 }] },
        { title: "Token başına ileri geçiş FLOP'u, bağlam uzunluğuna göre (log–log)", xlabel: "bağlam uzunluğu T (log₂ ekseni)", ylabel: "FLOP / token", ylog: true, xticks: xt, series: [{ name: "parametreli çarpımlar 2N", x: Ts, y: fm, c: 0 }, { name: "nedensel dikkat 2·L·T·d", x: Ts, y: fa, c: 2 }], marks: [{ x: Math.log2(q.T), label: `T = ${q.T}` }] },
      ],
      stats: [
        { label: "Toplam parametre", value: big(c.total) },
        { label: "Gömmeler hariç (12·L·d² ölçeği)", value: `${big(c.nonEmb)} (${fmt((100 * c.nonEmb) / c.total, 3)}%)` },
        { label: "Blok başına (12d² + 13d, r = 4'te)", value: big((c.attn + c.mlp) / q.L + 4 * q.d) },
        { label: "MLP / dikkat parametre oranı", value: fmt(c.mlp / c.attn, 3) },
        { label: "İleri FLOP / token (T'de)", value: `${fmt(c.fMat + c.fAtt(q.T), 3)} (dikkat payı %${fmt(100 * attShare, 3)})` },
        { label: "Eğitim FLOP / token ≈ 3 × ileri", value: fmt(3 * (c.fMat + c.fAtt(q.T)), 3) },
      ],
      note: `Blok başına parametre: dikkat 4d² + 4d (Q, K, V, O), MLP 2·r·d² + (r + 1)·d, iki LN 4d; r = 4'te toplam 12d² + 13d. ${c.nonEmb / c.total > 0.5 ? "Parametrelerin çoğu blokların 12·L·d² teriminde." : "<b>Bu küçük modelde parametrelerin çoğu gömmelerde</b> — V·d terimi baskın; ölçekleme yasalarında (Kaplan ve ark.) N bu yüzden gömmeler hariç sayılır."} ${q.tie === 1 ? `Bağlı çıkış katmanı ${big(q.V * q.d)} parametre tasarrufu sağlıyor (ama hesap tasarrufu yok: logit için 2·V·d FLOP yine ödenir).` : "Ayrı çıkış matrisi V·d parametre ekliyor."} Bu bağlamda dikkat hesabı ileri FLOP'un %${fmt(100 * attShare, 3)}'i: ${attShare < 0.25 ? "kısa bağlamda token başına maliyet ≈ 2N (eğitimde ≈ 6N) kuralı iyi bir yaklaşıklık." : "<b>uzun bağlamda dikkat terimi baskınlaşıyor</b>; 6N kuralı maliyeti olduğundan az gösterir."}`,
    };
  },
};
