// M5.1 — Doğrusal β-VAE'de posterior çöküşü (Lucas ve ark. 2019; olasılıksal PCA'nın genelleşmesi). Veri kovaryansının
// özdeğerleri λ_i = λ₁·ρ^{i−1} (i = 1..16), gözlem gürültüsü σ², KL ağırlığı β, gizil boyut k. Optimumda gizil boyut i
// (ilk k özyönden) ancak λ_i > βσ² ise kullanılır; kod çözücü sütun normu √(λ_i − βσ²), boyut başına KL ½·log(λ_i/(βσ²)).
// Diğerleri çöker: q(z_i|x) = p(z_i), kod çözücü o boyutu yok sayar.
import { fmt, range } from "./plot.js";

export function linvae(eigs, k, s2, beta) {
  const th = beta * s2;
  return eigs.map((l, i) => { const act = i < k && l > th; return { l, act, norm: act ? Math.sqrt(l - th) : 0, kl: act ? 0.5 * Math.log(l / th) : 0 }; });
}

export default {
  params: [
    { id: "beta", label: "KL ağırlığı β", min: 0.01, max: 50, value: 1, log: true },
    { id: "s2", label: "Gözlem gürültüsü σ²", min: 0.001, max: 1, value: 0.02, log: true },
    { id: "rho", label: "Özdeğer azalma oranı ρ (λ_i = ρ^{i−1})", min: 0.3, max: 0.98, step: 0.01, value: 0.7 },
    { id: "k", label: "Gizil boyut k", min: 1, max: 16, value: 8, int: true },
  ],
  run(q) {
    const D = 16, eigs = range(D).map((i) => q.rho ** i), R = linvae(eigs, q.k, q.s2, q.beta), act = R.filter((x) => x.act).length, th = q.beta * q.s2;
    const tot = eigs.reduce((a, b) => a + b, 0), cap = R.reduce((a, x) => a + (x.act ? x.l - th : 0), 0) / tot, kl = R.reduce((a, x) => a + x.kl, 0);
    const P = [`Eşik βσ² = ${fmt(th, 3)}. İlk k = ${q.k} özyönün ${act} tanesi kullanılıyor; ${q.k - act} gizil boyut <b>çökmüş</b> (KL = 0: kodlayıcı o boyutta önselin aynısını veriyor, kod çözücü yok sayıyor).`];
    P.push(`Kod çözücünün modellediği sinyal varyansı Σ(λ_i − βσ²) toplam varyansın %${fmt(100 * cap, 3)} kadarı; toplam KL ${fmt(kl, 3)} nat. β büyüdükçe eşik yükselir: KL'nin ağırlığı arttıkça, verinin yalnız eşikten güçlü yönleri gizil koda "girmeye değer".`);
    if (act === 0) P.push("<b>Tam çöküş:</b> hiçbir boyut eşiği aşmıyor; model z'yi hiç kullanmıyor ve her örnek için aynı (ortalama) çıktıyı üretiyor. Güçlü bir kod çözücüyle (ör. otoregresif) bu, β = 1'de bile olabilir: posterior çöküşü.");
    else if (act < q.k) P.push(`Eşik yüzünden k = ${q.k} gizil boyutun hepsi kullanılmıyor; β < ${fmt(eigs[Math.min(q.k, D) - 1] / q.s2, 3)} olursa k. boyut da açılır.`);
    return {
      plots: [
        { title: "Veri özdeğerleri ve eşik βσ²", xlabel: "özyön i", ylabel: "λ_i (log)", ylog: true, ylim: [Math.min(...eigs, th) / 2, 2], series: [
          { name: "kullanılan gizil boyut", x: R.map((_, i) => i + 1).filter((_, i) => R[i].act), y: R.filter((x) => x.act).map((x) => x.l), kind: "bar", c: 0 },
          { name: "çökmüş / k dışında", x: R.map((_, i) => i + 1).filter((_, i) => !R[i].act), y: R.filter((x) => !x.act).map((x) => x.l), kind: "bar", c: 4 }], marks: [{ y: th, label: "βσ²" }] },
        { title: "Boyut başına KL (nat)", xlabel: "özyön i", ylabel: "KL", yzero: true, series: [{ name: "½ log(λ_i / βσ²)", x: range(D).map((i) => i + 1), y: R.map((x) => x.kl), kind: "bar", c: 2 }] },
      ],
      stats: [
        { label: "Aktif gizil boyut", value: act },
        { label: "Eşik βσ²", value: th },
        { label: "Toplam KL (nat)", value: kl },
        { label: "Modellenen sinyal varyansı oranı", value: cap },
      ],
      note: P.join(" "),
    };
  },
};
