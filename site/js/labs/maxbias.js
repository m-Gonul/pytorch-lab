// M11.5 — Sutton & Barto Örnek 6.7: maksimizasyon yanlılığı. A'dan iki eylem: sağ → son durum (ödül 0); sol → B (ödül 0). B'den n eylem,
// her biri son duruma, ödül N(μ, 1). μ < 0 ise soldan gitmek kötüdür; en iyi politika A'da hep sağ. ε-açgözlü (ε = 0.1) davranışla
// Q-öğrenme ve çift Q-öğrenme (iki tablo, rastgele biri güncellenir, hedefte öteki tablonun değeri kullanılır), γ = 1, sabit α.
// R çalıştırmanın ortalamasında bölüm başına A'da sol seçilme oranı; en iyi oran ε/2 (rastgele adımın yarısı).
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "n", label: "B'deki eylem sayısı", min: 1, max: 20, value: 10, int: true },
    { id: "mu", label: "B'deki ödüllerin ortalaması μ", min: -0.5, max: 0.3, step: 0.05, value: -0.1 },
    { id: "eps", label: "ε", min: 0.02, max: 0.3, step: 0.01, value: 0.1 },
    { id: "alpha", label: "α", min: 0.02, max: 0.5, step: 0.01, value: 0.1 },
    { id: "R", label: "Çalıştırma", min: 50, max: 10000, step: 50, value: 1000, int: true },
  ],
  seed: true,
  run(q, rnd) {
    const E = 300, left = [Array(E).fill(0), Array(E).fill(0)];
    const am = (arr) => { const m = Math.max(...arr), c = []; arr.forEach((v, i) => v === m && c.push(i)); return c[Math.floor(rnd() * c.length)]; };
    for (let k = 0; k < q.R; k++) {
      // tek Q
      let QA = [0, 0], QB = Array(q.n).fill(0);
      for (let e = 0; e < E; e++) {
        const a = rnd() < q.eps ? Math.floor(rnd() * 2) : am(QA);
        if (a === 1) { left[0][e] += 1 / q.R; QA[1] += q.alpha * (0 + Math.max(...QB) - QA[1]); const b = rnd() < q.eps ? Math.floor(rnd() * q.n) : am(QB); QB[b] += q.alpha * (q.mu + rnd.normal() - QB[b]); }
        else QA[0] += q.alpha * (0 - QA[0]);
      }
      // çift Q
      const A1 = [0, 0], A2 = [0, 0], B1 = Array(q.n).fill(0), B2 = Array(q.n).fill(0);
      for (let e = 0; e < E; e++) {
        const sumA = [A1[0] + A2[0], A1[1] + A2[1]], a = rnd() < q.eps ? Math.floor(rnd() * 2) : am(sumA);
        if (a === 1) {
          left[1][e] += 1 / q.R;
          if (rnd() < 0.5) { const bs = am(B1); A1[1] += q.alpha * (B2[bs] - A1[1]); } else { const bs = am(B2); A2[1] += q.alpha * (B1[bs] - A2[1]); }
          const sb = B1.map((v, i) => v + B2[i]), b = rnd() < q.eps ? Math.floor(rnd() * q.n) : am(sb), r = q.mu + rnd.normal();
          if (rnd() < 0.5) B1[b] += q.alpha * (r - B1[b]); else B2[b] += q.alpha * (r - B2[b]);
        } else { if (rnd() < 0.5) A1[0] += q.alpha * (0 - A1[0]); else A2[0] += q.alpha * (0 - A2[0]); }
      }
    }
    const xs = range(E).map((i) => i + 1), opt = q.mu < 0 ? q.eps / 2 : q.mu > 0 ? 1 - q.eps / 2 : null, pk = (a) => Math.max(...a.slice(0, 50)), tl = (a) => a.slice(-50).reduce((x, y) => x + y, 0) / 50;
    const P = [`A'da sol seçilme oranı — Q-öğrenme: ilk 50 bölümde en yüksek ${fmt(100 * pk(left[0]), 3)}%, son 50 bölüm ortalaması ${fmt(100 * tl(left[0]), 3)}%; çift Q: ${fmt(100 * pk(left[1]), 3)}% ve ${fmt(100 * tl(left[1]), 3)}%. ${opt===null?'μ=0 için bütün sol/sağ karışımları optimal.':`En iyi ε-greedy davranışta sol ${fmt(100 * opt,3)}%.`}`];
    if (q.mu < 0 && q.n > 1) P.push(`B'deki ${q.n} kestirimin en büyüğü, gerçek ortalaması ${fmt(q.mu, 2)} olsa da gürültü yüzünden pozitif çıkar: Q-öğrenmenin hedefi max_b Q(B, b) yukarı yanlıdır ve A'da sol başta çekici görünür. Çift Q-öğrenme seçimi bir tabloyla,değerlendirmeyi ötekiyle yapar;bu örnekte aşırı sol tercihini azaltır. Adaptif öğrenilen iki tablonun genel olarak bağımsız olduğu veya hedefin her problemde tam yansız kaldığı sonucu çıkmaz.`);
    else if(q.mu===0)P.push("μ=0:iki başlangıç eyleminin gerçek beklenen getirisi eşittir. Tek bir optimal sol oranı yoktur;gösterilen öğrenme farkı hangi eşdeğer eylemin tercih edildiğini anlatır.");
    else P.push(`μ>0:sol gerçekten daha iyi;en iyi ε-greedy sol oranı%${fmt(100*opt,3)}. Gürültüσ1 olduğu için sınırlı bütçede öğrenme yavaş olabilir. Çift tablonun veri/öğrenme geçmişi farklıdır;genel tam yansızlık veya her koşulda üstünlük garantisi değildir.`);
    if (q.n === 1) P.push("B’de tek eylem varken max karşılaştırması kalmaz;bu maksimizasyon yanlılığı mekanizması yoktur. İki tablo veri ve güncelleme geçmişini ayırdığından iki yöntem aynı eğriyi vermek zorunda değildir.");
    return {
      plots: [{ title: "A'da sol seçilme oranı", xlabel: "bölüm", ylabel: "% sol", wide: true, size: { w: 620, h: 240 }, ylim: [0, 100],
        series: [{ name: "Q-öğrenme", x: xs, y: left[0].map((v) => 100 * v), c: 2 }, { name: "çift Q-öğrenme", x: xs, y: left[1].map((v) => 100 * v), c: 0 }, ...(opt===null?[]:[{ name: "en iyi ε-greedy", x: [1, E], y: [100 * opt, 100 * opt], c: 4, dash: true }])] }],
      stats: [{ label: "Q-öğrenme: en yüksek / son (%)", value: `${fmt(100 * pk(left[0]), 3)} / ${fmt(100 * tl(left[0]), 3)}` }, { label: "Çift Q: en yüksek / son (%)", value: `${fmt(100 * pk(left[1]), 3)} / ${fmt(100 * tl(left[1]), 3)}` }],
      note: P.join(" "),
    };
  },
};
