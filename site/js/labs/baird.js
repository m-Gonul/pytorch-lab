// M11.6 — Baird'in karşı örneği (1995; Sutton & Barto Örnek 11.1): 7 durum, doğrusal değer fonksiyonu, 8 ağırlık. Durum i (0..5) için
// v(i) = 2w_i + w_7, durum 6 için v(6) = w_6 + 2w_7. "Kesikli" eylem 6 durumdan birine düzgün, "düz" eylem durum 6'ya götürür; bütün
// ödüller 0, dolayısıyla gerçek değer her yerde 0 ve w = 0 kesin çözümdür. Davranış politikası 6/7 kesikli, 1/7 düz; hedef politika hep
// düz. Politika dışı yarı-gradyan TD(0) (önem örneklemesi ρ = π/b) ağırlıkları ıraksatır; aynı TD politika üstü örneklemeyle (hedef
// politikanın durum dağılımı) ya da politika dışı TDC (gradyan-TD) ile kararlıdır. Beklenen (DP) güncelleme seçeneği gürültüyü kaldırır.
import { fmt, range } from "./plot.js";

const X = range(7).map((i) => { const x = Array(8).fill(0); if (i < 6) { x[i] = 2; x[7] = 1; } else { x[6] = 1; x[7] = 2; } return x; });
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export default {
  params: [
    { id: "alg", label: "Yöntem", options: ["politika dışı yarı-gradyan TD(0)", "politika üstü yarı-gradyan TD(0)", "politika dışı TDC (gradyan-TD)"], value: 0 },
    { id: "g", label: "İndirim γ", min: 0.5, max: 0.99, step: 0.01, value: 0.99 },
    { id: "alpha", label: "α", min: 0.001, max: 0.02, step: 0.001, value: 0.01 },
    { id: "N", label: "Adım sayısı", min: 100, max: 5000, step: 100, value: 1000, int: true },
  ],
  seed: true,
  run(q, r) {
    let w = [1, 1, 1, 1, 1, 1, 10, 1], h = Array(8).fill(0), s = Math.floor(r() * 7);
    const hist = range(8).map(() => []), xs = [];
    for (let t = 0; t <= q.N; t++) {
      if (t % Math.max(1, Math.floor(q.N / 200)) === 0) { xs.push(t); w.forEach((v, i) => hist[i].push(v)); }
      if (t === q.N) break;
      let solid, rho;
      if (q.alg === 1) { solid = true; rho = 1; } else { solid = r() >= 6 / 7; rho = solid ? 7 : 0; }
      const s2 = solid ? 6 : Math.floor(r() * 6), x = X[s], x2 = X[s2], delta = 0 + q.g * dot(w, x2) - dot(w, x);
      if (q.alg === 2) { const beta = q.alpha * 5, hx = dot(h, x); w = w.map((v, i) => v + q.alpha * rho * (delta * x[i] - q.g * x2[i] * hx)); h = h.map((v, i) => v + beta * (rho * delta - hx) * x[i]); }
      else w = w.map((v, i) => v + q.alpha * rho * delta * x[i]);
      s = s2;
      if (!w.every(Number.isFinite) || Math.max(...w.map(Math.abs)) > 1e12) break;
    }
    const nrm = Math.sqrt(w.reduce((a, v) => a + v * v, 0)), ve = Math.sqrt(range(7).reduce((a, i) => a + dot(w, X[i]) ** 2, 0) / 7);
    const P = [`${q.N} adım sonra ‖w‖ = ${fmt(nrm, 3)}; değer hatası (gerçek değer 0) ${fmt(ve, 3)}.`];
    if (q.alg === 0) P.push(nrm > 20 ? "Ödüller hep 0 ve kesin çözüm temsil edilebilir (w = 0) olmasına karşın ağırlıklar ıraksıyor: fonksiyon yaklaşımı, bootstrapping ve politika dışı öğrenme bir arada — 'ölümcül üçlü'." : "Bu ayarda ağırlıklar başlangıç büyüklüğünü (‖w₀‖ ≈ 10.3) aşmıyor: küçük γ'da önem ağırlıklı güncellemelerin büyütme etkisi zayıf; γ'yı büyüt ya da adım sayısını artır.");
    if (q.alg === 1) P.push("Aynı yöntem politika üstü (veri hedef politikanın dağılımından) olunca kararlı: üçlünün bir ayağı kalktı. Doğrusal politika üstü TD'nin yakınsaması kanıtlıdır (Tsitsiklis ve Van Roy 1997). Hedef politika hep 'düz' olduğu için yalnız durum 6 ziyaret edilir: v(6) sıfıra iner, diğer durumların değerleri hiç eğitilmez ve kalan hata onlardandır.");
    if (q.alg === 2) P.push("TDC, izdüşürülmüş Bellman hatasının gerçek gradyanını (ikinci bir ağırlık vektörü h ile) izler; politika dışı ve doğrusal yaklaşımda ıraksamaz. Sabit adımla ve büyük önem ağırlıklarıyla (ρ = 7) değer hatası sıfırın üstünde dalgalanır; azalan adımla yakınsar. Yarı-gradyan TD'nin, hedefi türevlemeyen 'gradyanı' ise bir fonksiyonun gradyanı değildir.");
    return {
      plots: [{ title: "Ağırlıklar", xlabel: "adım", ylabel: "w_i", wide: true, size: { w: 620, h: 250 },
        series: range(8).map((i) => ({ name: i < 6 ? (i === 0 ? "w₀…w₅" : "") : i === 6 ? "w₆" : "w₇", x: xs.slice(0, hist[i].length), y: hist[i], c: i < 6 ? 0 : i === 6 ? 2 : 1 })) }],
      stats: [{ label: "‖w‖", value: fmt(nrm, 3) }, { label: "Değer hatası (RMS)", value: fmt(ve, 3) }],
      note: P.join(" "),
    };
  },
};
