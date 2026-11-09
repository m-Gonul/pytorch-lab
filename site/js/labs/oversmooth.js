// M5.4 — Yayılım ve aşırı düzleşme: stokastik blok modeli (c topluluk, topluluk içi p_in, arası p_out), düğüm özellikleri
// gürültülü topluluk göstergeleri. Eğitimsiz yayılım H_k = Â^k X (Â: simetrik normalize, öz-döngülü; ya da ortalama D̃⁻¹Ã;
// ya da öz-döngüsüz D^{-1/2}AD^{-1/2}). Her k için: az etiketle (topluluk başına m düğüm) en yakın merkez sınıflandırıcısının
// doğruluğu ve özelliklerin "yayılımı" (düğümler arası ortalama uzaklık / k = 0'daki).
import { fmt, range } from "./plot.js";

export function sbm(n, c, pin, pout, r) {
  const y = range(n).map((i) => i % c), adj = range(n).map(() => []);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (r() < (y[i] === y[j] ? pin : pout)) { adj[i].push(j); adj[j].push(i); }
  return { y, adj };
}
export function propagate(H, adj, norm) {
  const n = H.length, d = H[0].length, deg = adj.map((a) => a.length + (norm === 2 ? 0 : 1)), out = range(n).map(() => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const nb = norm === 2 ? adj[i] : [i, ...adj[i]];
    for (const j of nb) { const w = norm === 1 ? 1 / deg[i] : 1 / Math.sqrt(Math.max(1, deg[i]) * Math.max(1, deg[j])); for (let k = 0; k < d; k++) out[i][k] += w * H[j][k]; }
  }
  return out;
}

export default {
  params: [
    { id: "pin", label: "Topluluk içi kenar olasılığı p_in", min: 0.01, max: 0.5, step: 0.01, value: 0.1 },
    { id: "pout", label: "Topluluklar arası kenar olasılığı p_out", min: 0, max: 0.2, step: 0.005, value: 0.01 },
    { id: "noise", label: "Özellik gürültüsü σ", min: 0.1, max: 5, step: 0.1, value: 2 },
    { id: "norm", label: "Yayılım matrisi", options: ["simetrik D̃^{−½}ÃD̃^{−½} (GCN)", "ortalama D̃⁻¹Ã", "öz-döngüsüz D^{−½}AD^{−½}"], value: 0 },
    { id: "K", label: "En fazla yayılım adımı K", min: 2, max: 60, value: 30, int: true },
    { id: "m", label: "Topluluk başına etiketli düğüm", min: 1, max: 20, value: 3, int: true },
  ],
  seed: true,
  run(q, r) {
    const n = 150, c = 3, G = sbm(n, c, q.pin, q.pout, r), d = 8;
    let H = range(n).map((i) => range(d).map((k) => (k === G.y[i] ? 1 : 0) + q.noise * r.normal()));
    const lab = range(c).flatMap((k) => range(n).filter((i) => G.y[i] === k).slice(0, q.m));
    const acc = [], spread = [], ratio = [];
    const dist = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
    let s0 = null;
    for (let k = 0; k <= q.K; k++) {
      const cent = range(c).map((cl) => { const L = lab.filter((i) => G.y[i] === cl); return range(d).map((t) => L.reduce((s, i) => s + H[i][t], 0) / L.length); });
      let ok = 0; for (let i = 0; i < n; i++) { const best = cent.map((ct, cl) => [dist(H[i], ct), cl]).sort((a, b) => a[0] - b[0])[0][1]; if (best === G.y[i]) ok++; }
      acc.push(ok / n);
      const mu = range(d).map((t) => H.reduce((s, h) => s + h[t], 0) / n), sp = H.reduce((s, h) => s + dist(h, mu), 0) / n; if (s0 == null) s0 = sp; spread.push(sp / s0);
      const cm = range(c).map((cl) => { const L = range(n).filter((i) => G.y[i] === cl); return range(d).map((t) => L.reduce((s, i) => s + H[i][t], 0) / L.length); });
      const within = H.reduce((s, h, i) => s + dist(h, cm[G.y[i]]), 0) / n, between = (dist(cm[0], cm[1]) + dist(cm[0], cm[2]) + dist(cm[1], cm[2])) / 3;
      ratio.push(between / Math.max(within, 1e-12));
      if (k < q.K) H = propagate(H, G.adj, q.norm);
    }
    const best = acc.indexOf(Math.max(...acc)), avgDeg = G.adj.reduce((s, a) => s + a.length, 0) / n, isolated = G.adj.filter((a) => a.length === 0).length;
    const ks = range(q.K + 1);
    const P = [`150 düğüm, 3 topluluk, ortalama derece ${fmt(avgDeg, 3)}${isolated ? ` (${isolated} yalıtık düğüm)` : ""}. Yayılımsız (k = 0) en yakın merkez doğruluğu ${fmt(acc[0], 3)}; en iyi doğruluk ${fmt(acc[best], 3)} k = ${best} adımda; k = ${q.K}'da ${fmt(acc[q.K], 3)}.`];
    P.push(acc[q.K] < acc[best] - 0.05 ? `Önce yayılım komşuların gürültüsünü ortalar ve topluluk sinyalini güçlendirir (ayrışma oranı k = 0'da ${fmt(ratio[0], 3)}, en iyi k'da ${fmt(ratio[best], 3)}). Adım sayısı arttıkça topluluklar arası kenarlar da karışır ve tüm düğümler aynı vektöre yaklaşır: <b>aşırı düzleşme</b>. Özelliklerin yayılımı başlangıcın ${fmt(spread[q.K], 3)} katına indi.` : `Bu ayarda ${q.K} adımda belirgin bir düşüş yok: topluluklar arası kenar az, Âᵏ toplulukları ayrı tutuyor (yayılım başlangıcın ${fmt(spread[q.K], 3)} katı).`);
    if (spread[q.K] < 0.05 && acc[q.K] > acc[best] - 0.05) P.push(`Dikkat: özellikler neredeyse tek bir vektöre çöktü (yayılım ${fmt(spread[q.K], 3)}) ama doğruluk düşmedi. Kalan çok küçük farklar ikinci özvektör yönünde; topluluk grafında bu yön topluluk göstergesidir ve ölçekten bağımsız en yakın merkez sınıflandırıcısı onu hâlâ okuyabiliyor. Gerçek ağlarda gürültü, sonlu hassasiyet ve öğrenilen katmanlar arasındaki doğrusal olmayan dönüşümler bu küçük farkları siler: çöken özellikler kullanışsız olur.`);
    if (q.norm === 2) P.push("Öz-döngü yok: her düğüm kendi özelliğini değil yalnız komşularınkini alır; iki parçalı yapılarda (tek/çift adım) salınım olabilir.");
    return {
      plots: [
        { title: "En yakın merkez doğruluğu ve yayılım adımı", xlabel: "k (Âᵏ X)", ylabel: "doğruluk", ylim: [0, 1.02], series: [{ name: "doğruluk", x: ks, y: acc, c: 0, dots: q.K <= 30 }], marks: [{ y: 1 / c, label: "rastgele" }, { x: best, label: "en iyi" }] },
        { title: "Ayrışma ve çöküş", xlabel: "k", ylabel: "oran", ylog: true, series: [
          { name: "topluluklar arası / içi uzaklık", x: ks, y: ratio, c: 2 },
          { name: "özelliklerin yayılımı (k = 0'a göre)", x: ks, y: spread, c: 3, dash: true }] },
      ],
      stats: [
        { label: "Doğruluk, k = 0", value: acc[0] },
        { label: "En iyi doğruluk / k", value: `${fmt(acc[best], 3)} / ${best}` },
        { label: `Doğruluk, k = ${q.K}`, value: acc[q.K] },
        { label: "Ortalama derece", value: avgDeg },
        { label: `Yayılım (k = ${q.K} / k = 0)`, value: spread[q.K] },
      ],
      note: P.join(" "),
    };
  },
};
