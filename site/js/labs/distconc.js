// M2.1 — Boyutun laneti: yüksek boyutta en yakın ve en uzak komşu neredeyse aynı uzaklıktadır; "yerel" komşuluk bile küpün çoğunu kaplar.
import { fmt, linspace, range } from "./plot.js";

export function contrast(d, n, r) {
  const q = Array.from({ length: d }, () => r()), ds = [];
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) { const t = r() - q[j]; s += t * t; } ds.push(Math.sqrt(s)); }
  const mn = Math.min(...ds), mx = Math.max(...ds);
  return { ds, mn, mx, rc: (mx - mn) / mn };
}

export default {
  params: [
    { id: "d", label: "Boyut d", min: 1, max: 1000, value: 2, int: true, log: true },
    { id: "n", label: "Nokta sayısı n (birim küpte düzgün)", min: 50, max: 5000, value: 1000, int: true, log: true },
    { id: "frac", label: "Komşulukta istenen veri oranı r", min: 0.001, max: 0.5, value: 0.01, log: true, help: "Bu kadar veriyi içeren alt küpün kenar uzunluğu r^(1/d)." },
  ],
  seed: true,
  run(q, r) {
    const d = Math.round(q.d), n = Math.round(q.n), C = contrast(d, n, r);
    const nb = 40, lo = C.mn, hi = C.mx, w = (hi - lo) / nb || 1, cnt = new Array(nb).fill(0);
    for (const v of C.ds) cnt[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
    const dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000], rcs = dims.map((dd) => contrast(dd, 400, r).rc);
    const rs = linspace(0.001, 1, 200);
    return {
      plots: [
        { title: `d = ${d}: rastgele bir sorgudan ${n} noktaya uzaklıkların dağılımı`, xlabel: "uzaklık", ylabel: "sayı", yzero: true, series: [{ name: "uzaklık histogramı", x: range(nb).map((i) => lo + (i + 0.5) * w), y: cnt, kind: "bar", c: 0, bw: 0.95 }], marks: [{ x: C.mn, label: "en yakın" }, { x: C.mx, label: "en uzak" }] },
        { title: "Göreli kontrast (en uzak − en yakın) / en yakın, boyuta göre", xlabel: "log₁₀ d", ylabel: "göreli kontrast", ylog: true, series: [{ name: "400 nokta, deneysel", x: dims.map(Math.log10), y: rcs, c: 2, dots: true }], marks: [{ x: Math.log10(d), label: "seçili d" }] },
        { title: "Verinin r oranını içeren alt küpün kenarı: r^(1/d)", xlabel: "istenen veri oranı r", ylabel: "kenar uzunluğu (küpün kesri)", ylim: [0, 1.02], series: [1, 2, 10, 100].map((dd, i) => ({ name: `d = ${dd}`, x: rs, y: rs.map((v) => v ** (1 / dd)), c: i })), marks: [{ x: q.frac, label: `r = ${fmt(q.frac, 2)}` }] },
      ],
      stats: [
        { label: "En yakın / en uzak uzaklık", value: `${fmt(C.mn, 3)} / ${fmt(C.mx, 3)}` },
        { label: "Göreli kontrast", value: C.rc },
        { label: `r = ${fmt(q.frac, 3)} için alt küp kenarı, d = ${d}`, value: q.frac ** (1 / d) },
        { label: "Ortalama uzaklık (teori ≈ √(d/6))", value: `${fmt(C.ds.reduce((a, b) => a + b, 0) / n, 3)} (≈ ${fmt(Math.sqrt(d / 6), 3)})` },
      ],
      note: `Düşük boyutta en yakın komşu en uzaktan çok daha yakındır (kontrast büyük). Boyut arttıkça uzaklıklar ortalamaları √(d/6) etrafında <b>yoğunlaşır</b>: her nokta kabaca aynı uzaklıktadır ve "en yakın komşu" anlamını yitirir. Üçüncü grafik aynı gerçeği hacimle anlatır: d = 100'de verinin yalnız %1'ini kapsamak için küpün kenarının %95'ini almak gerekir — "yerel" komşuluk aslında neredeyse bütün uzaydır. Gerçek veride k-NN'yi kurtaran şey, verinin yüksek boyutlu uzayın düşük boyutlu bir yapısına (manifold) yakın durmasıdır; düzgün dağılmış veride çalışmaz.`,
    };
  },
};
