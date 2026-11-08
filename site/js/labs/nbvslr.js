// M2.2 — Ng & Jordan (2002): üretken (Gaussian NB, ortak varyans) vs ayırt edici (lojistik regresyon) öğrenme eğrileri, boyut d'ye göre.
// Özniteliklerin yarısı zayıf bilgi taşır (sınıf ortalamaları ±δ), diğer yarısı yalnız gürültüdür. İsteğe bağlı ortak faktör tüm öznitelikleri sınıf içinde ilişkili yapar:
// o zaman bilgisiz öznitelikler ortak gürültüyü "ölçer" ve en iyi sınıflandırıcı onları gürültüyü çıkarmak için kullanır — NB bunu göremez.
import { fmt } from "./plot.js";

export function sample(n, d, q, r) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) { const y = i % 2, f = r.normal(), row = []; for (let j = 0; j < d; j++) row.push((j < Math.ceil(d / 2) ? (y ? 1 : -1) * q.delta : 0) + Math.sqrt(q.rho) * f + Math.sqrt(1 - q.rho) * r.normal()); X.push(row); Y.push(y); }
  return { X, Y };
}
export function nb(X, Y) { // ortak varyanslı GNB → doğrusal skor
  const d = X[0].length, m = [0, 1].map((c) => { const P = X.filter((_, i) => Y[i] === c); return Array.from({ length: d }, (_, j) => P.reduce((s, p) => s + p[j], 0) / P.length); });
  const v = Array.from({ length: d }, (_, j) => X.reduce((s, x, i) => s + (x[j] - m[Y[i]][j]) ** 2, 0) / X.length + 1e-6);
  const w = v.map((vj, j) => (m[1][j] - m[0][j]) / vj), b = -0.5 * v.reduce((s, vj, j) => s + (m[1][j] ** 2 - m[0][j] ** 2) / vj, 0);
  return (x) => x.reduce((s, xj, j) => s + w[j] * xj, b);
}
export function lr(X, Y, lam = 1e-3, steps = 300) {
  const d = X[0].length, n = X.length; let w = new Array(d).fill(0), b = 0;
  for (let t = 0; t < steps; t++) {
    const g = new Array(d).fill(0); let gb = 0;
    for (let i = 0; i < n; i++) { let z = b; for (let j = 0; j < d; j++) z += w[j] * X[i][j]; const e = 1 / (1 + Math.exp(-z)) - Y[i]; for (let j = 0; j < d; j++) g[j] += e * X[i][j]; gb += e; }
    for (let j = 0; j < d; j++) w[j] -= 0.5 * (g[j] / n + lam * w[j]); b -= (0.5 * gb) / n;
  }
  return (x) => x.reduce((s, xj, j) => s + w[j] * xj, b);
}

export default {
  params: [
    { id: "d", label: "Öznitelik sayısı d", min: 2, max: 50, value: 50, int: true },
    { id: "delta", label: "Bilgili öznitelik başına sinyal δ (özniteliklerin yarısı)", min: 0.05, max: 1, step: 0.05, value: 0.3 },
    { id: "rho", label: "Sınıf içi ortak faktör (korelasyon) ρ", min: 0, max: 0.9, step: 0.05, value: 0, help: "0: naif varsayım doğru. Büyüdükçe öznitelikler aynı gürültüyü paylaşır." },
  ],
  seed: true,
  run(q, r) {
    const d = Math.round(q.d), T = sample(1000, d, q, r), ns = [4, 8, 16, 32, 64, 128, 256], R = 4;
    const err = (f) => T.X.reduce((s, x, i) => s + ((f(x) > 0 ? 1 : 0) !== T.Y[i] ? 1 : 0), 0) / T.X.length;
    const cur = ns.map((n) => { let a = 0, c = 0; for (let k = 0; k < R; k++) { const D = sample(n, d, q, r); a += err(nb(D.X, D.Y)); c += err(lr(D.X, D.Y)); } return [a / R, c / R]; });
    const cross = ns.find((n, i) => cur[i][1] < cur[i][0]);
    return {
      plots: [{ title: `d = ${d}: test hatası, eğitim boyuna göre (${R} tekrar ort.)`, xlabel: "log₂ n (eğitim örneği)", ylabel: "test hatası", series: [
        { name: "Gaussian NB (ortak varyans)", x: ns.map(Math.log2), y: cur.map((c) => c[0]), c: 2, dots: true },
        { name: "lojistik regresyon", x: ns.map(Math.log2), y: cur.map((c) => c[1]), c: 0, dots: true },
      ] }],
      stats: [
        { label: "n = 8'de NB / LR hatası", value: `${fmt(cur[1][0], 3)} / ${fmt(cur[1][1], 3)}` },
        { label: "n = 256'da NB / LR hatası", value: `${fmt(cur[6][0], 3)} / ${fmt(cur[6][1], 3)}` },
        { label: "LR'nin NB'yi ilk geçtiği n", value: cross ?? "denenen aralıkta geçmedi" },
      ],
      note: `İki model aynı aileden (doğrusal sınır); fark kestirimde. NB her özniteliğin ortalamasını ve varyansını <b>ayrı ayrı</b> kestirir: d büyük olsa bile her kestirim n örnekle yapılır (Ng ve Jordan: gereken örnek sayısı d ile logaritmik). LR d + 1 parametreyi <b>birlikte</b> optimize eder ve n ≈ d civarında daha yavaş iner. Çok küçük n'de (birkaç örnek) NB'nin varyans kestirimleri de gürültülüdür; eğriler o bölgede karışabilir. ${q.rho > 0.2 ? `ρ = ${fmt(q.rho, 2)}: tüm öznitelikler ortak bir gürültüyü paylaşıyor. Bilgisiz yarı tek başına sınıfla ilişkisizdir, NB onlara sıfır ağırlık verir; ama ortak gürültüyü ölçtükleri için en iyi sınıflandırıcı onları <b>çıkararak</b> gürültüyü temizler. LR bunu öğrenebilir, NB koşullu bağımsızlık varsayımı yüzünden öğrenemez: çok veride LR açıkça kazanır.` : "ρ = 0'da naif varsayım doğru: NB'nin asimptotik hatası da düşük, LR'nin üstünlüğü yalnız gürültüden."}`,
    };
  },
};
