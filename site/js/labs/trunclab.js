// M5.7 — Kırpma (truncation) hilesi: kalite–çeşitlilik ödünleşimi. Gerçek veri: yarıçapı 2 olan çember üzerinde K Gauss (std σ = 0.15).
// Üretici z ∈ R³, z ~ N(0, I): z₁ modu seçer (açı 2π·Φ(z₁), en yakın mod), z₂ radyal, z₃ teğetsel sapma. Kusursuz üretici (γ = 0)
// veriyle aynı dağılımı verir; γ > 0 iki kusur ekler: modlar arasına sızma (açı tam mod merkezine oturmaz) ve gizil kuyruklarda
// (büyük |z₂|) bozulan yarıçap. Kırpma z' = z̄ + ψ(z − z̄), z̄ = 0. Kesinlik / geri çağırma: k-NN bilyeleriyle (Kynkäänniemi ve ark. 2019).
import { fmt, range } from "./plot.js";
import { stats, frechet } from "./fidis.js";

const R0 = 2, SD = 0.15;
function Phi(x) { const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2), p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x > 0 ? 1 - p : p; }
export function real(r, n, K) {
  return range(n).map(() => { const k = Math.floor(r() * K), a = (2 * Math.PI * k) / K; return [R0 * Math.cos(a) + SD * r.normal(), R0 * Math.sin(a) + SD * r.normal()]; });
}
export function gen(r, n, K, psi, gam) {
  const leak = Math.min(1, 0.25 * gam);
  return range(n).map(() => {
    const z1 = psi * r.normal(), z2 = psi * r.normal(), z3 = psi * r.normal();
    const th = 2 * Math.PI * Phi(z1), k = Math.round((th * K) / (2 * Math.PI)), tk = (2 * Math.PI * k) / K, a = tk + leak * (th - tk);
    const rad = R0 + SD * (z2 + 0.3 * gam * z2 ** 3), tan = SD * z3;
    return [rad * Math.cos(a) - tan * Math.sin(a), rad * Math.sin(a) + tan * Math.cos(a)];
  });
}
const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
function knnRadius(X, k) {
  // her nokta için k. en yakın komşunun uzaklığının karesi (k küçük: sıralı ekleme)
  return X.map((x, i) => { const best = new Array(k).fill(Infinity); for (let j = 0; j < X.length; j++) { if (j === i) continue; const v = d2(x, X[j]); if (v < best[k - 1]) { let p = k - 1; while (p > 0 && best[p - 1] > v) { best[p] = best[p - 1]; p--; } best[p] = v; } } return best[k - 1]; });
}
export function prc(Rl, G, k) {
  const rR = knnRadius(Rl, k), rG = knnRadius(G, k);
  const prec = G.filter((g) => Rl.some((x, i) => d2(g, x) <= rR[i])).length / G.length;
  const rec = Rl.filter((x) => G.some((g, j) => d2(g, x) <= rG[j])).length / Rl.length;
  return { prec, rec };
}
const modeOf = (x, K) => ((Math.round((Math.atan2(x[1], x[0]) * K) / (2 * Math.PI)) % K) + K) % K;

export default {
  params: [
    { id: "psi", label: "Kırpma ψ (1 = kırpma yok)", min: 0.05, max: 1.5, step: 0.05, value: 1 },
    { id: "gam", label: "Üreticinin kusuru γ", min: 0, max: 2, step: 0.1, value: 1 },
    { id: "K", label: "Mod sayısı K", min: 3, max: 16, value: 8, int: true },
    { id: "k", label: "k-NN bilyesindeki komşu sayısı k", min: 1, max: 10, value: 3, int: true },
    { id: "N", label: "Örnek sayısı N", min: 200, max: 1500, value: 600, int: true },
  ],
  seed: true,
  run(q, r) {
    const Rl = real(r, q.N, q.K), G = gen(r, q.N, q.K, q.psi, q.gam), m = prc(Rl, G, q.k), sR = stats(Rl), fid = frechet(sR, stats(G));
    const cnt = new Array(q.K).fill(0); G.forEach((g) => cnt[modeOf(g, q.K)]++);
    const cov = cnt.filter((c) => c >= (0.2 * q.N) / q.K).length;
    const psis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.35, 1.5], sw = psis.map((p) => { const g = gen(r, Math.min(q.N, 400), q.K, p, q.gam), rl = Rl.slice(0, Math.min(q.N, 400)); return prc(rl, g, q.k); });
    const P = [`ψ = ${q.psi}: kesinlik ${fmt(m.prec, 3)} (üretilenlerin gerçek veri yüzeyine düşen oranı: kalite), geri çağırma ${fmt(m.rec, 3)} (gerçek verinin üretilenlerce kapsanan oranı: çeşitlilik). ${q.K} modun ${cov} tanesi kapsanıyor; FID ${fmt(fid, 3)}.`];
    if (q.psi < 0.95) P.push(`Kırpma tüm gizil kodları ortalamaya çeker. Kuyruktaki bozuk örnekler kaybolur${q.gam > 0 ? " (kesinlik artar)" : ""}, ama z₁ de daraldığı için açılar Φ(0)'a, yani tek bir bölgeye yığılır: modlar düşer ve geri çağırma azalır. Bu yapay örnekte z₁ modu doğrudan seçtiği için mod kaybı keskindir; StyleGAN'da kırpma w uzayında yapılır ve "ortalama yüze" doğru bir kayma olarak görünür.`);
    if (q.psi > 1.05) P.push("ψ > 1 kodları ortalamadan uzaklaştırır: çeşitlilik biraz artar ama kuyruk örnekleri kötüleşir.");
    if (q.gam > 0 && q.psi >= 0.95 && q.psi <= 1.05) P.push(`Kusurlu üretici (γ = ${q.gam}): modlar arasına düşen ve yarıçapı bozuk örnekler kesinliği düşürüyor, ama tek Gauss uydurmasına dayanan FID bunu neredeyse görmüyor. Kalite ile çeşitliliği ayrı ölçmek için kesinlik / geri çağırma önerilmesinin nedeni budur.`);
    if (q.gam === 0 && q.psi >= 0.95 && q.psi <= 1.05) P.push("γ = 0 ve ψ = 1: üretici veriyle aynı dağılımda; kesinlik ve geri çağırmanın 1'den küçük olması yalnız sonlu örneklemden (k-NN bilyelerinin boşlukları).");
    const show = (X) => X.slice(0, 500);
    return {
      plots: [
        { title: "Gerçek ve üretilen örnekler", xlabel: "x₁", ylabel: "x₂", size: { w: 320, h: 300 }, series: [
          { name: "gerçek", x: show(Rl).map((p) => p[0]), y: show(Rl).map((p) => p[1]), kind: "scatter", c: 4, r: 1.8 },
          { name: "üretilen", x: show(G).map((p) => p[0]), y: show(G).map((p) => p[1]), kind: "scatter", c: 0, r: 2.2 }] },
        { title: "ψ taraması (aynı üretici, N ≤ 400)", xlabel: "ψ", ylabel: "oran", ylim: [0, 1.02], series: [
          { name: "kesinlik (kalite)", x: psis, y: sw.map((s) => s.prec), c: 0 },
          { name: "geri çağırma (çeşitlilik)", x: psis, y: sw.map((s) => s.rec), c: 1 }], marks: [{ x: q.psi, label: "ψ" }] },
      ],
      stats: [
        { label: "Kesinlik / geri çağırma", value: `${fmt(m.prec, 3)} / ${fmt(m.rec, 3)}` },
        { label: "Kapsanan mod", value: `${cov} / ${q.K}` },
        { label: "FID (2-B)", value: fid },
      ],
      note: P.join(" "),
    };
  },
};
