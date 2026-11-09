// M5.7 — FID ile IS'i yan yana: sentetik "öznitelik uzayı". 10 sınıf, ilk iki boyutta yarıçapı 3 olan bir çember üzerinde Gauss bulutları
// (std 0.5); kalan d − 2 boyut sınıf içi değişim (poz, ışık gibi; gerçek veride N(0, 1)), sınıflandırıcı bunlara bakmaz.
// Sınıflandırıcı = Bayes sonsalı (sınıf merkezlerine uzaklığın softmax'ı, yalnız ilk iki boyut). Üretici: kapsanan sınıf sayısı (mod
// çöküşü), sınıf içi yayılım çarpanı (0 = sınıf başına tek "mükemmel" örnek), merkeze çekilme (bulanık / sınıflar arası örnekler).
// IS = exp E_x KL(p(y|x) ‖ p(y)); FID = ‖μ₁ − μ₂‖² + tr(Σ₁ + Σ₂ − 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2}) (Jacobi öz ayrışımıyla).
import { fmt, range } from "./plot.js";

const C = 10, RAD = 3, S = 0.5;
const ctr = range(C).map((k) => [RAD * Math.cos((2 * Math.PI * k) / C), RAD * Math.sin((2 * Math.PI * k) / C)]);
export function jacobi(A0) {
  const n = A0.length, A = A0.map((r) => r.slice()), V = range(n).map((i) => range(n).map((j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
    }
  }
  return { vals: range(n).map((i) => A[i][i]), V };
}
const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
export function stats(F) {
  const n = F.length, d = F[0].length, mu = range(d).map((j) => F.reduce((s, f) => s + f[j], 0) / n);
  const Sg = range(d).map((i) => range(d).map((j) => F.reduce((s, f) => s + (f[i] - mu[i]) * (f[j] - mu[j]), 0) / (n - 1)));
  return { mu, Sg };
}
export function frechet(a, b) {
  const d = a.mu.length, { vals, V } = jacobi(a.Sg), h = vals.map((v) => Math.sqrt(Math.max(v, 0)));
  const Ah = range(d).map((i) => range(d).map((j) => V[i].reduce((s, v, k) => s + v * h[k] * V[j][k], 0)));
  const M = mm(mm(Ah, b.Sg), Ah), ev = jacobi(M.map((r, i) => r.map((v, j) => (v + M[j][i]) / 2))).vals;
  const tr = (X) => X.reduce((s, r, i) => s + r[i], 0);
  return a.mu.reduce((s, v, i) => s + (v - b.mu[i]) ** 2, 0) + tr(a.Sg) + tr(b.Sg) - 2 * ev.reduce((s, v) => s + Math.sqrt(Math.max(v, 0)), 0);
}
export function post(x) {
  const l = ctr.map((c) => -((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2) / (2 * S * S)), m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), z = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / z);
}
export function inceptionScore(F) {
  const P = F.map(post), py = range(C).map((k) => P.reduce((s, p) => s + p[k], 0) / P.length);
  const kl = P.reduce((s, p) => s + p.reduce((t, v, k) => t + (v > 1e-300 ? v * Math.log(v / py[k]) : 0), 0), 0) / P.length;
  return Math.exp(kl);
}
function sample(r, n, d, cls, spread, pull) {
  return range(n).map(() => { const k = cls[Math.floor(r() * cls.length)], c = ctr[k];
    const x = [(1 - pull) * (c[0] + spread * S * r.normal()), (1 - pull) * (c[1] + spread * S * r.normal())];
    for (let j = 2; j < d; j++) x.push(spread * r.normal()); return x; });
}

export default {
  params: [
    { id: "modes", label: "Üreticinin kapsadığı sınıf sayısı (10'dan)", min: 1, max: 10, value: 10, int: true },
    { id: "spread", label: "Sınıf içi yayılım (gerçek = 1; 0 = sınıf başına tek örnek)", min: 0, max: 2, step: 0.05, value: 1 },
    { id: "pull", label: "Merkeze çekilme (bulanık, sınıflar arası örnekler)", min: 0, max: 0.9, step: 0.05, value: 0 },
    { id: "d", label: "Öznitelik boyutu d (ilk 2'si sınıf bilgisi taşır)", min: 2, max: 32, value: 8, int: true },
    { id: "N", label: "Örnek sayısı N (gerçek ve üretilen)", min: 20, max: 3000, value: 1000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const all = range(C), cls = range(q.modes);
    const real = sample(r, q.N, q.d, all, 1, 0), real2 = sample(r, q.N, q.d, all, 1, 0), gen = sample(r, q.N, q.d, cls, q.spread, q.pull);
    const sr = stats(real), fid = frechet(sr, stats(gen)), fidRR = frechet(sr, stats(real2)), isG = inceptionScore(gen), isR = inceptionScore(real);
    const Ns = [25, 50, 100, 200, 400, 800, 1600].filter((n) => n > q.d + 2);
    const cur = Ns.map((n) => { const a = sample(r, n, q.d, all, 1, 0), b = sample(r, n, q.d, cls, q.spread, q.pull), b2 = sample(r, n, q.d, all, 1, 0); return { fid: frechet(stats(a), stats(b)), rr: frechet(stats(a), stats(b2)) }; });
    const P = [`Gerçek veride IS = ${fmt(isR, 3)}, üretilende ${fmt(isG, 3)} (en çok 10). FID(gerçek, üretilen) = ${fmt(fid, 3)}; aynı N ile iki bağımsız gerçek örneklem arasındaki FID = ${fmt(fidRR, 3)} (örneklem gürültüsünün tabanı).`];
    if (q.spread < 0.3 && q.modes === C && q.pull < 0.1) P.push(`Sınıf başına neredeyse tek örnek: her örnek tek sınıfa çok güvenle atanıyor ve sınıflar dengeli, bu yüzden IS gerçek veriden <b>yüksek</b>. IS gerçek veriye hiç bakmadığı için sınıf içi çeşitliliğin kaybını görmez. ` + (q.d > 2 ? `FID görür: sınıf içi değişimin yaşadığı ${q.d - 2} boyutta üretilen kovaryans neredeyse sıfır, iz terimi büyüyor.` : `d = 2'de FID de neredeyse görmüyor: kovaryansın çoğu sınıflar arası yayılımdan (çember) geliyor ve tek Gauss uydurması sınıf içi yayılımın kaybını küçük bir fark olarak algılıyor. Gauss varsayımının sınırı budur; gerçek Inception öznitelikleri yüksek boyutlu olduğu için sınıf içi değişim kendi boyutlarında görünür.`));
    if (q.modes < C) P.push(`${q.modes} sınıf kapsanıyor (mod çöküşü): p(y) dengesiz kaldığı için IS düşer ve ortalama vektörü kaydığı için FID artar.`);
    if (q.pull > 0.2) P.push("Örnekler merkeze (sınıflar arasına) çekildi: sınıflandırıcı kararsız kaldığı için p(y|x) yüksek entropili, IS düşüyor; FID hem ortalama hem kovaryanstan cezalandırıyor.");
    if (q.d > 2) P.push(`d = ${q.d}: sınıf bilgisi taşımayan ${q.d - 2} boyut her iki kümede aynı dağılımda olsa da kovaryans kestirim hatası FID'e girer; N = ${Ns[0]}'te gerçek–gerçek FID ${fmt(cur[0].rr, 3)}, N = ${Ns.at(-1)}'da ${fmt(cur.at(-1).rr, 3)}. FID yanlı bir kestiricidir (yanlılık ≈ d²/N ölçeğinde); 2048 boyutlu Inception özniteliklerinde bu yüzden N ≥ 10 000 ile hesaplanır ve farklı N'lerle hesaplanmış FID'ler karşılaştırılmaz.`);
    const show = (F) => F.slice(0, 600);
    return {
      plots: [
        { title: "Öznitelik uzayının ilk iki boyutu", xlabel: "f₁", ylabel: "f₂", size: { w: 320, h: 300 }, series: [
          { name: "gerçek", x: show(real).map((f) => f[0]), y: show(real).map((f) => f[1]), kind: "scatter", c: 4, r: 1.8 },
          { name: "üretilen", x: show(gen).map((f) => f[0]), y: show(gen).map((f) => f[1]), kind: "scatter", c: 0, r: 2.2 }] },
        { title: "FID'in örnek sayısına bağımlılığı", xlabel: "N (log)", ylabel: "FID", series: [
          { name: "FID(gerçek, üretilen)", x: Ns.map(Math.log10), y: cur.map((c) => c.fid), c: 0 },
          { name: "FID(gerçek, gerçek)", x: Ns.map(Math.log10), y: cur.map((c) => c.rr), c: 4, dash: true }], xticks: Ns.map((n) => [Math.log10(n), String(n)]) },
      ],
      stats: [
        { label: "IS: gerçek / üretilen", value: `${fmt(isR, 3)} / ${fmt(isG, 3)}` },
        { label: "FID(gerçek, üretilen)", value: fid },
        { label: "FID(gerçek, gerçek) aynı N'de", value: fidRR },
      ],
      note: P.join(" "),
    };
  },
};
