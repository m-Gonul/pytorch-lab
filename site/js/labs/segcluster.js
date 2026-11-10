// M9.11 — Kümelemeyle bölütleme. 64 × 64 renkli sahne: mavi gökyüzü gradyanı, kırmızı disk, yeşil kare, iki tonlu (gölgeli)
// sarı üçgen; Gauss gürültüsü. Yöntemler: RGB'de k-means (Lloyd, rastgele örneklerden başlatma), RGB + λ·(x, y) ile k-means ve
// tam kovaryanslı GMM (EM, k-means'ten başlatılır). Ölçütler: saflık (her küme çoğunluk sınıfına), amaç fonksiyonu, iterasyon.
import { fmt, range } from "./plot.js";

const W = 64;
export function scene(r, sn) {
  const z = [], gt = [];
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let c = [0.25 + 0.3 * (y / W), 0.45 + 0.2 * (y / W), 0.85], g = 0;
    if ((x - 18) ** 2 + (y - 20) ** 2 <= 121) { c = [0.8, 0.15, 0.15]; g = 1; }
    if (x >= 36 && x < 56 && y >= 8 && y < 28) { c = [0.15, 0.6, 0.2]; g = 2; }
    if (y >= 36 && y < 60 && Math.abs(x - 32) <= (y - 36) * 0.9) { const sh = x < 32 ? 1 : 0.7; c = [0.9 * sh, 0.8 * sh, 0.1]; g = 3; }
    z.push(c.map((v) => Math.max(0, Math.min(1, v + sn * r.normal())))); gt.push(g);
  }
  return { z, gt };
}
const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
export function kmeans(X, k, r, maxIt = 30, pp = false) {
  const n = X.length, used = new Set(); let m = [];
  if (pp) { // k-means++: ilk merkez rastgele, sonrakiler en yakın merkeze uzaklığın karesiyle orantılı olasılıkla
    m.push(X[Math.floor(r() * n)].slice());
    while (m.length < k) { const D = X.map((x) => Math.min(...m.map((c) => d2(x, c)))), tot = D.reduce((a, b) => a + b, 0); let u = r() * tot, i = 0; while (i < n - 1 && (u -= D[i]) > 0) i++; m.push(X[i].slice()); }
  } else while (m.length < k) { const i = Math.floor(r() * n); if (!used.has(i)) { used.add(i); m.push(X[i].slice()); } }
  let lab = new Int32Array(n), hist = [];
  for (let it = 0; it < maxIt; it++) {
    let J = 0; for (let i = 0; i < n; i++) { let b = 0, bd = Infinity; for (let j = 0; j < k; j++) { const d = d2(X[i], m[j]); if (d < bd) { bd = d; b = j; } } lab[i] = b; J += bd; }
    hist.push(J);
    const nm = range(k).map(() => new Array(X[0].length).fill(0)), cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) { cnt[lab[i]]++; X[i].forEach((v, d) => (nm[lab[i]][d] += v)); }
    const E = nm.reduce((s, v, j) => { const c = cnt[j] ? v.map((t) => t / cnt[j]) : m[j]; const e = Math.sqrt(d2(c, m[j])); m[j] = c; return s + e; }, 0);
    if (E <= 1e-9) break;
  }
  return { lab, m, hist };
}
const inv3 = (A) => { const [[a, b, c], [d, e, f], [g, h, i]] = A, det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g); return { det, inv: [[e * i - f * h, c * h - b * i, b * f - c * e], [f * g - d * i, a * i - c * g, c * d - a * f], [d * h - e * g, b * g - a * h, a * e - b * d]].map((row) => row.map((v) => v / det)) }; };
export function gmm(X, k, init, maxIt = 40) {
  const n = X.length, D = 3; let mu = init.m.map((v) => v.slice(0, 3)), pi = new Array(k).fill(1 / k);
  let S = range(k).map(() => [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]), R = range(n).map(() => new Float64Array(k)), hist = [];
  for (let it = 0; it < maxIt; it++) {
    const P = S.map((s) => { const s2 = s.map((row, a) => row.map((v, b) => v + (a === b ? 1e-4 : 0))); return inv3(s2); });
    let ll = 0;
    for (let i = 0; i < n; i++) { let tot = 0; for (let j = 0; j < k; j++) { const dx = X[i].slice(0, 3).map((v, a) => v - mu[j][a]); let q = 0; for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) q += dx[a] * P[j].inv[a][b] * dx[b]; const p = pi[j] * Math.exp(-0.5 * q) / Math.sqrt((2 * Math.PI) ** D * Math.max(P[j].det, 1e-300)); R[i][j] = p; tot += p; } for (let j = 0; j < k; j++) R[i][j] /= tot || 1; ll += Math.log(tot || 1e-300); }
    hist.push(-ll);
    for (let j = 0; j < k; j++) {
      let Nk = 0; const m = [0, 0, 0]; for (let i = 0; i < n; i++) { Nk += R[i][j]; for (let a = 0; a < D; a++) m[a] += R[i][j] * X[i][a]; }
      mu[j] = m.map((v) => v / Nk); pi[j] = Nk / n;
      const s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (let i = 0; i < n; i++) { const dx = X[i].slice(0, 3).map((v, a) => v - mu[j][a]); for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) s[a][b] += R[i][j] * dx[a] * dx[b]; }
      S[j] = s.map((row) => row.map((v) => v / Nk));
    }
    if (it > 2 && Math.abs(hist[it] - hist[it - 1]) < 1e-6 * Math.abs(hist[it])) break;
  }
  return { lab: Int32Array.from(R.map((row) => row.indexOf(Math.max(...row)))), m: mu, hist };
}
const METHODS = ["k-means, RGB", "k-means, RGB + λ·(x, y)", "GMM (tam kovaryans, EM)"];
export default {
  params: [
    { id: "meth", label: "Yöntem", options: METHODS, value: 0 },
    { id: "k", label: "Küme sayısı k", min: 2, max: 7, value: 4, int: true },
    { id: "init", label: "Başlatma", options: ["rastgele k örnek", "k-means++"], value: 0 },
    { id: "lam", label: "Konum ağırlığı λ", min: 0, max: 1.5, step: 0.05, value: 0.5, on: { meth: [1] } },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.15, step: 0.005, value: 0.04 },
  ],
  seed: true,
  run(q, r) {
    const { z, gt } = scene(r, q.noise), X = q.meth === 1 ? z.map((c, i) => [...c, (q.lam * (i % W)) / W, (q.lam * ((i / W) | 0)) / W]) : z;
    const km = kmeans(X, q.k, r, 30, q.init === 1), res = q.meth === 2 ? gmm(z, q.k, km) : km;
    const pur = (() => { let s = 0; for (let j = 0; j < q.k; j++) { const c = [0, 0, 0, 0]; res.lab.forEach((l, i) => { if (l === j) c[gt[i]]++; }); s += Math.max(...c); } return s / gt.length; })();
    const meanCol = range(q.k).map((j) => { const c = [0, 0, 0]; let n = 0; res.lab.forEach((l, i) => { if (l === j) { n++; z[i].forEach((v, a) => (c[a] += v)); } }); return n ? c.map((v) => v / n) : [0, 0, 0]; });
    const seg = res.lab.reduce((a, l) => { a.push(...meanCol[l]); return a; }, []);
    const yellowSplit = (() => { const s = new Set(); res.lab.forEach((l, i) => { if (gt[i] === 3) s.add(l); }); return s.size; })();
    const P = [`Saflık ${fmt(100 * pur, 3)}% (her küme çoğunluk sınıfına sayıldı; sahnede 4 bölge). ${res.hist.length} iterasyon.`];
    P.push(q.meth === 0 ? `k-means yalnız renge bakar; gölgeli sarı üçgen ${yellowSplit} kümeye bölündü${yellowSplit > 1 ? ": iki ton, k-means için iki ayrı küre" : ""}. Gökyüzü gradyanı da k büyüdükçe bantlara ayrılır. Tohumu değiştir: sonuç başlatmaya bağlıdır.${pur < 0.85 && q.init === 0 ? " <b>Kötü yerel minimum:</b> rastgele başlangıç örneklerinin çoğu, piksellerin %69'unu kaplayan gökyüzünden seçildi; gökyüzü birkaç kümeye bölünürken nesneler tek kümede kaldı. k-means++ başlatma bunu büyük olasılıkla önler." : ""}` : q.meth === 1 ? `Konum eklenince kümeler uzayda da toplu olmak zorunda: λ büyüdükçe bölgeler renkten çok konuma göre (ızgaraya benzer) bölünür; bu SLIC'in fikridir (l3).` : `GMM her kümeye bir kovaryans verir: uzamış (gradyanlı gökyüzü, gölgeli üçgen) dağılımlar tek bileşende toplanabilir; atama yumuşaktır (gösterimde en olası bileşen). Gölgeli üçgen ${yellowSplit} bileşende.`);
    const lbl = q.meth === 2 ? "Negatif log-olabilirlik" : "Amaç J = Σ‖z − m‖²";
    return {
      plots: [
        { title: "Girdi", img: { w: W, h: W, rgb: true, data: z.flat() }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `${METHODS[q.meth]}, k = ${q.k} (küme ortalama renkleri)`, img: { w: W, h: W, rgb: true, data: seg }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: lbl, xlabel: "iterasyon", ylabel: lbl, wide: true, size: { w: 620, h: 150 }, series: [{ name: lbl, x: range(res.hist.length), y: res.hist, c: 0 }] },
      ],
      stats: [{ label: "Saflık", value: `${fmt(100 * pur, 3)}%` }, { label: "Sarı üçgeni bölen küme sayısı", value: yellowSplit }, { label: "İterasyon", value: res.hist.length }, { label: lbl + " (son)", value: res.hist[res.hist.length - 1] }],
      note: P.join(" "),
    };
  },
};
