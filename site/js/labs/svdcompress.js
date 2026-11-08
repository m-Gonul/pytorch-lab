// M2.6 — Kesik SVD = en iyi rank-k yaklaşım (Eckart–Young). 48×48 bir görüntü: rank-k kurma, tekil değer spektrumu, gürültü giderme.
import { fmt, range } from "./plot.js";

const N = 48;
export function jacobiEig(A) { // simetrik A (n×n) → {vals, vecs (sütunlar)}
  const n = A.length, a = A.map((r) => [...r]), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const ord = range(n).sort((i, j) => a[j][j] - a[i][i]);
  return { vals: ord.map((i) => a[i][i]), vecs: ord.map((i) => V.map((r) => r[i])) };
}
export function svd(M) { // M (m×n), n küçük: AᵀA'nın özvektörleri V, σ = √λ, u = Mv/σ
  const m = M.length, n = M[0].length, G = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => { let s = 0; for (let k = 0; k < m; k++) s += M[k][i] * M[k][j]; return s; }));
  const E = jacobiEig(G), S = E.vals.map((l) => Math.sqrt(Math.max(l, 0)));
  const U = E.vecs.map((v, k) => range(m).map((i) => (S[k] > 1e-12 ? M[i].reduce((s, x, j) => s + x * v[j], 0) / S[k] : 0)));
  return { S, U, V: E.vecs };
}
export function image(kind) {
  return range(N).map((i) => range(N).map((j) => {
    const x = j / (N - 1), y = i / (N - 1);
    if (kind === 0) { let v = 0.15 + 0.3 * x; if ((x - 0.35) ** 2 + (y - 0.4) ** 2 < 0.04) v = 0.95; if (x > 0.55 && x < 0.85 && y > 0.55 && y < 0.8) v = 0.75; if (Math.abs(y - 0.85 + 0.5 * x) < 0.025) v = 0.05; return v; }
    if (kind === 1) return 0.5 + 0.25 * Math.sin(12 * x) * Math.cos(7 * y) + 0.2 * Math.sin(5 * y); // düşük ranklı doku
    return Math.min(1, Math.max(0, 0.5 + 0.4 * Math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05) * Math.sign(Math.sin(30 * x * y)))); // keskin, yüksek ranklı desen
  }));
}

export default {
  params: [
    { id: "img", label: "Görüntü", options: ["şekiller (kenarlar)", "pürüzsüz doku (düşük rank)", "ince desen (yüksek rank)"], value: 0 },
    { id: "k", label: "Tutulan bileşen (rank) k", min: 1, max: 48, value: 8, int: true },
    { id: "noise", label: "Eklenen gürültü σ", min: 0, max: 0.3, step: 0.01, value: 0 },
  ],
  seed: true,
  seedOn: { noise: 0.1 },
  run(q, r) {
    const clean = image(q.img), A = clean.map((row) => row.map((v) => v + q.noise * r.normal()));
    const { S, U, V } = svd(A), k = Math.round(q.k);
    const Ak = range(N).map((i) => range(N).map((j) => { let s = 0; for (let t = 0; t < k; t++) s += S[t] * U[t][i] * V[t][j]; return s; }));
    const fro2 = S.reduce((s, x) => s + x * x, 0), tail = S.slice(k).reduce((s, x) => s + x * x, 0);
    const err = (B) => Math.sqrt(B.reduce((s, row, i) => s + row.reduce((t, v, j) => t + (v - clean[i][j]) ** 2, 0), 0) / (N * N));
    const ks = range(N).map((i) => i + 1), curve = ks.map((kk) => Math.sqrt(S.slice(kk).reduce((s, x) => s + x * x, 0) / (N * N)));
    const flat = (B) => ({ w: N, h: N, data: B.flat().map((v) => Math.max(0, Math.min(1, v))) });
    return {
      plots: [
        { title: q.noise ? "Gürültülü görüntü (girdi)" : "Özgün görüntü", img: flat(A), side: true },
        { title: `Rank-${k} yaklaşım (${k * (2 * N + 1)} sayı; özgün ${N * N})`, img: flat(Ak), side: true },
        { title: "Tekil değerler σᵢ (log)", xlabel: "i", ylabel: "σᵢ", ylog: true, series: [{ name: "σᵢ", x: ks, y: S.map((s) => Math.max(s, 1e-9)), c: 0, dots: true }], marks: [{ x: k + 0.5, label: "k" }] },
        { title: "Girdiye göre RMS hata (Eckart–Young: atılan σ'lar)", xlabel: "k", ylabel: "RMS", ylog: true, series: [{ name: "√(Σ_{i>k} σᵢ²)/N", x: ks, y: curve.map((v) => Math.max(v, 1e-9)), c: 2 }], marks: [{ x: k, label: "seçili" }] },
      ],
      stats: [
        { label: "Korunan enerji Σ_{i≤k}σᵢ² / Σσᵢ²", value: `%${fmt(100 * (1 - tail / fro2), 4)}` },
        { label: "Rank-k ile temiz görüntü arasındaki RMS hata", value: err(Ak) },
        { label: "Gürültülü girdi ile temiz görüntü arasındaki RMS hata", value: q.noise ? err(A) : 0 },
        { label: "Sıkıştırma oranı k(2N + 1) / N²", value: (k * (2 * N + 1)) / (N * N) },
      ],
      note: `Rank-k yaklaşım, tüm rank-k matrisler içinde Frobenius normunda en yakın olanıdır ve hatası tam olarak atılan tekil değerlerin karelerinin toplamıdır (Eckart–Young–Mirsky). Pürüzsüz dokuda σ'lar çok hızlı düşer: birkaç bileşen yeter. Keskin kenarlar ve ince desenler birçok bileşen ister. ${q.noise ? "<b>Gürültü</b> tüm tekil değerlere yaklaşık eşit bir taban ekler (spektrumun düz kuyruğu). Küçük k, sinyali tutup gürültünün çoğunu atar: temiz görüntüye olan hata, gürültülü girdininkinden <b>küçük</b> olabilir. k'yı büyütünce gürültü geri gelir — PCA ile gürültü gidermenin özü." : "Gürültü ekleyip küçük k'nın gürültü giderici etkisini dene."}`,
    };
  },
};
