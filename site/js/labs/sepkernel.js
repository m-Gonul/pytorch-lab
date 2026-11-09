// M4.8 — Uzamsal ayrılabilirlik: K×K bir çekirdek, iki 1-B çekirdeğin dış çarpımıysa (rank 1) K² yerine 2K çarpımla uygulanır.
// Tekil değer ayrışımı (SVD) her çekirdeği rank-1 terimlerin toplamına ayırır: K = Σ σᵢ uᵢ vᵢᵀ. Rank-r yaklaşıklık r·2K çarpım ister.
import { fmt, range } from "./plot.js";

// simetrik matrisin özdeğer/özvektörleri (Jacobi dönüşleri); küçük K için yeterli
export function eigSym(A0) {
  const n = A0.length, A = A0.map((r) => [...r]), V = range(n).map((i) => range(n).map((j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { w: range(n).map((i) => A[i][i]), V };
}
export function svd(M) {
  const n = M.length, AtA = range(n).map((i) => range(n).map((j) => M.reduce((s, r) => s + r[i] * r[j], 0)));
  const { w, V } = eigSym(AtA), ord = range(n).sort((a, b) => w[b] - w[a]);
  const S = ord.map((i) => Math.sqrt(Math.max(0, w[i]))), Vs = ord.map((i) => V.map((r) => r[i]));
  const U = Vs.map((v, k) => M.map((r) => (S[k] > 1e-12 ? r.reduce((s, x, j) => s + x * v[j], 0) / S[k] : 0)));
  return { S, U, V: Vs };
}
const binom = (n) => { let r = [1]; for (let i = 0; i < n - 1; i++) r = [...r, 0].map((v, j) => v + (j ? r[j - 1] : 0)); return r; };
export function kernel(type, K, sig, r) {
  const c = (K - 1) / 2, xs = range(K).map((i) => i - c), g = xs.map((x) => Math.exp(-(x * x) / (2 * sig * sig)));
  const outer = (a, b) => a.map((ai) => b.map((bj) => ai * bj));
  switch (type) {
    case 0: return outer(g, g);
    case 1: return outer(range(K).map(() => 1), range(K).map(() => 1));
    case 2: { const sm = binom(K), d = xs.map((x, i) => x * sm[i]); return outer(sm, d); } // Sobel benzeri: düzleştir ⊗ türev
    case 3: return xs.map((y) => xs.map((x) => ((x * x + y * y - 2 * sig * sig) / sig ** 4) * Math.exp(-(x * x + y * y) / (2 * sig * sig)))); // LoG
    case 4: return xs.map((y) => xs.map((x) => (x * x + y * y <= (c + 0.5) ** 2 ? 1 : 0))); // disk
    default: return range(K).map(() => range(K).map(() => r.normal()));
  }
}
const NAMES = ["Gauss", "kutu (ortalama)", "Sobel benzeri (türev)", "Gauss'un Laplace'ı (LoG)", "disk", "rastgele (öğrenilmiş bir filtre gibi)"];
const toImg = (M) => { const f = M.flat(), lo = Math.min(...f), hi = Math.max(...f); return f.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5)); };

export default {
  params: [
    { id: "type", label: "Çekirdek", options: NAMES, value: 3 },
    { id: "K", label: "Çekirdek boyu K", min: 3, max: 11, value: 7, int: true },
    { id: "sig", label: "σ (Gauss ve LoG için)", min: 0.4, max: 4, step: 0.05, value: 1.2 },
    { id: "rk", label: "Yaklaşıklık rankı r", min: 1, max: 11, value: 1, int: true },
  ],
  seed: true,
  seedOn: { type: 5 },
  validate(q) { return q.rk > q.K ? [{ level: "warn", msg: `r = ${q.rk} > K = ${q.K}: K×K bir matrisin rankı en çok K'dır; r = K kullanılıyor.` }] : []; },
  run(q, r) {
    const K = Math.round(q.K), rk = Math.min(Math.round(q.rk), K), M = kernel(q.type, K, q.sig, r), D = svd(M);
    const A = range(K).map((i) => range(K).map((j) => range(rk).reduce((s, k) => s + D.S[k] * D.U[k][i] * D.V[k][j], 0)));
    const nrm = Math.sqrt(M.flat().reduce((s, v) => s + v * v, 0)), err = Math.sqrt(M.flat().reduce((s, v, k) => s + (v - A.flat()[k]) ** 2, 0)) / nrm;
    const tot = D.S.reduce((s, v) => s + v * v, 0), rank = D.S.filter((s) => s > 1e-6 * D.S[0]).length;
    return {
      plots: [
        { title: "Tekil değerler σᵢ (log; 10⁻⁷·σ₁ altı sayısal sıfır)", xlabel: "i", ylabel: "σᵢ", ylog: true, xticks: range(K).map((i) => [i + 1, String(i + 1)]), ylim: [D.S[0] * 1e-8, D.S[0] * 3],
          series: [{ name: "tutulan (ilk r)", x: range(rk).map((i) => i + 1), y: D.S.slice(0, rk).map((s) => Math.max(s, D.S[0] * 3e-8)), kind: "bar", c: 0 }, { name: "atılan", x: range(K - rk).map((i) => i + rk + 1), y: D.S.slice(rk).map((s) => Math.max(s, D.S[0] * 3e-8)), kind: "bar", c: 4 }] },
        { title: `${NAMES[q.type]} (${K}×${K})`, img: { w: K, h: K, data: toImg(M) }, side: true, size: { w: 310, h: 300 } },
        { title: `rank-${rk} yaklaşıklık`, img: { w: K, h: K, data: toImg(A) }, side: true, size: { w: 310, h: 300 } },
      ],
      stats: [
        { label: "Sayısal rank (σᵢ > 10⁻⁶·σ₁)", value: rank },
        { label: "Göreli hata ‖K − K_r‖_F / ‖K‖_F", value: err < 1e-6 ? "≈ 0 (tam; kalan, SVD'nin sayısal hassasiyeti)" : err },
        { label: "Tutulan enerji Σᵢ≤r σᵢ² / Σ σᵢ²", value: D.S.slice(0, rk).reduce((s, v) => s + v * v, 0) / tot },
        { label: "Piksel başına çarpma: tam / rank-r ayrılabilir", value: `${K * K} / ${2 * K * rk}` },
      ],
      note: `Her matris SVD ile rank-1 dış çarpımların toplamına ayrılır: K = Σ σᵢ uᵢvᵢᵀ (Eckart–Young: ilk r terim, Frobenius normunda en iyi rank-r yaklaşıklıktır). Rank-1 terim, önce dikey uᵢ sonra yatay vᵢ ile iki 1-B konvolüsyon demektir: K² yerine 2K çarpım. ${rank === 1 ? "<b>Bu çekirdek tam ayrılabilir</b> (tek sıfır olmayan tekil değer): Gauss, kutu ve Sobel gibi çekirdekler iki 1-B çekirdeğin dış çarpımıdır — slayttaki [1,2,1] örneği gibi." : rank === 2 && q.type === 3 ? "<b>LoG rank 2:</b> ∂²g/∂x² ⊗ g + g ⊗ ∂²g/∂y² — iki ayrılabilir terimin toplamı; iki çift 1-B konvolüsyonla tam uygulanır." : `<b>Bu çekirdek ayrılabilir değil</b> (rank ${rank}); ${err < 0.05 ? "ama ilk birkaç terim onu iyi yaklaşıklıyor." : "az terimle yaklaşıklık kaba kalıyor."}`} Derin ağlarda "ayrılabilir konvolüsyon" çoğunlukla bunun kanal boyutundaki karşılığıdır (derinlemesine + 1×1); Inception-v3'ün n×n → 1×n + n×1 çarpanlaması ise bu uzamsal türdendir.`,
    };
  },
};
