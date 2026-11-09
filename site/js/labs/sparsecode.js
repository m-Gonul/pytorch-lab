// M5.6 — Seyreklik cezası (FISTA ile): aşırı tam (overcomplete) bir sözlükle kod bulma. Sinyal x (64 örnek) = D z_gerçek + gürültü; D: 128 atom
// (64 konumda iki genişlikte Gauss tümseği, birim norm). L1: min ½‖x − Dz‖² + λ‖z‖₁ (ISTA, yumuşak eşikleme);
// L2: min ½‖x − Dz‖² + λ‖z‖² (kapalı form, (DᵀD + 2λI) z = Dᵀx). Sıfırdan farklı kod sayısı, destek kurtarma, hata.
import { fmt, range } from "./plot.js";

const N = 64, K = 128;
export function dict() {
  return range(K).map((k) => { const c = k % N, w = k < N ? 1 : 3, a = range(N).map((i) => Math.exp(-0.5 * ((i - c) / w) ** 2)), n = Math.hypot(...a); return a.map((v) => v / n); });
}
const Dz = (D, z) => range(N).map((i) => D.reduce((s, a, k) => s + a[i] * z[k], 0));
const Dt = (D, r) => D.map((a) => a.reduce((s, v, i) => s + v * r[i], 0));
export function ista(D, x, lam, iters) {
  // Lipschitz sabiti: ‖DᵀD‖ (kuvvet iterasyonu)
  let v = range(K).map(() => 1), L = 1;
  for (let t = 0; t < 50; t++) { const w = Dt(D, Dz(D, v)); L = Math.hypot(...w); v = w.map((u) => u / L); }
  // FISTA: ISTA + Nesterov ivmesi (Beck & Teboulle 2009)
  let z = new Array(K).fill(0), y = z.slice(), tk = 1;
  for (let t = 0; t < iters; t++) {
    const g = Dt(D, Dz(D, y).map((u, i) => u - x[i])), zn = y.map((u, k) => { const v = u - g[k] / L, th = lam / L; return Math.sign(v) * Math.max(0, Math.abs(v) - th); });
    const tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2; y = zn.map((u, k) => u + ((tk - 1) / tn) * (u - z[k])); z = zn; tk = tn;
  }
  return z;
}
export function ridge(D, x, lam) {
  const A = range(K).map((i) => range(K).map((j) => D[i].reduce((s, v, t) => s + v * D[j][t], 0) + (i === j ? 2 * lam : 0))), b = Dt(D, x), M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < K; i++) { let p = i; for (let k = i + 1; k < K; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k; [M[i], M[p]] = [M[p], M[i]]; for (let k = i + 1; k < K; k++) { const f = M[k][i] / M[i][i]; if (f) for (let j = i; j <= K; j++) M[k][j] -= f * M[i][j]; } }
  const z = new Array(K).fill(0); for (let i = K - 1; i >= 0; i--) { let s = M[i][K]; for (let j = i + 1; j < K; j++) s -= M[i][j] * z[j]; z[i] = s / M[i][i]; } return z;
}

export default {
  params: [
    { id: "k", label: "Gerçek koddaki etkin atom sayısı", min: 1, max: 8, value: 4, int: true },
    { id: "gap", label: "Gerçek atomlar arası en küçük uzaklık (örnek)", min: 0, max: 12, value: 8, int: true },
    { id: "lam", label: "Ceza ağırlığı λ", min: 0.001, max: 1, value: 0.05, log: true },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.3, step: 0.01, value: 0.05 },
    { id: "pen", label: "Gösterilen ceza", options: ["L1 (yumuşak eşikleme, FISTA)", "L2 (ridge)"], value: 0 },
    { id: "iters", label: "FISTA adımı", min: 10, max: 3000, value: 800, int: true, log: true, on: { pen: [0] } },
  ],
  seed: true,
  run(q, r) {
    const D = dict(), zt = new Array(K).fill(0), supp = [];
    let guard = 0;
    while (supp.length < q.k && guard++ < 5000) { const k = Math.floor(r() * K); if (supp.every((s) => Math.abs((s % N) - (k % N)) >= q.gap)) { supp.push(k); zt[k] = (r() < 0.5 ? -1 : 1) * (0.6 + r()); } }
    const clean = Dz(D, zt), x = clean.map((v) => v + q.noise * r.normal());
    const z1 = ista(D, x, q.lam, q.pen === 0 ? q.iters : 800), z2 = ridge(D, x, q.lam), z = q.pen ? z2 : z1;
    const nz = (v) => v.filter((u) => Math.abs(u) > 1e-3).length, rec = Dz(D, z), mse = rec.reduce((s, v, i) => s + (v - clean[i]) ** 2, 0) / N;
    const top = range(K).sort((a, b) => Math.abs(z[b]) - Math.abs(z[a])).slice(0, q.k), hit = top.filter((t) => supp.includes(t) || supp.some((s) => Math.abs((s % N) - (t % N)) <= 1 && Math.floor(s / N) === Math.floor(t / N))).length;
    const e1 = Dz(D, z1).reduce((s, v, i) => s + (v - clean[i]) ** 2, 0) / N, e2 = Dz(D, z2).reduce((s, v, i) => s + (v - clean[i]) ** 2, 0) / N;
    const P = [`Sözlük 128 atom, sinyal 64 örnek: kod sinyalden büyük (aşırı tam), problem tek çözümlü değil; ceza hangi çözümün seçileceğini belirler.`];
    P.push(`${q.pen ? "L2" : "L1"} cezasıyla (λ = ${fmt(q.lam, 3)}) 128 katsayının ${nz(z)} tanesi sıfırdan farklı (gerçek kodda ${q.k}). En büyük ${q.k} katsayının ${hit} tanesi gerçek atomların (±1 konum) üzerinde. Temiz sinyale göre hata: L1 ${fmt(e1, 3)}, L2 ${fmt(e2, 3)}; L1 kodunda ${nz(z1)}, L2 kodunda ${nz(z2)} etkin katsayı.`);
    if (q.gap < 3) P.push("Gerçek atomlar birbirine çok yakın olabilir: komşu atomlar neredeyse aynı biçimde olduğu için (yüksek tutarlılık) sözlük onları ayırt edemez; L1 farklı ama eşit derecede iyi açıklayan bir seyrek kod bulabilir. Seyrek kurtarma kuramındaki tutarlılık koşulunun anlamı budur.");
    P.push(q.pen ? "L2 cezası tüm katsayıları orantılı küçültür ama sıfırlamaz: enerji birbirine benzeyen birçok atoma yayılır. Kod yoğun kalır." : "L1'in yakınsal adımı yumuşak eşiklemedir: |y| < λ/L olan katsayılar tam sıfır olur. Az sayıda atomla açıklamak 'ucuz' olduğu için kod seyrek çıkar; λ büyüdükçe daha çok katsayı sıfırlanır, ama gerçek katsayılar da λ kadar küçülür (yanlılık).");
    const ks = range(K);
    return {
      plots: [
        { title: "Kodlar (128 atom; 0–63 dar, 64–127 geniş tümsekler)", xlabel: "atom", ylabel: "katsayı", wide: true, size: { w: 620, h: 220 }, series: [
          { name: "gerçek kod", x: ks, y: zt, kind: "bar", c: 4 },
          { name: q.pen ? "L2 kodu" : "L1 kodu", x: ks, y: z, kind: "bar", c: q.pen ? 2 : 0, bw: 0.5 }] },
        { title: "Sinyal ve yeniden yapılandırma", xlabel: "örnek", ylabel: "değer", series: [
          { name: "gürültülü x", x: range(N), y: x, kind: "scatter", c: 4, r: 2 },
          { name: "temiz", x: range(N), y: clean, c: 1, dash: true },
          { name: "Dz", x: range(N), y: rec, c: q.pen ? 2 : 0 }] },
      ],
      stats: [
        { label: "Sıfırdan farklı katsayı: L1 / L2", value: `${nz(z1)} / ${nz(z2)}` },
        { label: "Temiz sinyale hata: L1 / L2", value: `${fmt(e1, 3)} / ${fmt(e2, 3)}` },
        { label: `En büyük ${q.k} katsayıdan gerçek desteğe düşen`, value: hit },
        { label: "Gösterilen kodun hatası", value: mse },
      ],
      note: P.join(" "),
    };
  },
};
