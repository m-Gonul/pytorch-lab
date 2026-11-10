// M6.3 — DP-SGD (Abadi ve ark. 2016) 2-B lojistik regresyonda: örnek başı gradyan gᵢ = (pᵢ − yᵢ)[xᵢ, 1], ‖gᵢ‖'yi C'ye kırp,
// topla, N(0, σ²C²I) gürültü ekle, B'ye böl. Karşılaştırma: aynı tohumla sıradan SGD (ortalama gradyan). Veride bir oranda etiketi
// ters, uzaktaki aykırı örnekler: büyük gradyanlı örneklerin etkisini kırpmanın nasıl sınırladığı görülür. Test kümesi temiz.
import { fmt, range } from "./plot.js";

const sig = (t) => 1 / (1 + Math.exp(-t));
function data(r, n, out) {
  return range(n).map((i) => { const y = i % 2, o = r() < out; const c = o ? [y ? -6 : 6, y ? -6 : 6] : [y ? 1 : -1, y ? 1 : -1]; return { x: [c[0] + 0.9 * r.normal(), c[1] + 0.9 * r.normal()], y }; });
}
const acc = (w, D) => D.filter((d) => (sig(w[0] * d.x[0] + w[1] * d.x[1] + w[2]) > 0.5 ? 1 : 0) === d.y).length / D.length;
export function train(q, r, dp, D, T) {
  const w = [0, 0, 0], hist = [], clipped = [];
  for (let t = 0; t < q.steps; t++) {
    const idx = range(q.B).map(() => Math.floor(r() * D.length)), g = [0, 0, 0];
    let nclip = 0;
    for (const i of idx) {
      const d = D[i], p = sig(w[0] * d.x[0] + w[1] * d.x[1] + w[2]), gi = [(p - d.y) * d.x[0], (p - d.y) * d.x[1], p - d.y], nrm = Math.hypot(...gi);
      const s = dp ? Math.min(1, q.C / Math.max(nrm, 1e-12)) : 1; if (dp && nrm > q.C) nclip++;
      gi.forEach((v, k) => (g[k] += s * v));
    }
    for (let k = 0; k < 3; k++) { if (dp) g[k] += q.sigma * q.C * r.normal(); w[k] -= (q.lr * g[k]) / q.B; }
    hist.push(acc(w, T)); clipped.push(nclip / q.B);
  }
  return { w, hist, clipped };
}
export default {
  params: [
    { id: "C", label: "Kırpma eşiği C", min: 0.01, max: 10, value: 1, log: true },
    { id: "sigma", label: "Gürültü çarpanı σ", min: 0, max: 4, step: 0.1, value: 1 },
    { id: "B", label: "Batch B", min: 4, max: 256, value: 64, int: true, log: true },
    { id: "lr", label: "Öğrenme oranı", min: 0.01, max: 2, value: 0.5, log: true },
    { id: "steps", label: "Adım", min: 20, max: 1000, value: 300, int: true, log: true },
    { id: "out", label: "Aykırı (ters etiketli, uzak) örnek oranı", min: 0, max: 0.3, step: 0.01, value: 0.05 },
  ],
  seed: true,
  run(q, r) {
    const D = data(r, 400, q.out), T = data(r, 2000, 0), seedState = r();
    const mk = (s) => { let x = Math.floor(s * 2 ** 31) || 1; const f = () => { x = (x * 1103515245 + 12345) % 2 ** 31; return x / 2 ** 31; }; f.normal = () => { const u = Math.max(f(), 1e-12), v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; return f; };
    const S = train(q, mk(seedState), false, D, T), P_ = train(q, mk(seedState), true, D, T);
    const w = P_.w, norms = D.map((d) => { const p = sig(w[0] * d.x[0] + w[1] * d.x[1] + w[2]); return Math.abs(p - d.y) * Math.hypot(d.x[0], d.x[1], 1); });
    const fracClip = P_.clipped.slice(-50).reduce((s, v) => s + v, 0) / Math.min(50, P_.clipped.length);
    const noiseRel = (q.sigma * q.C) / q.B, sd = (a) => { const z = a.slice(-50), m = z.reduce((s, v) => s + v, 0) / z.length; return Math.sqrt(z.reduce((s, v) => s + (v - m) ** 2, 0) / z.length); }, jS = sd(S.hist), jP = sd(P_.hist);
    const nb = 30, mx = Math.max(...norms, q.C) * 1.02, bw = mx / nb, h = new Array(nb).fill(0); norms.forEach((v) => h[Math.min(nb - 1, Math.floor(v / bw))]++);
    const P = [`Son test doğruluğu: SGD ${fmt(S.hist.at(-1), 3)}, DP-SGD ${fmt(P_.hist.at(-1), 3)}. Son 50 adımda kırpılan örnek oranı ${fmt(fracClip, 3)}. Ortalama gradyana eklenen gürültünün std'si σC/B = ${fmt(noiseRel, 3)}; son 50 adımda doğruluğun dalgalanması (std) SGD ${fmt(jS, 2)}, DP-SGD ${fmt(jP, 2)}.`];
    P.push("Kırpma her örneğin güncellemeye katkısını en fazla C ile sınırlar (duyarlılık); ancak bu sınır varken eklenen N(0, σ²C²) gürültü bir gizlilik güvencesine çevrilebilir. Örnek başı gradyan bu yüzden gereklidir: batch'in ortalama gradyanı tek tek katkıları göstermez.");
    if (q.out > 0) P.push(`Aykırı örnekler (oran ${q.out}) yanlış tarafta ve uzakta: gradyan normları büyüktür. Sıradan SGD'de güncellemeyi orantısız çekerler, DP-SGD'de katkıları C ile sınırlıdır.`);
    if (P_.hist.at(-1) >= S.hist.at(-1) - 0.005 && q.sigma > 0) P.push(`Bu kolay 2-B problemde DP-SGD'nin doğruluk bedeli görünmüyor; ${jP < jS ? "üstelik DP-SGD daha az dalgalanıyor, çünkü SGD'yi sarsan aykırı örneklerin büyük gradyanları kırpılıyor" : "gürültü yalnız dalgalanmayı artırıyor"}. Gerçek ağlarda gürültü milyonlarca koordinatın her birine eklendiği için bedel çok daha büyüktür.`);
    if (q.sigma > 0 && noiseRel > 0.1) P.push("Gürültü sinyale göre büyük: B'yi artırmak gürültünün ortalamadaki payını 1/B ile azaltır. DP eğitiminin büyük batch'leri sevmesinin nedeni budur.");
    if (q.C < 0.05) P.push("C çok küçük: neredeyse her gradyan kırpılıyor, güncelleme yönü korunsa da büyüklüğü C'ye iniyor; öğrenme yavaşlar ve gürültü baskın olur.");
    const ts = range(q.steps);
    return {
      plots: [
        { title: "Test doğruluğu", xlabel: "adım", ylabel: "doğruluk", ylim: [0.4, 1], series: [{ name: "SGD", x: ts, y: S.hist, c: 4 }, { name: "DP-SGD", x: ts, y: P_.hist, c: 0 }] },
        { title: "Son modelde örnek başı gradyan normu", xlabel: "‖gᵢ‖", ylabel: "örnek", yzero: true, series: [{ name: "örnekler", x: range(nb).map((i) => (i + 0.5) * bw), y: h, kind: "bar", c: 0 }], marks: [{ x: q.C, label: "C" }] },
      ],
      stats: [
        { label: "Test doğruluğu: SGD / DP-SGD", value: `${fmt(S.hist.at(-1), 3)} / ${fmt(P_.hist.at(-1), 3)}` },
        { label: "Kırpılan örnek oranı (son 50 adım)", value: fracClip },
        { label: "Gürültü std'si (ortalamada) σC/B", value: noiseRel },
        { label: "Doğruluk dalgalanması (son 50): SGD / DP-SGD", value: `${fmt(jS, 2)} / ${fmt(jP, 2)}` },
      ],
      note: P.join(" "),
    };
  },
};
