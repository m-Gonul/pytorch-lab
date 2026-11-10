// M9.15 — RANSAC (Fischler & Bolles 1981) ile doğru uydurma: n nokta, bir kısmı y = a·x + b doğrusundan (σ gürültüyle), kalanı
// rastgele aykırı. Her iterasyonda s = 2 nokta seç, doğruyu kur, |kalıntı| < t olanları say; en çok içerdeğerli modeli içerdeğerlerle
// en küçük kareler ile yeniden kestir. Karşılaştırma: bütün noktalara en küçük kareler. Ayrıca kuramsal başarı olasılığı
// 1 − (1 − w^s)^N ile, aynı N'de 300 bağımsız denemenin deneysel başarı oranı (en az bir temiz örnek çekilmesi).
import { fmt, range } from "./plot.js";

const A0 = 0.6, B0 = 1;
function lsq(P) { const n = P.length, sx = P.reduce((s, p) => s + p[0], 0), sy = P.reduce((s, p) => s + p[1], 0), sxx = P.reduce((s, p) => s + p[0] * p[0], 0), sxy = P.reduce((s, p) => s + p[0] * p[1], 0), d = n * sxx - sx * sx; const a = (n * sxy - sx * sy) / d; return [a, (sy - a * sx) / n]; }
export function data(r, n, eps, sn) {
  return range(n).map(() => { const x = 10 * r(); if (r() < eps) return [x, -2 + 12 * r(), 1]; return [x, A0 * x + B0 + sn * r.normal(), 0]; });
}
export function ransac(P, N, t, r) {
  let best = null, bestIn = [];
  for (let k = 0; k < N; k++) {
    const i = Math.floor(r() * P.length); let j = Math.floor(r() * P.length); if (j === i) j = (j + 1) % P.length;
    const [x1, y1] = P[i], [x2, y2] = P[j]; if (Math.abs(x2 - x1) < 1e-9) continue;
    const a = (y2 - y1) / (x2 - x1), b = y1 - a * x1, inl = P.filter((p) => Math.abs(p[1] - (a * p[0] + b)) / Math.sqrt(1 + a * a) < t);
    if (inl.length > bestIn.length) { bestIn = inl; best = [a, b]; }
  }
  return { model: bestIn.length >= 2 ? lsq(bestIn) : best, inl: bestIn };
}
export default {
  params: [
    { id: "n", label: "Nokta sayısı", min: 20, max: 300, value: 100, int: true },
    { id: "eps", label: "Aykırı oranı ε", min: 0, max: 0.9, step: 0.05, value: 0.5 },
    { id: "sn", label: "İçerdeğer gürültüsü σ", min: 0, max: 0.5, step: 0.01, value: 0.1 },
    { id: "t", label: "Eşik t (dik uzaklık)", min: 0.05, max: 2, step: 0.05, value: 0.3 },
    { id: "N", label: "İterasyon N", min: 1, max: 200, value: 17, int: true },
  ],
  seed: true,
  run(q, r) {
    const P = data(r, q.n, q.eps, q.sn), R = ransac(P, q.N, q.t, r), L = lsq(P);
    const nIn = P.filter((p) => !p[2]).length, w = nIn / P.length, w2 = (nIn * (nIn - 1)) / (P.length * (P.length - 1)), pth = 1 - (1 - w2) ** q.N, need = w2 > 0 && w2 < 1 ? Math.ceil(Math.log(1 - 0.99) / Math.log(1 - w2)) : 1, needNom = q.eps > 0 && q.eps < 1 ? Math.ceil(Math.log(0.01) / Math.log(1 - (1 - q.eps) ** 2)) : 1;
    // Deneysel: aynı N ile 1000 deneme; başarı = en az bir örneğin iki içerdeğerden oluşması.
    let succ = 0; for (let k = 0; k < 1000; k++) { let ok = false; for (let i = 0; i < q.N && !ok; i++) { const a = Math.floor(r() * P.length); let b = Math.floor(r() * P.length); if (b === a) b = (b + 1) % P.length; if (!P[a][2] && !P[b][2]) ok = true; } if (ok) succ++; }
    const err = (m) => Math.abs(m[0] - A0) + Math.abs(m[1] - B0);
    const xs = [0, 10], line = (m) => xs.map((x) => m[0] * x + m[1]);
    const Ptxt = [`Bu örnekte gerçekleşen içerdeğer oranı w = ${fmt(w, 3)} (ε = ${q.eps} beklenen ${fmt(1 - q.eps, 3)}). N = ${q.N} iterasyonla en az bir temiz örnek çekme olasılığı (kuram, yerine koymadan çekilen iki nokta) ${fmt(pth, 4)}; 1000 denemede deneysel oran ${fmt(succ / 1000, 3)}. p = 0.99 için gereken N: gerçekleşen w ile ${need}, nominal ε ile ${needNom}.`];
    Ptxt.push(`RANSAC modeli a = ${fmt(R.model[0], 4)}, b = ${fmt(R.model[1], 4)} (${R.inl.length} içerdeğer), parametre hatası ${fmt(err(R.model), 3)}; bütün noktalara en küçük kareler a = ${fmt(L[0], 4)}, b = ${fmt(L[1], 4)}, hata ${fmt(err(L), 3)}.`);
    if (q.t < 2 * q.sn) Ptxt.push("Eşik gürültüye göre küçük: gerçek içerdeğerlerin bir kısmı dışarıda kalıyor, en iyi model daha az destekle seçiliyor.");
    if (q.t > 1) Ptxt.push("Eşik büyük: aykırı noktaların da bir kısmı içerdeğer sayılıyor, yeniden kestirimi bozuyor.");
    return {
      plots: [{ title: "Noktalar, RANSAC ve en küçük kareler", xlabel: "x", ylabel: "y", wide: true, size: { w: 620, h: 260 }, series: [{ name: "içerdeğer (RANSAC)", x: R.inl.map((p) => p[0]), y: R.inl.map((p) => p[1]), kind: "scatter", c: 2, r: 3 }, { name: "diğer", x: P.filter((p) => !R.inl.includes(p)).map((p) => p[0]), y: P.filter((p) => !R.inl.includes(p)).map((p) => p[1]), kind: "scatter", c: 4, r: 2.5 }, { name: "RANSAC", x: xs, y: line(R.model), c: 0 }, { name: "en küçük kareler", x: xs, y: line(L), c: 1, dash: true }, { name: "gerçek", x: xs, y: line([A0, B0]), c: 3, dash: true }], ylim: [-3, 11] }],
      stats: [{ label: "Gerçekleşen içerdeğer oranı w", value: w }, { label: "Başarı olasılığı: kuram / deney", value: `${fmt(pth, 4)} / ${fmt(succ / 1000, 3)}` }, { label: "p = 0.99 için gereken N (gerçek w / nominal)", value: `${need} / ${needNom}` }, { label: "Parametre hatası: RANSAC / EKK", value: `${fmt(err(R.model), 3)} / ${fmt(err(L), 3)}` }, { label: "İçerdeğer sayısı", value: R.inl.length }],
      note: Ptxt.join(" "),
    };
  },
};
