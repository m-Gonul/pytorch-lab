// M5.5 — REINFORCE, Sutton & Barto'nun "eylemleri yer değiştirmiş kısa koridor" örneğinde (Örnek 13.1). Üç durum + hedef;
// her adım −1 ödül; 2. durumda sağ/sol ters çalışır. Özellikler tüm durumlarda aynı olduğu için politika tek bir sayıdır:
// P(sağ) = σ(θ). En iyi politika stokastiktir (p* ≈ 0.59); kesin J(p) doğrusal sistemden. REINFORCE (taban çizgisiz ve
// öğrenilen ortalama getiri taban çizgisiyle), R çalıştırmanın ortalaması.
import { fmt, range } from "./plot.js";

export function J(p) {
  // v0 = −1 + p v1 + (1−p) v0;  v1 = −1 + p v0 + (1−p) v2;  v2 = −1 + (1−p) v1   (γ = 1)
  if (p <= 0 || p >= 1) return -Infinity;
  const A = [[p, -p, 0], [-p, 1, -(1 - p)], [0, -(1 - p), 1]], b = [-1, -1, -1];
  // Gauss eliminasyonu
  const M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < 3; i++) { let m = i; for (let k = i + 1; k < 3; k++) if (Math.abs(M[k][i]) > Math.abs(M[m][i])) m = k; [M[i], M[m]] = [M[m], M[i]]; for (let k = i + 1; k < 3; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j < 4; j++) M[k][j] -= f * M[i][j]; } }
  const x = [0, 0, 0]; for (let i = 2; i >= 0; i--) x[i] = (M[i][3] - M[i].slice(i + 1, 3).reduce((s, v, j) => s + v * x[i + 1 + j], 0)) / M[i][i];
  return x[0];
}
function episode(p, r, maxT = 200) {
  let s = 0, T = 0; const acts = [];
  while (s !== 3 && T < maxT) { const right = r() < p; acts.push(right); const dir = s === 1 ? (right ? -1 : 1) : right ? 1 : -1; s = Math.max(0, s + dir); T++; }
  return acts; // her adım −1; getiri G_t = −(T − t)
}
const sig = (x) => 1 / (1 + Math.exp(-x));

export default {
  params: [
    { id: "alpha", label: "Öğrenme oranı α", min: 1e-4, max: 0.05, value: 0.002, log: true },
    { id: "base", label: "Taban çizgisi", options: ["yok", "öğrenilen ortalama getiri"], value: 1 },
    { id: "episodes", label: "Bölüm sayısı", min: 50, max: 2000, value: 1000, int: true, log: true },
    { id: "theta0", label: "Başlangıç θ (P(sağ) = σ(θ))", min: -3, max: 3, step: 0.1, value: -1.5 },
    { id: "runs", label: "Ortalama alınan çalıştırma", min: 1, max: 50, value: 20, int: true },
  ],
  seed: true,
  run(q, r) {
    const ps = range(99).map((i) => (i + 1) / 100), Js = ps.map(J), bi = Js.indexOf(Math.max(...Js)), pstar = ps[bi], Jstar = Js[bi];
    const curve = new Array(q.episodes).fill(0), rets = new Array(q.episodes).fill(0); let gvar = 0, gn = 0;
    for (let run = 0; run < q.runs; run++) {
      let th = q.theta0, b = 0;
      for (let e = 0; e < q.episodes; e++) {
        const p = sig(th), acts = episode(p, r), T = acts.length;
        let g = 0;
        acts.forEach((right, t) => { const G = -(T - t), adv = G - (q.base ? b : 0), dlog = right ? 1 - p : -p; g += adv * dlog; });
        th += q.alpha * g; if (q.base) b += 0.05 * (-T - b);
        gvar += g * g; gn++;
        curve[e] += sig(th) / q.runs; rets[e] += -T / q.runs;
      }
    }
    const pEnd = curve.at(-1), JEnd = J(pEnd), win = Math.max(1, Math.floor(q.episodes / 20)), smooth = rets.map((_, i) => rets.slice(Math.max(0, i - win + 1), i + 1).reduce((a, b) => a + b, 0) / Math.min(i + 1, win));
    const P = [`Kesin çözüm: J(p) en büyük p* = ${fmt(pstar, 2)}'de, J* = ${fmt(Jstar, 3)}. Deterministik politikalar (p = 0 ya da 1) hedefe hiç ulaşamaz (J = −∞): ikinci durumda eylemler ters çalıştığı ve özellikler durumları ayırmadığı için en iyi politika <b>stokastiktir</b>.`];
    P.push(`${q.runs} çalıştırmanın ortalamasında ${q.episodes} bölüm sonra P(sağ) = ${fmt(pEnd, 3)}, J = ${fmt(JEnd, 3)}. ${Math.abs(pEnd - pstar) < 0.05 ? "Politika optimuma ulaştı." : "Politika henüz optimumda değil."} Bölüm başına gradyan kestiriminin ortalama karesi ${fmt(gvar / gn, 3)}${q.base ? " (taban çizgisiyle)" : " (taban çizgisiz)"}.`);
    P.push(q.base ? "Taban çizgisi b, getirilerin hareketli ortalamasıdır; gradyandan b·∇log π çıkarmak beklentiyi değiştirmez (E[∇log π] = 0) ama varyansı düşürür." : "Taban çizgisi yok: bu ortamda tüm getiriler negatif olduğu için her bölümde seçilen eylemlerin olasılığı düşürülür; sinyal yalnız 'daha az negatif' bölümlerin daha az itilmesinden gelir ve gürültülüdür.");
    return {
      plots: [
        { title: "P(sağ) öğrenme sürecinde", xlabel: "bölüm", ylabel: "P(sağ)", ylim: [0, 1], series: [{ name: "REINFORCE (ortalama)", x: range(q.episodes), y: curve, c: 0 }], marks: [{ y: pstar, label: `p* = ${fmt(pstar, 2)}` }] },
        { title: "Kesin J(p) ve bölüm getirisi", xlabel: "p = P(sağ) / bölüm (kayan ort.)", ylabel: "J", ylim: [-60, 0], series: [
          { name: "kesin J(p)", x: ps, y: Js.map((v) => Math.max(v, -60)), c: 3 },
          { name: "öğrenme sırasında getiri (x: bölüm / bölüm sayısı)", x: range(q.episodes).map((e) => e / q.episodes), y: smooth.map((v) => Math.max(v, -60)), c: 1, dash: true }], marks: [{ x: pstar, label: "p*" }] },
      ],
      stats: [
        { label: "p*, J*", value: `${fmt(pstar, 2)}, ${fmt(Jstar, 3)}` },
        { label: "Son P(sağ)", value: pEnd },
        { label: "Son politikanın kesin J'si", value: JEnd },
        { label: "Gradyan kestiriminin ortalama karesi", value: gvar / gn },
      ],
      note: P.join(" "),
    };
  },
};
