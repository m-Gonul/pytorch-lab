// M11.5 — Sutton & Barto Örnek 6.6: uçurum yürüyüşü. 4 × 12 ızgara, başlangıç sol alt (3, 0), hedef sağ alt (3, 11); alt sıradaki
// aradaki 10 hücre uçurum: girmek −100 ödül verir ve başlangıca döndürür; diğer her adım −1. Eylemler ↑ ↓ ← →, ızgara dışı hareket yerinde
// bırakır. ε-açgözlü davranışla SARSA, Q-öğrenme ve beklenen SARSA, sabit α; R çalıştırmanın bölüm başına toplam ödül ortalaması
// (10 bölümlük kayan ortalama) ve son çalıştırmada öğrenilen açgözlü yol.
import { fmt, range } from "./plot.js";

const H = 4, W = 12, START = [3, 0], GOAL = [3, 11], MV = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const isCliff = (r, c) => r === 3 && c >= 1 && c <= 10;
function step(r, c, a) { let rr = Math.min(H - 1, Math.max(0, r + MV[a][0])), cc = Math.min(W - 1, Math.max(0, c + MV[a][1])); if (isCliff(rr, cc)) return [START[0], START[1], -100, false]; return [rr, cc, -1, rr === GOAL[0] && cc === GOAL[1]]; }
const argmax = (q, rnd) => { const m = Math.max(...q), c = range(4).filter((a) => q[a] === m); return c[Math.floor(rnd() * c.length)]; };
export function train(alg, q, rnd, E) {
  const Q = range(H * W).map(() => [0, 0, 0, 0]), sums = [];
  const pol = (s) => (rnd() < q.eps ? Math.floor(rnd() * 4) : argmax(Q[s], rnd));
  for (let e = 0; e < E; e++) {
    let r = START[0], c = START[1], s = r * W + c, a = pol(s), tot = 0;
    for (let t = 0; t < 1000; t++) {
      const [r2, c2, rw, done] = step(r, c, a), s2 = r2 * W + c2; tot += rw;
      const a2 = pol(s2), qs = Q[s2];
      let tgt;
      if (done) tgt = rw;
      else if (alg === 0) tgt = rw + qs[a2];
      else if (alg === 1) tgt = rw + Math.max(...qs);
      else { const m = Math.max(...qs), nb = qs.filter((v) => v === m).length; tgt = rw + qs.reduce((acc, v) => acc + v * (q.eps / 4 + (v === m ? (1 - q.eps) / nb : 0)), 0); }
      Q[s][a] += q.alpha * (tgt - Q[s][a]);
      r = r2; c = c2; s = s2; a = a2; if (done) break;
    }
    sums.push(tot);
  }
  return { Q, sums };
}
function greedyPath(Q) { let r = START[0], c = START[1]; const p = [[r, c]]; for (let t = 0; t < 60; t++) { const q = Q[r * W + c], m = Math.max(...q), a = q.indexOf(m), [r2, c2, , done] = step(r, c, a); if (r2 === START[0] && c2 === START[1] && isCliff(r + MV[a][0], c + MV[a][1])) { p.push([3, c + 1]); break; } r = r2; c = c2; p.push([r, c]); if (done) break; } return p; }
const NAMES = ["SARSA", "Q-öğrenme", "beklenen SARSA"];
export default {
  params: [
    { id: "eps", label: "ε (keşif)", min: 0, max: 0.3, step: 0.01, value: 0.1 },
    { id: "alpha", label: "α", min: 0.05, max: 1, step: 0.05, value: 0.5 },
    { id: "E", label: "Bölüm sayısı", min: 50, max: 500, step: 50, value: 500, int: true },
    { id: "R", label: "Ortalaması alınan çalıştırma", min: 1, max: 50, value: 20, int: true },
  ],
  seed: true,
  run(q, rnd) {
    const avg = [0, 1, 2].map(() => Array(q.E).fill(0)), last = [];
    for (let k = 0; k < q.R; k++) for (let alg = 0; alg < 3; alg++) { const o = train(alg, q, rnd, q.E); o.sums.forEach((v, i) => (avg[alg][i] += v / q.R)); if (k === q.R - 1) last.push(o.Q); }
    const sm = (a) => a.map((_, i) => { const lo = Math.max(0, i - 9); let s = 0; for (let j = lo; j <= i; j++) s += a[j]; return s / (i - lo + 1); });
    const tail = (a) => a.slice(-100).reduce((x, y) => x + y, 0) / Math.min(100, a.length);
    const paths = last.map(greedyPath), reach = paths.map((p) => { const [r, c] = p[p.length - 1]; return r === GOAL[0] && c === GOAL[1]; }), plen = paths.map((p) => p.length - 1), edge = paths.map((p) => p.some(([r]) => r === 2) && !p.some(([r]) => r <= 1));
    const P = [`Son 100 bölümün ortalama toplam ödülü (ε = ${q.eps} ile davranırken): ${NAMES.map((n, i) => `${n} ${fmt(tail(avg[i]), 3)}`).join(", ")}.`];
    P.push(`Öğrenilen açgözlü yolun uzunluğu: ${NAMES.map((n, i) => `${n} ${reach[i] ? plen[i] : "hedefe ulaşmıyor (henüz öğrenilmedi)"}${reach[i] && edge[i] ? " (uçurum kenarı)" : ""}`).join(", ")}. En kısa yol 13 adımdır ve uçurumun hemen üstünden geçer.`);
    if (q.eps > 0) P.push("Q-öğrenme açgözlü politikanın değerini öğrenir ve kenardan gider; ama ε-açgözlü davranırken kenardaki her rastgele adım −100 getirir. SARSA keşfi değere katar ve daha güvenli, uzun bir yol seçer: eğitimde daha çok ödül toplar.");
    else P.push("ε = 0: keşif yok, SARSA ile Q-öğrenme aynı hedefleri kullanır; iki yöntem aynı en kısa yolu öğrenir (sıfır başlangıç değerleri bu küçük ızgarada keşfi iyimserlik yoluyla sağlıyor).");
    const cliff = range(10).map((i) => i + 1);
    return {
      plots: [
        { title: "Bölüm başına toplam ödül (10 bölümlük ortalama)", xlabel: "bölüm", ylabel: "toplam ödül", side: true, size: { w: 320, h: 240 }, ylim: [-120, 0], series: [0, 1, 2].map((i) => ({ name: NAMES[i], x: range(q.E).map((x) => x + 1), y: sm(avg[i]), c: i })) },
        { title: "Son çalıştırmada açgözlü yollar", xlabel: "sütun", ylabel: "satır (aşağı = 3)", side: true, size: { w: 300, h: 160 }, xlim: [-0.5, 11.5], ylim: [-3.5, 0.5],
          series: [{ name: "uçurum", x: cliff, y: cliff.map(() => -3), kind: "scatter", c: 4 }, ...[0, 1, 2].map((i) => ({ name: NAMES[i], x: paths[i].map((p) => p[1] + (i - 1) * 0.08), y: paths[i].map((p) => -p[0] + (i - 1) * 0.08), c: i }))] },
      ],
      stats: NAMES.map((n, i) => ({ label: `${n}: son 100 bölüm / yol`, value: `${fmt(tail(avg[i]), 3)} / ${reach[i] ? plen[i] : "—"}` })),
      note: P.join(" "),
    };
  },
};
