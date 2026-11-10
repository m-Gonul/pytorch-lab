// M10.6 — Paralel tarama algoritmaları: naif (her çıktı ayrı indirgeme), Kogge–Stone (adım 1, 2, 4, …: XY[i] += XY[i − adım]),
// Brent–Kung (yukarı süpürme + aşağı süpürme). Her adımda etkin işlem sayısı, toplam toplama ve adım sayısı. Kogge–Stone'da
// yazma-sonrası-okuma (WAR) tehlikesi: geçici değişken ve ikinci bariyer olmadan, bir adım içinde thread'ler rastgele sırayla
// "oku-topla-yaz" yaparsa bazı thread'ler komşunun bu adımda güncellenmiş değerini okur; 200 rastgele sırada yanlış çıktı oranı.
import { fmt, range } from "./plot.js";

function ksSteps(n) { const st = []; for (let s = 1; s < n; s *= 2) st.push(n - s); return st; }
function bkSteps(n) { const up = [], dn = []; for (let s = 1; s < n; s *= 2) up.push(Math.floor(n / (2 * s))); for (let s = n / 4; s >= 1; s /= 2) { let c = 0; for (let i = 2 * s - 1; i + s < n; i += 2 * s) c++; dn.push(c); } return [...up, ...dn]; }
function ksRace(x, r) { // geçici değişken yok: thread'ler adım içinde rastgele sırayla oku-topla-yaz
  const XY = x.slice(), n = x.length;
  for (let s = 1; s < n; s *= 2) { const order = range(n - s).map((k) => k + s); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } for (const i of order) XY[i] = XY[i] + XY[i - s]; }
  return XY;
}
export default {
  params: [
    { id: "alg", label: "Algoritma", options: ["naif (her çıktı ayrı)", "Kogge–Stone", "Brent–Kung"], value: 1 },
    { id: "logn", label: "Bölüm boyu n = 2^k", min: 3, max: 10, value: 4, int: true },
    { id: "safe", label: "Kogge–Stone güncellemesi", options: ["geçici değişken + 2. bariyer (ya da çift tampon)", "yerinde, tek bariyer (WAR tehlikesi)"], value: 0, on: { alg: [1] } },
  ],
  seed: true,
  seedOn: { alg: 1, safe: 1 },
  run(q, r) {
    const n = 2 ** q.logn, L = q.logn, steps = q.alg === 0 ? range(n).map((i) => i).reverse().slice(0, n - 1).map(() => null) : null;
    const naiveWork = (n * (n - 1)) / 2, ks = ksSteps(n), bk = bkSteps(n), ksW = ks.reduce((a, b) => a + b, 0), bkW = bk.reduce((a, b) => a + b, 0);
    const per = q.alg === 0 ? range(n - 1).map((k) => n - 1 - k) : q.alg === 1 ? ks : bk, work = q.alg === 0 ? naiveWork : q.alg === 1 ? ksW : bkW, nsteps = q.alg === 0 ? n - 1 : per.length;
    let raceTxt = "", wrongFrac = null;
    if (q.alg === 1 && q.safe === 1) { const x = range(n).map((i) => 1 + (i % 3)), ok = x.map((_, i) => x.slice(0, i + 1).reduce((a, b) => a + b, 0)); let bad = 0, badEl = 0; for (let t = 0; t < 200; t++) { const y = ksRace(x, r), e = y.filter((v, i) => v !== ok[i]).length; if (e) bad++; badEl += e; } wrongFrac = bad / 200; raceTxt = ` WAR tehlikesi: thread'ler adım içinde tamamen rastgele sırayla çalıştırılınca 200 zamanlamanın ${fmt(100 * wrongFrac, 3)}%'inde çıktı yanlış (ortalama ${fmt(badEl / 200, 3)} yanlış öğe). Bu kötü durumdur: gerçek GPU'da bir warp'ın okumaları yazmalarından önce yapılır ve yarış warp'lar arasında, zamanlamaya bağlı olarak arada bir ortaya çıkar; bu da onu daha tehlikeli kılar.`; }
    const P = [`n = ${n}: ${q.alg === 0 ? "naif" : q.alg === 1 ? "Kogge–Stone" : "Brent–Kung"} ${work} toplama, ${nsteps} adım. Sıralı tarama ${n - 1} toplama, ${n - 1} adım. Kogge–Stone n log₂n − (n − 1) = ${ksW} (sıralının ${fmt(ksW / (n - 1), 3)} katı), Brent–Kung 2n − 2 − log₂n = ${bkW} (${fmt(bkW / (n - 1), 3)} katı), naif n(n − 1)/2 = ${naiveWork}.${raceTxt}`];
    if (q.alg === 0) P.push("Naif yaklaşımda en uzun thread n − 1 adım sürer: sıralıdan hızlı değildir ve O(n²) iş yapar.");
    if (q.alg === 2) P.push(`Brent–Kung en çok n/2 = ${n / 2} thread ister ama 2 log₂n − 1 = ${2 * L - 1} adım sürer; etkin thread sayısı hızla düşer.`);
    return {
      plots: [{ title: "Adım başına toplama sayısı", xlabel: "adım", ylabel: "toplama", wide: true, size: { w: 620, h: 220 }, xlim: [0.4, Math.min(per.length, 64) + 0.6], series: [{ name: "toplama", x: range(Math.min(per.length, 64)).map((i) => i + 1), y: per.slice(0, 64), kind: "bar", c: q.alg }] }],
      stats: [{ label: "Toplam toplama", value: work }, { label: "Adım", value: nsteps }, { label: "İş / sıralı", value: fmt(work / (n - 1), 3) }, ...(wrongFrac !== null ? [{ label: "Yanlış çıktı oranı (WAR)", value: fmt(wrongFrac, 3) }] : [])],
      note: P.join(" "),
    };
  },
};
