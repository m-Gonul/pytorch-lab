// M11.10 — Dyna labirenti (Sutton–Barto Şekil 8.2 ve Örnek 8.3). İki deney aynı anda:
// (A) statik labirentte bölüm başına adım sayısı: n = 0 (Dyna'sız Q-öğrenme), seçili n ve n = 50;
// (B) kısayol deneyi: 3000. adımda sağdan geçit açılır; Dyna-Q ve Dyna-Q+ (κ√τ bonusu, κ = 0'da
// yalnız denenmemiş eylemlerin modele eklenmesi) birikimli ödül olarak karşılaştırılır.
// Model: tablo, yalnız gerçekte görülmüş (s, a) çiftlerinden simüle adım; Dyna-Q+ bir durumu ilk
// gördüğünde denenmemiş eylemleri "aynı durumda kal, ödül 0" olarak modele ekler.
import { fmt, range } from "./plot.js";

const H = 6, W = 9, START = 2 * W + 0, GOAL = 0 * W + 8;
const WALLS = new Set([1 * W + 2, 2 * W + 2, 3 * W + 2, 4 * W + 5, 0 * W + 7, 1 * W + 7, 2 * W + 7]);
const MV = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const sc = (r, c) => r * W + c;
function stepMaze(s, a) {
  const r = Math.floor(s / W) + MV[a][0], c = (s % W) + MV[a][1];
  let s2 = s;
  if (r >= 0 && r < H && c >= 0 && c < W && !WALLS.has(sc(r, c))) s2 = sc(r, c);
  const done = s2 === GOAL;
  return [s2, done ? 1 : 0, done];
}
// kısayol labirenti: başlangıç (5,3), hedef (0,8); 3. satırda duvar (sütun 1–7 hep kapalı),
// sütun 8 başta kapalı, 3000. adımda açılır (sağdan kısayol); geçit her zaman solda (sütun 0)
const START2 = sc(5, 3);
function stepShortcut(s, a, open8) {
  const r = Math.floor(s / W) + MV[a][0], c = (s % W) + MV[a][1];
  let s2 = s;
  const wall = r === 3 && c >= 1 && (c <= 7 || !open8);
  if (r >= 0 && r < H && c >= 0 && c < W && !wall) s2 = sc(r, c);
  const done = s2 === GOAL;
  return [s2, done ? 1 : 0, done];
}
function bfs(open) {
  // statik labirentte en kısa yol (open = 9: kısayol yok)
  const dist = new Array(H * W).fill(Infinity);
  dist[START] = 0;
  const qu = [START];
  while (qu.length) {
    const s = qu.shift();
    for (let a = 0; a < 4; a++) {
      const [s2] = stepMaze(s, a);
      if (dist[s2] === Infinity) { dist[s2] = dist[s] + 1; qu.push(s2); }
    }
  }
  return dist[GOAL];
}
const argmax = (qs, rnd) => { const m = Math.max(...qs); const c = range(4).filter((a) => qs[a] === m); return c[Math.floor(rnd() * c.length)]; };

// --- (A) statik labirent ---
function dynaStatic(n, q, rnd) {
  const Q = Array.from({ length: H * W }, () => [0, 0, 0, 0]);
  const model = new Map(), keys = [];
  const steps = [];
  const pol = (s) => (rnd() < q.eps ? Math.floor(rnd() * 4) : argmax(Q[s], rnd));
  for (let e = 0; e < q.E; e++) {
    let s = START, k = 0;
    for (let t = 0; t < 2000; t++) {
      const a = pol(s), [s2, r, done] = stepMaze(s, a);
      k++;
      const qs = Q[s];
      qs[a] += q.alpha * (r + (done ? 0 : 0.95 * Math.max(...Q[s2])) - qs[a]);
      const key = s * 4 + a;
      if (!model.has(key)) keys.push(key);
      model.set(key, [r, s2, done]);
      for (let i = 0; i < n; i++) {
        const pk = keys[Math.floor(rnd() * keys.length)];
        const [pr, ps2, pd] = model.get(pk);
        const ps = Math.floor(pk / 4), pa = pk % 4, qq = Q[ps];
        qq[pa] += q.alpha * (pr + (pd ? 0 : 0.95 * Math.max(...Q[ps2])) - qq[pa]);
      }
      s = s2;
      if (done) break;
    }
    steps.push(k);
  }
  return steps;
}
// --- (B) kısayol deneyi ---
function dynaShortcut(plus, q, rnd) {
  const Q = Array.from({ length: H * W }, () => [0, 0, 0, 0]);
  const model = new Map(), keys = [], last = new Map(), seen = new Set();
  const cum = [];
  let s = START2, tot = 0;
  const CHANGE = 3000, T = 6000;
  const pol = () => (rnd() < q.eps ? Math.floor(rnd() * 4) : argmax(Q[s], rnd));
  const put = (k, v) => { if (!model.has(k)) keys.push(k); model.set(k, v); };
  for (let t = 0; t < T; t++) {
    const open8 = t >= CHANGE; // sütun 8'in açılıp açılmadığı
    const a = pol(), [s2, r, done] = stepShortcut(s, a, open8);
    const qs = Q[s];
    qs[a] += q.alpha * (r + (done ? 0 : 0.95 * Math.max(...Q[s2])) - qs[a]);
    if (plus && !seen.has(s)) for (let b = 0; b < 4; b++) if (b !== a && !model.has(s * 4 + b)) put(s * 4 + b, [0, s, false]);
    seen.add(s);
    put(s * 4 + a, [r, s2, done]);
    last.set(s * 4 + a, t);
    for (let i = 0; i < q.n; i++) {
      const pk = keys[Math.floor(rnd() * keys.length)];
      let [pr, ps2, pd] = model.get(pk);
      if (plus) pr += q.kappa * Math.sqrt(t - (last.get(pk) ?? 0));
      const ps = Math.floor(pk / 4), pa = pk % 4, qq = Q[ps];
      qq[pa] += q.alpha * (pr + (pd ? 0 : 0.95 * Math.max(...Q[ps2])) - qq[pa]);
    }
    tot += r;
    cum.push(tot);
    s = done ? START2 : s2;
  }
  return cum;
}

export default {
  params: [
    { id: "n", label: "Planlama adımı n", min: 0, max: 50, step: 5, value: 10, int: true },
    { id: "eps", label: "ε (keşif)", min: 0, max: 0.3, step: 0.01, value: 0.1 },
    { id: "alpha", label: "α", min: 0.05, max: 1, step: 0.05, value: 0.5 },
    { id: "E", label: "Bölüm sayısı", min: 10, max: 150, step: 10, value: 40, int: true },
    { id: "R", label: "Çalıştırma sayısı", min: 3, max: 20, value: 10, int: true },
    { id: "kappa", label: "Dyna-Q+ bonus κ", min: 0, max: 0.02, step: 0.001, value: 0.001 },
  ],
  seed: true,
  run(q, rnd) {
    // (A) üç ayar: n = 0, seçili n, n = 50
    const cfgs = [...new Set([0, q.n, 50])].sort((x, y) => x - y).map((n) => ({ n, name: `n = ${n}` }));
    const avg = cfgs.map(({ n }) => {
      const acc = Array(q.E).fill(0);
      for (let k = 0; k < q.R; k++) dynaStatic(n, q, rnd).forEach((v, i) => (acc[i] += v / q.R));
      return acc;
    });
    const ep = (a, i) => fmt(a[i], 3);
    // (B) kısayol deneyi: Dyna-Q ve Dyna-Q+
    const cumA = dynaShortcut(false, q, rnd), cumB = dynaShortcut(true, q, rnd);
    const halfA = cumA[5999] - cumA[2999], halfB = cumB[5999] - cumB[2999]; // değişimden sonraki 3000 adım
    const firstA = cumA[2999], firstB = cumB[2999]; // ilk 3000 adım
    const ratio = halfB / Math.max(halfA, 1e-9);
    const P = [];
    P.push(`Statik labirentte en kısa yol ${bfs(9)} adımdır. İkinci bölümde ortalama adım sayısı: ${cfgs.map((c, i) => `n = ${c.n} için ${ep(avg[i], 1)}`).join(", ")}.`);
    P.push(q.n === 0
      ? "n = 0: her gerçek adımda yalnız bir Q güncellemesi yapılır; keşif ε'ye kaldığı için bilgi başlangıç durumunun çevresinde yavaş birikir."
      : `Planlama, gerçek deneyimi ${q.n + 1} katına çıkarır: n = 0'dan n = ${q.n}'e geçince ikinci bölüm ${fmt(avg[0][1] / avg[cfgs.findIndex((c) => c.n === q.n)][1], 2)} kat kısalıyor.`);
    if (q.kappa > 0) P.push(`Kısayol deneyinde (3000. adımda sağdaki geçit açılır) değişimden sonraki 3000 adımda toplam ödül: Dyna-Q için ${fmt(halfA, 3)}, Dyna-Q+ için ${fmt(halfB, 3)} (oran ${fmt(ratio, 3)}). Dyna-Q+ κ√τ bonusu sayesinde kısayolu bulur; bedeli, değişimden önceki ilk yarıda ${fmt(firstB, 3)} ile ${fmt(firstA, 3)} arasında kalmasıdır (Dyna-Q+ değişmeyen ortamda da keşfe harcama yapar).`);
    else P.push(`Kısayol deneyinde κ = 0: bonus kapalı, Dyna-Q+ yalnız denenmemiş eylemleri modele ekliyor. Değişimden sonraki ödül Dyna-Q için ${fmt(halfA, 3)}, Dyna-Q+ için ${fmt(halfB, 3)}: bonus olmadan kısayolun keşfi için gerekli itme yok.`);
    return {
      plots: [
        { title: "Statik labirent: bölüm başına adım (ortalama)", xlabel: "bölüm", ylabel: "adım sayısı", wide: true, ylog: true, size: { w: 520, h: 280 },
          series: cfgs.map((c, i) => ({ name: c.name, x: range(q.E).map((x) => x + 1), y: avg[i], c: i, dots: q.E <= 20 })) },
        { title: "Kısayol deneyi: birikimli ödül (değişim 3000. adımda)", xlabel: "gerçek adım", ylabel: "birikimli ödül", wide: true, size: { w: 520, h: 280 },
          marks: [{ x: 3000, label: "kısayol açılır" }],
          series: [
            { name: "Dyna-Q", x: range(6000).filter((t) => t % 50 === 0), y: range(6000).filter((t) => t % 50 === 0).map((t) => cumA[t]), c: 3 },
            { name: "Dyna-Q+", x: range(6000).filter((t) => t % 50 === 0), y: range(6000).filter((t) => t % 50 === 0).map((t) => cumB[t]), c: 0 },
          ] },
      ],
      stats: [
        { label: "En kısa yol (BFS)", value: bfs(9) },
        ...cfgs.map((c, i) => ({ label: `n = ${c.n}: bölüm 1 / 2 / son`, value: `${fmt(avg[i][0], 4)} / ${fmt(avg[i][1], 4)} / ${fmt(avg[i][q.E - 1], 4)}` })),
        { label: "İlk yarı ödül: Dyna-Q / Dyna-Q+", value: `${fmt(firstA, 4)} / ${fmt(firstB, 4)}` },
        { label: "İkinci yarı ödül: Dyna-Q / Dyna-Q+", value: `${fmt(halfA, 4)} / ${fmt(halfB, 4)}` },
        { label: "Değişim sonrası ödül oranı (+/düz)", value: fmt(ratio, 3) },
      ],
      note: P.join(" "),
    };
  },
};
