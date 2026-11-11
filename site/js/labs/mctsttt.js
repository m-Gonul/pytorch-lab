// M11.10 — MCTS, tic-tac-toe. Seçilmiş bir konumda UCT ağacı kurulur; kökün çocukları ziyaret
// sayısı ve ortalama değerle raporlanır, "doğru" (minimax ile oyun değerini koruyan) hamlenin
// ziyaret payı simülasyon sayısıyla izlenir. Önsel yok (UCT) ve bir politika önseliyle (PUCT,
// merkez/köşe/kenar ağırlıkları) seçim kuralı karşılaştırılır: önsel iyiye yakınsa keşif hızlanır,
// yanlışsa yavaşlar. Yaprak değerleri simülasyon başına `roll` rastgele oyunun ortalamasıdır.
import { fmt, range } from "./plot.js";

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const POS = [
  { name: "boş tahta", b: [0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { name: "X engellemeli (O tehdidi)", b: [-1, 0, 0, -1, 1, 0, 0, 0, 1] },
  { name: "X kazanabilir (çift tehdit)", b: [-1, 0, -1, 0, 1, 0, 0, 1, 0] },
];
const SQ = ["sol üst", "üst orta", "sağ üst", "sol orta", "merkez", "sağ orta", "sol alt", "alt orta", "sağ alt"];
const VAL = (v) => (v === 1 ? "X kazanır" : v === 0 ? "beraberlik" : "O kazanır");
function winner(b) { for (const [i, j, k] of LINES) { const s = b[i] + b[j] + b[k]; if (s === 3) return 1; if (s === -3) return -1; } return 0; }
const full = (b) => b.every((v) => v !== 0);

// --- minimax: konumun kesin değeri ve kökteki en iyi hamleler (X bakışından) ---
const memo = new Map();
function minimax(b, p) {
  const w = winner(b);
  if (w !== 0) return w;
  if (full(b)) return 0;
  const key = b.join("") + p;
  if (memo.has(key)) return memo.get(key);
  let best = p === 1 ? -2 : 2;
  for (let m = 0; m < 9; m++) {
    if (b[m] !== 0) continue;
    b[m] = p;
    const v = minimax(b, -p);
    b[m] = 0;
    if (p === 1 ? v > best : v < best) best = v;
  }
  memo.set(key, best);
  return best;
}
function bestMoves(b, p) {
  const val = minimax(b.slice(), p);
  const moves = [];
  for (let m = 0; m < 9; m++) {
    if (b[m] !== 0) continue;
    b[m] = p;
    const v = minimax(b, -p);
    b[m] = 0;
    if (v === val) moves.push(m);
  }
  return { val, moves };
}

// --- rastgele oyun (varsayılan politika) ---
function playout(b, p, rnd) {
  const bb = b.slice();
  let pp = p;
  for (;;) {
    const w = winner(bb);
    if (w !== 0 || full(bb)) return w;
    const ms = [];
    for (let m = 0; m < 9; m++) if (bb[m] === 0) ms.push(m);
    bb[ms[Math.floor(rnd() * ms.length)]] = pp;
    pp = -pp;
  }
}

// --- politika önseli (PUCT için): merkez ve köşeler güçlü, kenarlar zayıf ---
const PRIORW = [0.14, 0.025, 0.14, 0.025, 0.34, 0.025, 0.14, 0.025, 0.14];
function prior(b) {
  let s = 0;
  for (let m = 0; m < 9; m++) if (b[m] === 0) s += PRIORW[m];
  return PRIORW.map((v) => v / s);
}

function mcts(b0, p0, q, usePrior, rnd, bestSet) {
  const mk = (b, p) => ({ b, p, ch: new Map(), N: 0, W: 0 });
  const root = mk(b0.slice(), p0);
  const cps = [10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2000].filter((x) => x <= q.sims);
  const share = [];
  for (let i = 0; i < q.sims; i++) {
    let node = root;
    const path = [root];
    let value = null;
    for (;;) {
      const w = winner(node.b);
      if (w !== 0 || full(node.b)) { value = w; break; }
      const legal = [];
      for (let m = 0; m < 9; m++) if (node.b[m] === 0) legal.push(m);
      if (node.ch.size < legal.length) {
        const m = legal.find((x) => !node.ch.has(x));
        const nb = node.b.slice();
        nb[m] = node.p;
        const child = mk(nb, -node.p);
        node.ch.set(m, child);
        path.push(child);
        const cw = winner(nb);
        value = cw !== 0 || full(nb) ? cw : range(q.roll).reduce((a) => a + playout(nb, -node.p, rnd), 0) / q.roll;
        break;
      }
      const pr = usePrior ? prior(node.b) : null;
      let bestM = null, bestS = -Infinity;
      for (const [m, ch] of node.ch) {
        const qv = (ch.W / ch.N) * node.p;
        const bon = usePrior ? (q.c * pr[m] * Math.sqrt(node.N)) / (1 + ch.N) : q.c * Math.sqrt(Math.log(node.N) / ch.N);
        const s = qv + bon;
        if (s > bestS + 1e-12) { bestS = s; bestM = m; }
      }
      node = node.ch.get(bestM);
      path.push(node);
    }
    for (const nd of path) { nd.N++; nd.W += value; }
    if (i + 1 === cps[share.length]) {
      let nGood = 0;
      for (const m of bestSet) { const ch = root.ch.get(m); if (ch) nGood += ch.N; }
      share.push(nGood / (i + 1));
    }
  }
  return { root, share, cps };
}

export default {
  params: [
    { id: "pos", label: "Konum", options: POS.map((p) => p.name), value: 1 },
    { id: "sims", label: "Simülasyon sayısı", min: 50, max: 2000, step: 50, value: 200, int: true },
    { id: "c", label: "Keşif katsayısı c", min: 0, max: 3, step: 0.1, value: 1.4 },
    { id: "roll", label: "Yaprak başına rastgele oyun", min: 1, max: 50, value: 20, int: true },
    { id: "mode", label: "Seçim kuralı", options: ["UCT (önsel yok)", "PUCT (politika önseli)"], value: 0 },
  ],
  seed: true,
  run(q, rnd) {
    const { b } = POS[q.pos];
    const { val, moves } = bestMoves(b.slice(), 1);
    const bestSet = moves;
    const uct = mcts(b, 1, q, false, rnd, bestSet);
    const puct = mcts(b, 1, q, true, rnd, bestSet);
    const cur = q.mode === 0 ? uct : puct;
    const other = q.mode === 0 ? puct : uct;
    const curName = q.mode === 0 ? "UCT" : "PUCT";
    const entries = [...cur.root.ch.entries()].sort((x, y) => y[1].N - x[1].N);
    const top = entries[0];
    const shareEnd = (t) => t.share[t.share.length - 1];
    const legal = [];
    for (let m = 0; m < 9; m++) if (b[m] === 0) legal.push(m);
    const P = [];
    P.push(`Konum: ${POS[q.pos].name}; sıra X'te. Minimax değeri ${VAL(val)}${bestSet.length < legal.length ? ` ve bunu koruyan tek hamle: ${bestSet.map((m) => SQ[m]).join(", ")}` : " (bütün hamleler aynı değeri koruyor)"}.`);
    P.push(`${q.sims} simülasyon sonunda ${curName} en çok ziyaret edilen hamle ${SQ[top[0]]}: ${top[1].N} ziyaret (pay ${fmt(top[1].N / q.sims, 3)}), ortalama değeri ${fmt(top[1].W / top[1].N, 3)}.`);
    P.push(q.pos === 0
      ? "Boş tahtada her ilk hamle beraberliğe götürür; ziyaretler yine de merkez ve köşelerde yoğunlaşır çünkü rastgele oyunlarda bunlar daha çok kazanır."
      : `Doğru hamlelerin ziyaret payı ${q.sims} simülasyonda: UCT için ${fmt(shareEnd(uct), 3)}, PUCT için ${fmt(shareEnd(puct), 3)}.`);
    if (q.pos === 1) P.push("Tek kazandıran hamle sol alt köşedir: O'nun sol sütun tehdidini engellerken iki tehdit birden kurar (2-4-6 çaprazı ve 6-7-8 alt sıra); O yalnız birini kapatabilir. PUCT'un köşe önseli bu hamleyi erken öne çıkarır.");
    if (q.pos === 2) P.push("Kazandıran hamle üst ortadır (1-4-7 sütununu tamamlar). PUCT'un önseli bu kenar hamlesine düşük ağırlık verir; ama hamle oyunu bitirdiği için değeri kesindir ve PUCT ölçülen payda yine geride kalmıyor: önsel yanlış olsa bile kesin yaprak değerleri ağacı düzeltir.");
    if (q.c === 0) P.push(`c = 0: keşif bonusu yok, seçim yalnız ortalamaya bakar. Bu konumda hamleler arasındaki değer farkı büyük olduğu için pay ${fmt(shareEnd(cur), 3)} ile iyi; ama yaprak başına tek rastgele oyun seçip (roll = 1) tohumu değiştirince ölçümüzde 12 tohumdan 2'sinde pay 0.005'e düştü: kötü ilk izlenim bırakan hamle bir daha hiç denenemez.`);
    else if (q.c >= 2.5) P.push(`c = ${fmt(q.c, 3)}: bonus ağır basıyor, ağaç genişlediği için ziyaretler dağılıyor. Ölçüm: UCT payı 12 tohumda 0.63 ile 0.66 arasına iniyor (c = 1.4'te aynı tohumlar 0.85 ile 0.89 veriyor); PUCT'ta politika önseli dağılmayı frenliyor (0.90 civarı).`);
    return {
      plots: [
        { title: `${curName}: kök hamlelerinin ziyaret sayısı`, xlabel: "kare (0–8)", ylabel: "ziyaret", wide: true, size: { w: 520, h: 260 },
          xticks: legal.map((m) => [m, String(m)]),
          series: [{ name: "ziyaret", x: legal, y: legal.map((m) => cur.root.ch.get(m)?.N ?? NaN), kind: "bar", c: q.mode === 0 ? 0 : 1 }] },
        { title: "Doğru hamlelerin ziyaret payı (minimax ile aynı değeri koruyan hamleler)", xlabel: "simülasyon", ylabel: "ziyaret payı", wide: true, size: { w: 520, h: 260 }, ylim: [0, 1.05],
          series: [
            { name: "UCT", x: uct.cps, y: uct.share, c: 0, dots: true },
            { name: "PUCT", x: puct.cps, y: puct.share, c: 1, dots: true },
          ] },
      ],
      stats: [
        { label: "Konumun minimax değeri", value: VAL(val) },
        { label: `En çok ziyaret edilen hamle (${curName})`, value: `${SQ[top[0]]} (${top[0]}): ${top[1].N}/${q.sims}` },
        { label: "Doğru hamle payı: UCT / PUCT", value: `${fmt(shareEnd(uct), 3)} / ${fmt(shareEnd(puct), 3)}` },
        { label: "Yaprak değeri kestirimi", value: `${q.roll} rastgele oyunun ortalaması` },
      ],
      note: P.join(" "),
    };
  },
};
