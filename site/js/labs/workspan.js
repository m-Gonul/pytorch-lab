// M10.1 — İş–yol (work–span) modeli ve açgözlü liste zamanlama. Görev bağımlılık grafikleri: böl ve yönet (merge sort: birleştirme maliyeti
// alt dizi boyu), ağaç indirgeme, rastgele katmanlı DAG, seri baş/son + bağımsız gövde (fork–join). İş W = görev maliyetleri toplamı,
// yol S∞ = en uzun (ağırlıklı) yol. p işlemcide açgözlü zamanlama: bir işlemci boşaldığında hazır bir görev başlatılır. Hazır kuyruğu
// sırası: FIFO ya da kritik yol önceliği (görevden çıkışa en uzun yol). Graham/Brent: max(W/p, S∞) ≤ T_p ≤ W/p + S∞ (her açgözlü zamanlama).
import { fmt, range } from "./plot.js";

function buildGraph(kind, n, r) {
  const cost = [], succ = [], pred = [];
  const add = (c, ps) => { const id = cost.length; cost.push(c); succ.push([]); pred.push(ps); for (const p of ps) succ[p].push(id); return id; };
  if (kind === 0) { // merge sort n öğe: yaprak (boy 1) maliyet 1, birleştirme maliyeti boy
    const ms = (m) => (m <= 1 ? add(1, []) : (() => { const a = ms(Math.floor(m / 2)), b = ms(m - Math.floor(m / 2)); return add(m, [a, b]); })());
    ms(n);
  } else if (kind === 1) { // ağaç indirgeme: n sayı, her toplama maliyet 1
    let lvl = range(n).map(() => add(0, [])); while (lvl.length > 1) { const nx = []; for (let i = 0; i + 1 < lvl.length; i += 2) nx.push(add(1, [lvl[i], lvl[i + 1]])); if (lvl.length % 2) nx.push(lvl[lvl.length - 1]); lvl = nx; }
  } else if (kind === 2) { // rastgele katmanlı DAG: L = ⌈√n⌉ katman
    const L = Math.max(2, Math.round(Math.sqrt(n))); let prev = [];
    for (let l = 0; l < L; l++) { const w = 1 + Math.floor(r() * (2 * n / L)), cur = []; for (let k = 0; k < w; k++) { const ps = prev.length ? prev.filter(() => r() < 2 / prev.length) : []; if (prev.length && !ps.length) ps.push(prev[Math.floor(r() * prev.length)]); cur.push(add(1 + Math.floor(r() * 9), ps)); } prev = cur; }
  } else { // fork–join: seri baş (n/10), n bağımsız görev (maliyet 1..9), seri son (n/10)
    const s = Math.max(1, Math.round(n / 10)); let h = add(s, []); const body = range(n).map(() => add(1 + Math.floor(r() * 9), [h])); add(s, body);
  }
  return { cost, succ, pred };
}
function bottomLevel(G) { const n = G.cost.length, bl = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let m = 0; for (const s of G.succ[i]) m = Math.max(m, bl[s]); bl[i] = G.cost[i] + m; } return bl; } // id'ler topolojik sırada
export function schedule(G, p, cp) {
  const n = G.cost.length, bl = cp ? bottomLevel(G) : null, indeg = G.pred.map((ps) => ps.length), ready = [], gantt = [];
  for (let i = 0; i < n; i++) if (!indeg[i]) ready.push(i);
  const free = range(p).map(() => 0), running = []; let t = 0, done = 0;
  const pick = () => { if (!cp) return ready.shift(); let bi = 0; for (let k = 1; k < ready.length; k++) if (bl[ready[k]] > bl[ready[bi]]) bi = k; return ready.splice(bi, 1)[0]; };
  while (done < n) {
    for (let q = 0; q < p && ready.length; q++) if (free[q] <= t) { const i = pick(); free[q] = t + G.cost[i]; running.push([free[q], i]); gantt.push([q, t, free[q]]); }
    running.sort((a, b) => a[0] - b[0]);
    const [tf] = running[0]; t = tf;
    while (running.length && running[0][0] <= t) { const [, i] = running.shift(); done++; for (const s of G.succ[i]) if (--indeg[s] === 0) ready.push(s); }
  }
  return { T: t, gantt };
}
export default {
  params: [
    { id: "kind", label: "Görev grafiği", options: ["böl ve yönet (merge sort)", "ağaç indirgeme", "rastgele katmanlı DAG", "seri baş/son + bağımsız gövde"], value: 0 },
    { id: "n", label: "Problem boyu n", min: 8, max: 512, value: 128, int: true, log: true },
    { id: "p", label: "İşlemci sayısı p", min: 1, max: 64, value: 8, int: true, log: true },
    { id: "cp", label: "Hazır kuyruğu sırası", options: ["FIFO", "kritik yol önceliği"], value: 0, on: { kind: [2, 3] } },
  ],
  seed: true,
  seedOn: { kind: 2 },
  run(q, r) {
    const G = buildGraph(q.kind, q.n, r), W = G.cost.reduce((a, b) => a + b, 0), bl = bottomLevel(G), S = Math.max(...bl);
    const ps = [1, 2, 4, 8, 16, 32, 64], Ts = ps.map((p) => schedule(G, p, q.cp === 1).T), cur = schedule(G, q.p, q.cp === 1), T1 = W;
    const gx = [[], []], gy = [[], []], cnt = new Int32Array(64); for (const [proc, a, b] of cur.gantt) if (proc < 16 && b > a) { const k = cnt[proc]++ % 2; gx[k].push(a + 0.1, b - 0.1, NaN); gy[k].push(proc + 0.5, proc + 0.5, NaN); }
    const sp = T1 / cur.T, eff = sp / q.p, lb = Math.max(W / q.p, S), ub = W / q.p + S;
    const P = [`İş W = ${W}, yol S∞ = ${S}, paralellik W/S∞ = ${fmt(W / S, 3)}. p = ${q.p} ile T_p = ${cur.T}: hızlanma ${fmt(sp, 3)}, verim ${fmt(100 * eff, 3)}%. Sınırlar: max(W/p, S∞) = ${fmt(lb, 4)} ≤ T_p ≤ W/p + S∞ = ${fmt(ub, 4)}.`];
    if (q.p > W / S) P.push(`p, paralellikten (${fmt(W / S, 3)}) büyük: işlemcilerin çoğu boşta, T_p yola dayanıyor.`);
    if (q.kind === 0) P.push("Merge sort'un yolu son birleştirmeler (n + n/2 + … ≈ 2n) tarafından belirleniyor: birleştirme seri olduğu sürece paralellik log n mertebesinde kalır.");
    return {
      plots: [
        { title: "T_p ve sınırları", xlabel: "p", ylabel: "süre", side: true, ylog: true, size: { w: 300, h: 260 }, series: [{ name: "T_p (açgözlü)", x: ps, y: Ts, c: 0, dots: true }, { name: "max(W/p, S∞)", x: ps, y: ps.map((p) => Math.max(W / p, S)), c: 2, dash: true }, { name: "W/p + S∞", x: ps, y: ps.map((p) => W / p + S), c: 1, dash: true }] },
        { title: `Gantt şeması, p = ${q.p}${q.p > 16 ? " (ilk 16 işlemci)" : ""}`, xlabel: "zaman", ylabel: "işlemci", side: true, size: { w: 300, h: 260 }, ylim: [0, Math.min(q.p, 16)], series: [{ name: "görev (tek)", x: gx[0], y: gy[0], c: 0 }, { name: "görev (çift)", x: gx[1], y: gy[1], c: 2 }] },
      ],
      stats: [{ label: "İş W / yol S∞", value: `${W} / ${S}` }, { label: "Paralellik W/S∞", value: fmt(W / S, 3) }, { label: `T_p (p = ${q.p})`, value: cur.T }, { label: "Hızlanma / verim", value: `${fmt(sp, 3)} / ${fmt(100 * eff, 3)}%` }],
      note: P.join(" "),
    };
  },
};
