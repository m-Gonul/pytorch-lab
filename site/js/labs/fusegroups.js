// M6.5 — Füzyon grupları ve taşınan bayt (aşama c'nin modeli). Graf: (ad, işlem, girdiler); eleman bazlı işlemler (add, mul, tanh, exp)
// ve indirgeme (sum, skaler çıktı). Kurallar: (0) füzyon yok; (1) aşama c: eleman bazlı düğüm, eleman bazlı girdisinin grubuna katılır,
// eğer o girdinin tek tüketicisi bu düğümse ve girdi bir çıkış değilse; sum kendi grubunda; (2) Inductor benzeri: tüketicilerinin hepsi
// eleman bazlıysa üretici onlarla birleşir (yatay füzyon) ve tek tüketicisi indirgeme olan eleman bazlı üretici indirgemeye katılır.
// Her grup dışarıdan okuduğu değerlerin baytını okur, dışarıda kullanılan ya da çıkış olan değerlerinin baytını yazar.
import { fmt, range } from "./plot.js";

const G = [
  { n: "GELU benzeri: x·(1 + tanh(x + x³))", nodes: [["x", "input", []], ["x2", "mul", ["x", "x"]], ["x3", "mul", ["x2", "x"]], ["s", "add", ["x", "x3"]], ["t", "tanh", ["s"]], ["u", "add", ["t", 1]], ["y", "mul", ["x", "u"]]], out: ["y"] },
  { n: "iki tüketici: t = x·w; y = tanh(t) + exp(t)", nodes: [["x", "input", []], ["w", "input", []], ["t", "mul", ["x", "w"]], ["a", "tanh", ["t"]], ["b", "exp", ["t"]], ["y", "add", ["a", "b"]]], out: ["y"] },
  { n: "indirgeme: z = Σ exp(x·w), y = x + z", nodes: [["x", "input", []], ["w", "input", []], ["t", "mul", ["x", "w"]], ["e", "exp", ["t"]], ["z", "sum", ["e"]], ["y", "add", ["x", "z"]]], out: ["y"] },
  { n: "ara çıktı: t = tanh(x), y = exp(t), t de döndürülüyor", nodes: [["x", "input", []], ["t", "tanh", ["x"]], ["y", "exp", ["t"]]], out: ["y", "t"] },
];
const EW = new Set(["add", "mul", "tanh", "exp"]);
export function groups(g, rule) {
  const nodes = g.nodes.filter((n) => n[1] !== "input"), op = Object.fromEntries(g.nodes.map((n) => [n[0], n[1]]));
  const cons = {}; g.nodes.forEach(([nm, , args]) => args.forEach((a) => { if (typeof a === "string") (cons[a] = cons[a] || []).push(nm); }));
  const par = Object.fromEntries(nodes.map((n) => [n[0], n[0]])), find = (a) => (par[a] === a ? a : (par[a] = find(par[a]))), uni = (a, b) => { par[find(a)] = find(b); };
  if (rule === 1) for (const [nm, o, args] of nodes) { if (!EW.has(o)) continue; for (const a of args) if (typeof a === "string" && EW.has(op[a]) && (cons[a] || []).length === 1 && !g.out.includes(a)) uni(nm, a); }
  if (rule === 2) for (const [nm, o] of nodes) {
    const cs = cons[nm] || [];
    if (EW.has(o) && cs.length && cs.every((c) => EW.has(op[c]))) cs.forEach((c) => uni(c, nm));
    if (EW.has(o) && cs.length === 1 && op[cs[0]] === "sum") uni(nm, cs[0]);
  }
  const by = {}; nodes.forEach(([nm]) => (by[find(nm)] = by[find(nm)] || []).push(nm));
  return Object.values(by);
}
export function bytesMoved(g, grs, N, b) {
  const op = Object.fromEntries(g.nodes.map((n) => [n[0], n[1]])), size = (v) => (op[v] === "sum" ? 1 : N) * b;
  const cons = {}; g.nodes.forEach(([nm, , args]) => args.forEach((a) => { if (typeof a === "string") (cons[a] = cons[a] || []).push(nm); }));
  return grs.map((gr) => {
    const S = new Set(gr), reads = new Set(); gr.forEach((nm) => g.nodes.find((n) => n[0] === nm)[2].forEach((a) => { if (typeof a === "string" && !S.has(a)) reads.add(a); }));
    const writes = gr.filter((nm) => g.out.includes(nm) || (cons[nm] || []).some((c) => !S.has(c)));
    return { gr, r: [...reads].reduce((s, v) => s + size(v), 0), w: writes.reduce((s, v) => s + size(v), 0), reads: [...reads], writes };
  });
}
const RULES = ["füzyon yok", "aşama c kuralı (tek tüketici)", "Inductor benzeri (yatay + indirgemeye katma)"];
export default {
  params: [
    { id: "g", label: "Graf", options: G.map((x) => x.n), value: 0 },
    { id: "rule", label: "Füzyon kuralı", options: RULES, value: 1 },
    { id: "logN", label: "Eleman sayısı: log₂ N", min: 10, max: 28, value: 20, int: true },
    { id: "dt", label: "Veri tipi", options: ["float32 (4 bayt)", "bf16 (2 bayt)"], value: 0 },
  ],
  run(q) {
    const g = G[q.g], N = 2 ** q.logN, b = q.dt ? 2 : 4, all = [0, 1, 2].map((r) => bytesMoved(g, groups(g, r), N, b)), cur = all[q.rule];
    const tot = (bm) => bm.reduce((s, x) => s + x.r + x.w, 0), T = all.map(tot);
    const desc = cur.map((x, i) => `çekirdek ${i + 1} {${x.gr.join(", ")}}: okur ${x.reads.join(", ") || "—"}, yazar ${x.writes.join(", ")}`).join("; ");
    const P = [`${RULES[q.rule]}: ${cur.length} çekirdek, toplam ${fmt(T[q.rule] / 2 ** 20, 4)} MiB trafik. ${desc}.`];
    P.push(`Üç kural: ${T.map((t, i) => `${RULES[i]} ${fmt(t / 2 ** 20, 4)} MiB`).join(", ")}. Füzyonsuza göre kazanç ${fmt(T[0] / T[q.rule], 3)} kat.`);
    if (q.g === 1 && q.rule === 1) P.push("t'nin iki tüketicisi olduğu için aşama c kuralı onu kendi grubunda bırakır ve t belleğe yazılıp iki kez okunur. Yatay füzyon tanh ve exp'i aynı çekirdekte hesaplayarak t'yi hiç yazmaz.");
    if (q.g === 2 && q.rule === 2) P.push("exp, tek tüketicisi olan toplamın çekirdeğine katıldı: e hiç belleğe yazılmaz (giriş tarafı füzyonu). x hem toplam hem son toplama için iki kez okunur.");
    if (q.g === 3) P.push("t bir çıktı olduğu için her durumda belleğe yazılmalıdır; füzyon yalnız onu yeniden okumayı engelleyebilir.");
    return {
      plots: [{ title: "Çekirdek başına okunan ve yazılan (MiB)", xlabel: "çekirdek", ylabel: "MiB", yzero: true, xlim: [-0.6, cur.length - 0.4], series: [
        { name: "okunan", x: range(cur.length).map((i) => i - 0.15), y: cur.map((x) => x.r / 2 ** 20), kind: "bar", c: 0, bw: 0.3 },
        { name: "yazılan", x: range(cur.length).map((i) => i + 0.15), y: cur.map((x) => x.w / 2 ** 20), kind: "bar", c: 1, bw: 0.3 }],
        xticks: range(cur.length).map((i) => [i, String(i + 1)]) }],
      stats: [
        { label: "Çekirdek sayısı", value: cur.length },
        { label: "Toplam trafik (MiB): yok / c / Inductor", value: T.map((t) => fmt(t / 2 ** 20, 4)).join(" / ") },
        { label: "Füzyonsuza göre kazanç", value: T[0] / T[q.rule] },
      ],
      note: P.join(" "),
    };
  },
};
