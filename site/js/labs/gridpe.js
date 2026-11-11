// M11.3 — Sutton & Barto'nun 4 × 4 ızgara dünyası (Örnek 4.1): köşe hücreler (0 ve 15) son durum, dört belirlenimci eylem (ızgaradan
// çıkan eylem yerinde bırakır), her adım ödülü r. Rastgele politikanın (her eylem 1/4) yinelemeli değerlendirmesi: eşzamanlı (iki dizi:
// V_{k+1} = H_π V_k) ya da yerinde (Gauss–Seidel: yeni değerler aynı taramada hemen kullanılır). k taramadan sonraki V_k ısı haritası;
// açgözlü politika V_k'ye göre (en iyi olup olmadığı V*'a, yani değer yinelemesinin sonucuna göre denetlenir); iki yöntemin en büyük değişimi tarama sayısına göre. γ = 1 ve r = −1'de kesin değerler 0, −14, −20, −22, …
import { fmt, range } from "./plot.js";

const N = 4, TERM = [0, 15], MV = [[-1, 0], [1, 0], [0, -1], [0, 1]], AR = ["↑", "↓", "←", "→"];
const nxt = (s, a) => { const r = Math.floor(s / N), c = s % N, rr = r + MV[a][0], cc = c + MV[a][1]; return rr < 0 || rr >= N || cc < 0 || cc >= N ? s : rr * N + cc; };
function sweep(V, g, rw, inplace) {
  const W = inplace ? V : V.slice(); let d = 0;
  for (let s = 0; s < N * N; s++) { if (TERM.includes(s)) continue; let v = 0; for (let a = 0; a < 4; a++) v += 0.25 * (rw + g * (inplace ? W : V)[nxt(s, a)]); d = Math.max(d, Math.abs(v - V[s])); W[s] = v; }
  return { V: W, d };
}
export function evaluate(k, g, rw, inplace) { let V = Array(N * N).fill(0); const ds = []; for (let i = 0; i < k; i++) { const o = sweep(V, g, rw, inplace); V = o.V; ds.push(o.d); } return { V, ds }; }
export default {
  params: [
    { id: "k", label: "Tarama sayısı k", min: 0, max: 300, value: 3, int: true },
    { id: "mode", label: "Güncelleme", options: ["eşzamanlı (iki dizi)", "yerinde (Gauss–Seidel)"], value: 0 },
    { id: "g", label: "İndirim γ", min: 0.5, max: 1, step: 0.01, value: 1 },
    { id: "rw", label: "Adım ödülü r", min: -2, max: -0.1, step: 0.1, value: -1 },
  ],
  run(q) {
    const ip = q.mode === 1, { V } = evaluate(q.k, q.g, q.rw, ip), long = 400;
    const a = evaluate(long, q.g, q.rw, false), b = evaluate(long, q.g, q.rw, true), lo = Math.min(...a.V);
    const greedy = V.map((_, s) => (TERM.includes(s) ? "■" : (() => { const qs = range(4).map((x) => q.rw + q.g * V[nxt(s, x)]), m = Math.max(...qs); return range(4).filter((x) => qs[x] > m - 1e-9).map((x) => AR[x]).join(""); })()));
    let Vst = Array(N * N).fill(0); for (let i = 0; i < long; i++) Vst = Vst.map((v, s) => (TERM.includes(s) ? 0 : Math.max(...range(4).map((x) => q.rw + q.g * Vst[nxt(s, x)]))));
    const optimal = V.every((_, s) => TERM.includes(s) || (() => { const qs = range(4).map((x) => Vst[nxt(s, x)]), m = Math.max(...qs), mine = range(4).map((x) => V[nxt(s, x)]), mm = Math.max(...mine); return range(4).filter((x) => mine[x] > mm - 1e-9).every((x) => qs[x] > m - 1e-9); })());
    const conv = (ds) => { const i = ds.findIndex((d) => d < 1e-4); return i < 0 ? null : i + 1; }, ca = conv(a.ds), cb = conv(b.ds);
    const rows = range(N).map((r) => range(N).map((c) => fmt(V[r * N + c], 3)).join("  ")).join(" | ");
    const P = [`k = ${q.k} taramadan sonra V (satır satır): ${rows}.`, `V_k'ye göre açgözlü politika: ${range(N).map((r) => range(N).map((c) => greedy[r * N + c]).join(" ")).join(" | ")}${optimal ? " — en iyi politikadır" : ""}.`];
    P.push(`En büyük değişimin 10⁻⁴'ün altına inmesi: eşzamanlı ${ca ?? "—"}, yerinde ${cb ?? "—"} tarama.`);
    if (optimal && (ip ? cb : ca) && q.k < (ip ? cb : ca)) P.push("Değerler henüz yakınsamadan (rastgele politikanın değerleri bile) açgözlü politika en iyi politikayı veriyor: tek bir politika iyileştirme adımı yetiyor.");
    if (q.g === 1 && q.rw === -1 && q.k >= 300) P.push("Kesin değerler: 0, −14, −20, −22 / −14, −18, −20, −20 / … : hücreden son duruma rastgele yürüyüşle beklenen adım sayısının eksi işaretlisi.");
    return {
      plots: [
        { title: `V_k ısı haritası (beyaz = 0, siyah = ${fmt(Math.min(...V), 3)})`, side: true, size: { w: 260, h: 260 }, img: { w: N, h: N, data: V.map((v) => 1 - v / Math.min(Math.min(...V), -1e-9)) } },
        { title: "En büyük değişim ve tarama (log)", xlabel: "tarama", ylabel: "max |ΔV|", side: true, size: { w: 330, h: 260 }, ylog: true,
          series: [{ name: "eşzamanlı", x: range(Math.min(long, 200)).map((i) => i + 1), y: a.ds.slice(0, 200).map((d) => Math.max(d, 1e-12)), c: 0 }, { name: "yerinde", x: range(Math.min(long, 200)).map((i) => i + 1), y: b.ds.slice(0, 200).map((d) => Math.max(d, 1e-12)), c: 2 }], marks: q.k > 0 && q.k <= 200 ? [{ x: q.k, label: "k" }] : [] },
      ],
      stats: [{ label: "V(1) (son durumun yanı)", value: fmt(V[1], 4) }, { label: "V(3) (karşı köşe)", value: fmt(V[3], 4) }, { label: "Açgözlü politika en iyi mi", value: optimal ? "evet" : "hayır" }],
      note: P.join(" "),
    };
  },
};
