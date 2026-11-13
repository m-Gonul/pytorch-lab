// M11.11 — Tablo tabanlı mini AlphaZero, tic-tac-toe. "Ağ" bir tablodur: her konum için p (9 hamle olasılığı, başta yasal hamlelere
// düzgün) ve v (sıradaki oyuncunun bakışıyla, başta 0). Kendine karşı oyun: her hamlede PUCT araması (S simülasyon, c = 1.5, yaprakta
// tablo değeri, son konumda kesin sonuç, kökte ε = 0.25 / α = 0.3 Dirichlet), ilk 4 hamlede τ = 1 ile ziyaret sayılarından örnekleme,
// sonra en çok ziyaret edilen. Oyun sonunda ziyaret edilen her konumda p ← p + η(π − p), v ← v + η(z − v) (z sıradaki oyuncu için
// +1/0/−1). Değerlendirme: bütün erişilebilir konumlarda (minimax ile) tablonun açgözlü hamlesinin en iyi olma oranı ve değer hatası;
// ayrıca aramasız tablo politikasının mükemmel oyuncuya karşı skoru.
import { fmt, range } from "./plot.js";

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const win = (b) => { for (const [x, y, z] of LINES) if (b[x] && b[x] === b[y] && b[x] === b[z]) return b[x]; return b.includes(0) ? 0 : 2; }; // 1/−1 kazanan, 2 beraberlik
const key = (b) => b.join(",");
// minimax: sıradaki oyuncu p için değer (+1 kazanır)
const MM = new Map();
function mm(b, p) { const k = key(b) + p; if (MM.has(k)) return MM.get(k); const w = win(b); let v; if (w === 2) v = 0; else if (w) v = w === p ? 1 : -1; else { v = -2; for (let i = 0; i < 9; i++) if (!b[i]) { b[i] = p; v = Math.max(v, -mm(b, -p)); b[i] = 0; } } MM.set(k, v); return v; }
const STATES = []; (function walk(b, p, seen) { const k = key(b); if (seen.has(k) || win(b)) return; seen.add(k); STATES.push([b.slice(), p]); for (let i = 0; i < 9; i++) if (!b[i]) { b[i] = p; walk(b, -p, seen); b[i] = 0; } })(Array(9).fill(0), 1, new Set());
function gamma(r, a) { if (a < 1) return gamma(r, a + 1) * Math.pow(r(), 1 / a); const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = r.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
export function train(q, r, checkpoints) {
  const T = new Map(); const get = (b) => { const k = key(b); if (!T.has(k)) { const legal = range(9).filter((i) => !b[i]); const p = Array(9).fill(0); legal.forEach((i) => (p[i] = 1 / legal.length)); T.set(k, { p, v: 0 }); } return T.get(k); };
  const curve = [];
  const seen = new Set();
  const evalNow = (g) => { let ok = 0, err = 0, okS = 0, nS = 0; for (const [b, p] of STATES) { const t = get(b), legal = range(9).filter((i) => !b[i]); let bi = legal[0]; for (const i of legal) if (t.p[i] > t.p[bi]) bi = i; const best = mm(b, p); b[bi] = p; const vv = -mm(b, -p); b[bi] = 0; if (vv === best) ok++; err += Math.abs(t.v - best); if (seen.has(key(b))) { nS++; if (vv === best) okS++; } } curve.push([g, ok / STATES.length, err / STATES.length, nS ? okS / nS : NaN, nS]); };
  function search(b0, p0, root) {
    const N = new Map(), W = new Map(); const st = (b) => key(b);
    for (let s = 0; s < q.S; s++) {
      const b = b0.slice(); let p = p0; const path = []; let val;
      for (;;) {
        const w = win(b); if (w) { val = w === 2 ? 0 : w === p ? 1 : -1; break; }
        const k = st(b), t = get(b), legal = range(9).filter((i) => !b[i]);
        if (!N.has(k)) { N.set(k, Array(9).fill(0)); W.set(k, Array(9).fill(0)); val = t.v; break; }
        const n = N.get(k), w2 = W.get(k), tot = n.reduce((x, y) => x + y, 0); let P = t.p;
        if (path.length === 0 && root.noise) P = P.map((v, i) => (b[i] ? 0 : 0.75 * v + 0.25 * root.noise[i]));
        let a = legal[0], bv = -Infinity; for (const i of legal) { const Q = n[i] ? w2[i] / n[i] : 0, u = Q + 1.5 * P[i] * Math.sqrt(Math.max(tot, 1)) / (1 + n[i]); if (u > bv) { bv = u; a = i; } }
        path.push([k, a]); b[a] = p; p = -p;
      }
      // val: değerin sahibi o anda sırası gelen oyuncu; geriye doğru işaret değiştir
      for (let j = path.length - 1; j >= 0; j--) { val = -val; const [k, a] = path[j]; N.get(k)[a]++; W.get(k)[a] += val; }
    }
    return N.get(st(b0)) || Array(9).fill(0);
  }
  evalNow(0); const step = Math.max(1, Math.floor(q.G / 20));
  for (let g = 1; g <= q.G; g++) {
    const b = Array(9).fill(0); let p = 1; const hist = [];
    while (!win(b)) {
      const legal = range(9).filter((i) => !b[i]); const gs = legal.map(() => gamma(r, 0.3)), gsum = gs.reduce((x, y) => x + y, 0), noise = Array(9).fill(0); legal.forEach((i, j) => (noise[i] = gs[j] / gsum));
      const n = q.S > 0 ? search(b, p, { noise }) : get(b).p.map((v) => v * 100), tot = n.reduce((x, y) => x + y, 0), pi = n.map((v) => v / tot);
      let a; if (hist.length < 4) { let u = r(); a = legal[legal.length - 1]; for (const i of legal) { u -= pi[i]; if (u <= 0) { a = i; break; } } } else { a = legal[0]; for (const i of legal) if (n[i] > n[a]) a = i; }
      hist.push([b.slice(), p, pi]); seen.add(key(b)); b[a] = p; p = -p;
    }
    const w = win(b); for (const [hb, hp, pi] of hist) { const t = get(hb), z = w === 2 ? 0 : w === hp ? 1 : -1; for (let i = 0; i < 9; i++) t.p[i] += q.eta * (pi[i] - t.p[i]); t.v += q.eta * (z - t.v); }
    if (g % step === 0 || g === q.G) evalNow(g);
  }
  return { curve, T, get };
}
export default {
  params: [
    { id: "G", label: "Kendine karşı oyun sayısı", min: 0, max: 3000, step: 100, value: 1000, int: true },
    { id: "S", label: "Hamle başına arama simülasyonu S (0 = aramasız)", min: 0, max: 100, step: 5, value: 25, int: true },
    { id: "eta", label: "Tablo öğrenme oranı η", min: 0.05, max: 1, step: 0.05, value: 0.3 },
  ],
  seed: true,
  run(q, r) {
    const { curve } = train(q, r), last = curve[curve.length - 1];
    const P = [`${q.G} oyundan sonra, erişilebilir ${STATES.length} konum için tablonun açgözlü hamlesinin minimax'a göre en iyi hamlelerden biri olma oranı ${fmt(100 * last[1], 3)}%; ortalama değer hatası |v − v*| = ${fmt(last[2], 3)}. Başlangıçta: ${fmt(100 * curve[0][1], 3)}% ve ${fmt(curve[0][2], 3)}.`];
    if (last[4]) P.push(`Kendine karşı oyunlarda görülen ${last[4]} konumda bu oran ${fmt(100 * last[3], 3)}%.`);
    P.push(q.S === 0 ? "S = 0: arama yok; hamleler tablonun kendi politikasından seçiliyor ve politika hedefi kendi kopyası (π = p): politika hiç iyileşmez. Değerler yalnız oyun sonuçlarıyla öğreniliyor. Arama, politika iyileştirme operatörüdür; onsuz kendine karşı oyun yerinde sayar." : `Her hamlede ${q.S} simülasyonluk arama, tablonun p'sinden daha iyi bir π üretiyor; tablo π'yi taklit edip z'ye uydukça arama da iyileşiyor (politika yinelemesinin örneklemeli biçimi).`);
    P.push("Not: oranlar bütün erişilebilir konumlar üzerinden; bunların çoğu iyi oyunda hiç görülmez, bu yüzden oran %100'e ulaşmayabilir.");
    return {
      plots: [{ title: "Tablonun açgözlü politikası: en iyi hamle oranı ve değer hatası", xlabel: "kendine karşı oyun", ylabel: "oran / hata", wide: true, size: { w: 620, h: 240 }, ylim: [0, 1],
        series: [{ name: "en iyi hamle oranı", x: curve.map((c) => c[0]), y: curve.map((c) => c[1]), c: 0, dots: true }, { name: "aynı oran, görülen konumlarda", x: curve.map((c) => c[0]), y: curve.map((c) => c[3]), c: 1, dots: true }, { name: "ortalama |v − v*|", x: curve.map((c) => c[0]), y: curve.map((c) => c[2]), c: 2, dots: true }] }],
      stats: [{ label: "En iyi hamle oranı", value: `${fmt(100 * last[1], 3)}%` }, { label: "Değer hatası", value: fmt(last[2], 3) }, { label: "Görülen konumlarda oran", value: last[4] ? `${fmt(100 * last[3], 3)}% (${last[4]} konum)` : "—" }],
      note: P.join(" "),
    };
  },
};
