// M11.11 — AlphaGo Zero'nun kökteki iki keşif aracı. (1) Sıcaklık: kendine karşı oyunda hamle, kökteki ziyaret sayılarından
// π(a) ∝ N(a)^{1/τ} ile örneklenir (ilk 30 hamlede τ = 1, sonra τ → 0). (2) Dirichlet gürültüsü: kök önseli P = (1 − ε)p + εη,
// η ~ Dir(α) (Go'da α = 0.03, ε = 0.25). Örnek bir kök: K yasal hamle, ağın önseli p'de tek bir hamle baskın, "gizli iyi hamle"nin
// önseli çok küçük (0.1 %). PUCT seçimi Q + c·P·√N/(1 + n) ile kökte S simülasyon yapılır; çocukların Q değerleri sabit kabul edilir
// (gizli hamle gerçekte en iyisidir, değeri +0.2; önselin baskın hamlesi 0, diğerleri −0.3…−0.1; değerler [−1, 1] ölçeğinde). Ziyaret
// edilmemiş çocuğun Q'su 0 kabul edilir, her ziyaret gürültülü bir değerlendirme ekler. Tohumlu tekrarlarda gizli hamlenin payı ölçülür.
import { fmt, range } from "./plot.js";

function gamma(r, a) { if (a < 1) return gamma(r, a + 1) * Math.pow(r(), 1 / a); const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { let x, v; do { x = r.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
const dirichlet = (r, K, a) => { const g = range(K).map(() => gamma(r, a)), s = g.reduce((x, y) => x + y, 0); return g.map((v) => v / s); };
// Ziyaret edilmemiş çocuğun Q'su 0 (AlphaGo Zero'daki gibi); her ziyaret gerçek değerin gürültülü (σ = 0.5) bir değerlendirmesini ekler.
export function rootSearch(P, V, S, c, r) { const K = P.length, n = Array(K).fill(0), W = Array(K).fill(0); for (let t = 0; t < S; t++) { let b = 0, bv = -Infinity; for (let a = 0; a < K; a++) { const Qa = n[a] ? W[a] / n[a] : 0, u = Qa + c * P[a] * Math.sqrt(Math.max(t, 1)) / (1 + n[a]); if (u > bv) { bv = u; b = a; } } n[b]++; W[b] += Math.max(-1, Math.min(1, V[b] + 0.5 * r.normal())); } return n; }
export default {
  params: [
    { id: "K", label: "Yasal hamle sayısı K", min: 5, max: 100, value: 30, int: true },
    { id: "eps", label: "Gürültü payı ε", min: 0, max: 0.5, step: 0.05, value: 0.25 },
    { id: "alpha", label: "Dirichlet α", min: 0.01, max: 1, step: 0.01, value: 0.03, log: true },
    { id: "S", label: "Kök simülasyonu S", min: 50, max: 1600, step: 50, value: 400, int: true },
    { id: "tau", label: "Sıcaklık τ", min: 0.05, max: 2, step: 0.05, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const K = q.K, hid = K - 1, c = 1.5;
    const p = range(K).map((a) => (a === 0 ? 0.6 : a === hid ? 0.001 : 0.399 / (K - 2)));
    const V = range(K).map((a) => (a === hid ? 0.2 : a === 0 ? 0 : -0.3 + (0.2 * a) / K));
    const R = 200; let share = 0, tried = 0, pick = 0; let lastN = null, lastP = null;
    for (let k = 0; k < R; k++) {
      const eta = dirichlet(r, K, q.alpha), P = p.map((v, a) => (1 - q.eps) * v + q.eps * eta[a]), n = rootSearch(P, V, q.S, c, r);
      share += n[hid] / q.S / R; if (n[hid] > 0) tried += 1 / R;
      const w = n.map((v) => Math.pow(v, 1 / q.tau)), s = w.reduce((x, y) => x + y, 0); pick += w[hid] / s / R;
      if (k === R - 1) { lastN = n; lastP = P; }
    }
    let baseShare = 0; for (let k = 0; k < 50; k++) baseShare += rootSearch(p, V, q.S, c, r)[hid] / q.S / 50;
    const P0 = [`Gürültüsüz önselle ${q.S} simülasyonda gizli iyi hamlenin ortalama ziyaret payı ${fmt(100 * baseShare, 3)}% (önseli %0.1, gerçek değeri en yüksek).`];
    P0.push(`Gürültüyle (ε = ${q.eps}, α = ${q.alpha}) ${R} tekrarda: gizli hamle ortalama ziyaret payı ${fmt(100 * share, 3)}%, en az bir kez denenme olasılığı ${fmt(100 * tried, 3)}%, τ = ${q.tau} ile kökte seçilme olasılığı ${fmt(100 * pick, 3)}%.`);
    P0.push(`α gürültünün biçimini belirler. Küçük α (0.03) gürültüyü birkaç hamleye yığar: gizli hamle ancak çekilişte büyük pay alan birkaç hamleden biriyse öne çıkar, çoğu oyunda çıkmaz. Büyük α gürültüyü düzgüne yaklaştırır ve her hamleye ≈ ε/K ekler. Go'da α = 0.03 ≈ 10/K seçilmişti (K ≈ 361); bu ufak tahtada daha büyük α gizli hamleyi daha sık buluyor.`);
    if (q.eps === 0) P0.push("ε = 0: gürültü yok; önselin neredeyse sıfır verdiği hamle, değeri en yüksek olsa da PUCT'un bonusu tarafından nadiren deneniyor. Ağ bu hamleyi hiç öğrenemezse kendine karşı oyun onu da hiç denemez: kendini besleyen bir kör nokta.");
    if (q.tau <= 0.1) P0.push("τ → 0: hamle en çok ziyaret edilene kilitlenir; kendine karşı oyunda çeşitlilik yalnız gürültüden gelir.");
    const xs = range(K);
    return {
      plots: [
        { title: "Son tekrarda kök önseli: ağ (yeşil) ve gürültülü (mavi)", xlabel: "hamle (sonuncusu gizli iyi hamle)", ylabel: "olasılık", side: true, size: { w: 320, h: 230 }, ylim: [0, 0.7], xlim: [-1, q.K],
          series: [{ name: "ağın önseli p", x: xs, y: p, kind: "bar", c: 1, bw: 0.45, off: -0.5 }, { name: "gürültülü P", x: xs, y: lastP, kind: "bar", c: 0, bw: 0.45, off: 0.5 }] },
        { title: `Ziyaret sayıları N ve π ∝ N^(1/τ), τ = ${q.tau}`, xlabel: "hamle", ylabel: "pay", side: true, size: { w: 300, h: 230 }, ylim: [0, 1], xlim: [-1, q.K],
          series: [{ name: "N / S", x: xs, y: lastN.map((v) => v / q.S), kind: "bar", c: 0, bw: 0.45, off: -0.5 }, { name: "π", x: xs, y: (() => { const w = lastN.map((v) => Math.pow(v, 1 / q.tau)), s = w.reduce((x, y) => x + y, 0); return w.map((v) => v / s); })(), kind: "bar", c: 2, bw: 0.45, off: 0.5 }] },
      ],
      stats: [{ label: "Gizli hamle: denenme olasılığı", value: `${fmt(100 * tried, 3)}%` }, { label: "Ortalama ziyaret payı", value: `${fmt(100 * share, 3)}%` }, { label: "Kökte seçilme olasılığı", value: `${fmt(100 * pick, 3)}%` }],
      note: P0.join(" "),
    };
  },
};
