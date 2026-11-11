// M11.4 — Sutton & Barto Örnek 6.2: beş durumlu rastgele yürüyüş (A–E, C'den başlanır, her adım sola/sağa 1/2; soldan çıkış ödül 0,
// sağdan çıkış ödül 1, γ = 1). Gerçek değerler 1/6, 2/6, …, 5/6. Rastgele politikanın değerlendirmesi: her ziyaret Monte Carlo
// (bölüm sonu getirisine doğru) ve TD(0) (r + V(s')'ye doğru), sabit α ya da α = 1/n(s). Başlangıç V = 0.5. R bağımsız çalıştırmanın
// ortalama RMS hatası bölüm sayısına göre; ayrıca son çalıştırmanın tahminleri.
import { fmt, range } from "./plot.js";

const NS = 5, TRUE = range(NS).map((i) => (i + 1) / 6);
function episode(r) { const s = [2], rw = []; for (;;) { const c = s[s.length - 1], n = r() < 0.5 ? c - 1 : c + 1; if (n < 0) { rw.push(0); return { s, rw, end: true }; } if (n >= NS) { rw.push(1); return { s, rw, end: true }; } rw.push(0); s.push(n); } }
export function run1(q, r, E) {
  const Vm = Array(NS).fill(0.5), Vt = Array(NS).fill(0.5), nm = Array(NS).fill(0), nt = Array(NS).fill(0), em = [], et = [];
  const rms = (V) => Math.sqrt(V.reduce((a, v, i) => a + (v - TRUE[i]) ** 2, 0) / NS);
  em.push(rms(Vm)); et.push(rms(Vt));
  for (let e = 0; e < E; e++) {
    const { s, rw } = episode(r), T = s.length;
    // TD(0)
    for (let t = 0; t < T; t++) { const st = s[t], nx = t + 1 < T ? Vt[s[t + 1]] : 0; nt[st]++; const a = q.sched === 1 ? 1 / nt[st] : q.at; Vt[st] += a * (rw[t] + nx - Vt[st]); }
    // her ziyaret MC: getiri = bölümün son ödülü (γ = 1, ara ödüller 0)
    const G = rw[T - 1];
    for (let t = 0; t < T; t++) { const st = s[t]; nm[st]++; const a = q.sched === 1 ? 1 / nm[st] : q.am; Vm[st] += a * (G - Vm[st]); }
    em.push(rms(Vm)); et.push(rms(Vt));
  }
  return { em, et, Vm, Vt };
}
export default {
  params: [
    { id: "sched", label: "Öğrenme oranı", options: ["sabit α", "α = 1/n(s) (örnek ortalaması)"], value: 0 },
    { id: "am", label: "MC için α", min: 0.005, max: 0.2, step: 0.005, value: 0.02, on: { sched: [0] } },
    { id: "at", label: "TD için α", min: 0.005, max: 0.3, step: 0.005, value: 0.1, on: { sched: [0] } },
    { id: "E", label: "Bölüm sayısı", min: 10, max: 300, step: 10, value: 100, int: true },
    { id: "R", label: "Ortalaması alınan çalıştırma", min: 1, max: 200, value: 100, int: true },
  ],
  seed: true,
  run(q, r) {
    let sm = Array(q.E + 1).fill(0), stt = Array(q.E + 1).fill(0), last = null;
    for (let k = 0; k < q.R; k++) { const o = run1(q, r, q.E); o.em.forEach((v, i) => (sm[i] += v / q.R)); o.et.forEach((v, i) => (stt[i] += v / q.R)); last = o; }
    const xs = range(q.E + 1), P = [`${q.R} çalıştırma ortalaması, ${q.E} bölüm sonunda RMS hata: MC ${fmt(sm[q.E], 3)}, TD ${fmt(stt[q.E], 3)}.`];
    const cross = xs.find((i) => i > 0 && stt[i] < sm[i]);
    if (q.sched === 0) P.push(`Sabit α'da hata sıfıra inmez: tahminler gerçek değer etrafında α ile orantılı bir gürültüyle dolaşır (ör. TD'de α = ${q.at}).`);
    else P.push("α = 1/n(s) ile MC her durumun getirilerinin örnek ortalamasıdır ve yansızdır; TD'de 1/n, ilk bölümlerin yanlı bootstrapping hedeflerini uzun süre ağırlıkta tutar ve yavaşlar.");
    if (cross !== undefined && stt[q.E] < sm[q.E]) P.push(`TD ${cross}. bölümden itibaren MC'den daha düşük hatada.`);
    if (q.sched === 0 && q.at >= 0.15) P.push("Büyük α'da TD hızlı iner ama yüksek bir hata tabanında durur.");
    return {
      plots: [
        { title: "Ortalama RMS hata (gerçek değerlere göre)", xlabel: "bölüm", ylabel: "RMS", side: true, size: { w: 320, h: 240 }, ylim: [0, 0.3], series: [{ name: "MC", x: xs, y: sm, c: 0 }, { name: "TD(0)", x: xs, y: stt, c: 2 }] },
        { title: "Son çalıştırmanın tahminleri", xlabel: "durum", ylabel: "V", side: true, size: { w: 280, h: 240 }, ylim: [0, 1], xticks: range(NS).map((i) => [i, "ABCDE"[i]]),
          series: [{ name: "gerçek", x: range(NS), y: TRUE, c: 4, dash: true }, { name: "MC", x: range(NS), y: last.Vm, c: 0, dots: true }, { name: "TD", x: range(NS), y: last.Vt, c: 2, dots: true }] },
      ],
      stats: [{ label: "RMS (MC / TD)", value: `${fmt(sm[q.E], 3)} / ${fmt(stt[q.E], 3)}` }, { label: "V(C) tahmini (MC / TD), gerçek 0.5", value: `${fmt(last.Vm[2], 3)} / ${fmt(last.Vt[2], 3)}` }],
      note: P.join(" "),
    };
  },
};
