// M11.4 — Toplu (batch) MC ve toplu TD: aynı sonlu deneyim kümesi üzerinde yakınsayana dek tekrarlanan güncellemeler. Toplu MC'nin
// sabit noktası her durumun gözlenen getirilerinin ortalamasıdır (her ziyaret; eğitim verisinde kare hatayı en küçükler). Toplu TD(0)'ın
// sabit noktası, veriden en çok olabilirlikle kestirilen Markov modelinin (geçiş sayım oranları, ortalama ödüller) tam değeridir
// ("kesinlik eşdeğeri"). Veri: Sutton'ın A–B örneği (A→B ödül 0 bir bölüm; B'den 1 alan k bölüm, 0 alan 8 − 1 − k bölüm) ya da
// beş durumlu rastgele yürüyüşten E bölüm (R tekrar; gerçek değerlere RMS).
import { fmt, range } from "./plot.js";

const NS = 5, TRUE = range(NS).map((i) => (i + 1) / 6);
function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; if (Math.abs(M[c][c]) < 1e-14) continue; for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } } return M.map((r, i) => (Math.abs(r[i]) < 1e-14 ? NaN : r[n] / r[i])); }
// eps: [{s: [..], rw: [..]}], son durum −1 (terminal); γ = 1
export function batch(eps, n) {
  const G = Array(n).fill(0), cnt = Array(n).fill(0), C = range(n).map(() => Array(n).fill(0)), Rs = Array(n).fill(0), out = Array(n).fill(0);
  for (const e of eps) { let g = 0; const ret = []; for (let t = e.s.length - 1; t >= 0; t--) { g += e.rw[t]; ret[t] = g; }
    e.s.forEach((s, t) => { G[s] += ret[t]; cnt[s]++; Rs[s] += e.rw[t]; const nx = t + 1 < e.s.length ? e.s[t + 1] : -1; if (nx >= 0) C[s][nx]++; else out[s]++; }); }
  const mc = G.map((g, i) => (cnt[i] ? g / cnt[i] : NaN));
  const vis = range(n).filter((i) => cnt[i] > 0), A = vis.map((i) => vis.map((j) => (i === j ? 1 : 0) - C[i][j] / cnt[i])), b = vis.map((i) => Rs[i] / cnt[i]);
  const x = solve(A, b), td = Array(n).fill(NaN); vis.forEach((i, k) => (td[i] = x[k]));
  return { mc, td, cnt, C, out };
}
function rwEpisode(r) { const s = [2], rw = []; for (;;) { const c = s[s.length - 1], n = r() < 0.5 ? c - 1 : c + 1; if (n < 0) { rw.push(0); return { s, rw }; } if (n >= NS) { rw.push(1); return { s, rw }; } rw.push(0); s.push(n); } }
export default {
  params: [
    { id: "data", label: "Veri", options: ["A–B örneği (8 bölüm)", "rastgele yürüyüş"], value: 0 },
    { id: "k", label: "B'den 1 ödül alan bölüm sayısı", min: 0, max: 7, value: 6, int: true, on: { data: [0] } },
    { id: "E", label: "Bölüm sayısı", min: 1, max: 100, value: 20, int: true, on: { data: [1] } },
    { id: "R", label: "Tekrar (RMS ortalaması)", min: 1, max: 200, value: 100, int: true, on: { data: [1] } },
  ],
  seed: true,
  seedOn: { data: 1 },
  run(q, r) {
    if (q.data === 0) {
      const eps = [{ s: [0, 1], rw: [0, 0] }, ...range(q.k).map(() => ({ s: [1], rw: [1] })), ...range(7 - q.k).map(() => ({ s: [1], rw: [0] }))];
      const o = batch(eps, 2);
      const P = [`Veri: A, 0, B, 0 bir bölüm; B, 1 ${q.k} bölüm; B, 0 ${7 - q.k} bölüm. B'nin değerinde iki yöntem anlaşır: ${fmt(o.mc[1], 4)} (B'nin 8 gözleminin ortalaması).`];
      if (Math.abs(o.mc[0] - o.td[0]) < 1e-12) P.push("Bu veride V(B) = 0 olduğu için iki yöntem A'da da aynı cevabı veriyor; fark ancak B'nin değeri A'dan gözlenen getiriden farklıysa ortaya çıkar.");
      else P.push(`A'da ayrılırlar. Toplu MC, A'dan görülen tek getiriyi kullanır: V(A) = ${fmt(o.mc[0], 4)}. Toplu TD, A'dan her zaman B'ye geçildiğini (1/1) öğrenir ve V(A) = 0 + V(B) = ${fmt(o.td[0], 4)} verir.`);
      P.push("MC eğitim verisindeki getirilere en iyi uyan cevabı verir; TD verinin Markov yapısını kullanır ve gelecekteki veride genellikle daha az hata yapar (süreç gerçekten Markov ise).");
      return {
        plots: [{ title: "Toplu MC ve toplu TD tahminleri", xlabel: "", ylabel: "V", wide: true, size: { w: 620, h: 220 }, ylim: [0, 1.05], xlim: [-0.6, 1.6], xticks: [[0, "A"], [1, "B"]],
          series: [{ name: "toplu MC", x: [0, 1], y: o.mc, kind: "bar", c: 0, bw: 0.3, off: -0.5 }, { name: "toplu TD", x: [0, 1], y: o.td, kind: "bar", c: 2, bw: 0.3, off: 0.5 }] }],
        stats: [{ label: "V(A): MC / TD", value: `${fmt(o.mc[0], 4)} / ${fmt(o.td[0], 4)}` }, { label: "V(B)", value: fmt(o.mc[1], 4) }],
        note: P.join(" "),
      };
    }
    const Es = range(q.E).map((i) => i + 1), em = Array(q.E).fill(0), et = Array(q.E).fill(0);
    for (let k = 0; k < q.R; k++) {
      const eps = [];
      for (let e = 0; e < q.E; e++) {
        eps.push(rwEpisode(r));
        const o = batch(eps, NS), rms = (V) => Math.sqrt(V.reduce((a, v, i) => a + ((Number.isFinite(v) ? v : 0.5) - TRUE[i]) ** 2, 0) / NS);
        em[e] += rms(o.mc) / q.R; et[e] += rms(o.td) / q.R;
      }
    }
    const P = [`${q.R} tekrar ortalaması, ${q.E} bölümde RMS: toplu MC ${fmt(em[q.E - 1], 3)}, toplu TD ${fmt(et[q.E - 1], 3)}. Hiç ziyaret edilmemiş durumlar 0.5 sayıldı.`];
    const lower = em.filter((v, i) => et[i] < v).length;
    P.push(lower === q.E ? "Her veri miktarında toplu TD daha düşük hata veriyor: Markov yapısını kullanarak her geçişi, o geçişten sonra gelen bütün durumların tahminine bağlıyor." : `Toplu TD, ${q.E} veri miktarından ${lower} tanesinde daha düşük hata veriyor (tek bölümde iki yöntem aynı: her durumun tek getirisi ve tek geçişi aynı bilgiyi taşır).`);
    if (q.R < 10) P.push("Az tekrarla eğriler gürültülü; tek bir veri kümesinde MC şans eseri daha iyi olabilir.");
    return {
      plots: [{ title: "Toplu eğitimde RMS hata ve veri miktarı", xlabel: "bölüm sayısı", ylabel: "RMS", wide: true, size: { w: 620, h: 230 }, series: [{ name: "toplu MC", x: Es, y: em, c: 0 }, { name: "toplu TD", x: Es, y: et, c: 2 }] }],
      stats: [{ label: "RMS (MC / TD)", value: `${fmt(em[q.E - 1], 3)} / ${fmt(et[q.E - 1], 3)}` }],
      note: P.join(" "),
    };
  },
};
