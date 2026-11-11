// M11.1 — Durağan (zamanla homojen) birinci dereceden Markov zinciri, sütun-stokastik geçiş matrisi: p_{t+1} = T p_t (sütun j, S_j'den
// çıkış olasılıkları). Zincirler: e-ticaret oturumu (ders örneği, tek soğurucu durum "satın alma"), terk durumu eklenmiş e-ticaret (q: her
// adımda siteden çıkma olasılığı), iki durumlu hava (a, b: kalma olasılıkları), üç durumlu döngü (ε: tembellik; ε = 0 periyodik).
// Kesin dağılım p_t matris–vektör çarpımıyla; N örnek yol tohumlu üreteçle simüle edilir ve durum oranları p_t ile karşılaştırılır.
// Durağan dağılım (T − I)π = 0, Σπ = 1 doğrusal sisteminden; soğurucu zincirlerde temel matris N = (I − Q)⁻¹ ile soğurulma olasılıkları
// ve beklenen adım sayısı.
import { fmt, range } from "./plot.js";

const NAMES = ["e-ticaret (ders örneği)", "e-ticaret + terk durumu", "iki durumlu hava", "üç durumlu döngü"];
export function chain(q) {
  if (q.ch === 0) return { S: ["H", "P", "C", "Pur"], T: [[0.3, 0.2, 0.1, 0], [0.6, 0.4, 0.1, 0], [0.1, 0.15, 0.5, 0], [0, 0.25, 0.3, 1]], abs: [3] };
  if (q.ch === 1) {
    const k = 1 - q.q, b = [[0.3, 0.2, 0.1], [0.6, 0.4, 0.1], [0.1, 0.15, 0.5], [0, 0.25, 0.3]];
    const T = range(5).map((i) => range(5).map((j) => (j >= 3 ? (i === j ? 1 : 0) : i < 4 ? k * b[i][j] : q.q)));
    return { S: ["H", "P", "C", "Pur", "Çık"], T, abs: [3, 4] };
  }
  if (q.ch === 2) return { S: ["güneş", "yağmur"], T: [[q.a, 1 - q.b], [1 - q.a, q.b]], abs: [] };
  const e = q.eps, c = 1 - e;
  return { S: ["A", "B", "C"], T: [[e, 0, c], [c, e, 0], [0, c, e]], abs: [] };
}
const mv = (T, p) => T.map((row) => row.reduce((s, v, j) => s + v * p[j], 0));
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
export function stationary(T) { const n = T.length, A = T.map((r, i) => r.map((v, j) => v - (i === j ? 1 : 0))); A[n - 1] = Array(n).fill(1); const b = Array(n).fill(0); b[n - 1] = 1; return solve(A, b); }
export function absorb(T, abs, s0) {
  const tr = range(T.length).filter((i) => !abs.includes(i)), m = tr.length;
  // sütun-stokastik: Q[i][j] = T[tr_i][tr_j]; soğurulma olasılığı x_j = P(sonunda a | j'den başla) satır vektörü için (I − Qᵀ) x = R_aᵀ
  const A = range(m).map((i) => range(m).map((j) => (i === j ? 1 : 0) - T[tr[j]][tr[i]]));
  const probs = abs.map((a) => solve(A, tr.map((j) => T[a][j])));
  const steps = solve(A, Array(m).fill(1));
  const k = tr.indexOf(s0);
  return { p: probs.map((v) => v[k]), steps: steps[k] };
}
export default {
  params: [
    { id: "ch", label: "Zincir", options: NAMES, value: 0 },
    { id: "q", label: "Her adımda terk olasılığı q", min: 0, max: 0.5, step: 0.01, value: 0.1, on: { ch: [1] } },
    { id: "a", label: "P(güneş → güneş)", min: 0, max: 1, step: 0.01, value: 0.9, on: { ch: [2] } },
    { id: "b", label: "P(yağmur → yağmur)", min: 0, max: 1, step: 0.01, value: 0.5, on: { ch: [2] } },
    { id: "eps", label: "Tembellik ε (yerinde kalma)", min: 0, max: 0.9, step: 0.01, value: 0, on: { ch: [3] } },
    { id: "t", label: "Adım sayısı", min: 2, max: 60, value: 20, int: true },
    { id: "logN", label: "Simüle edilen yol sayısı 10^k", min: 1, max: 4, step: 0.5, value: 2 },
  ],
  seed: true,
  run(q, r) {
    const { S, T, abs } = chain(q), n = S.length, N = Math.round(10 ** q.logN);
    let p = Array(n).fill(0); p[0] = 1;
    const P = [p]; for (let t = 0; t < q.t; t++) { p = mv(T, p); P.push(p); }
    const cnt = range(q.t + 1).map(() => Array(n).fill(0));
    for (let k = 0; k < N; k++) { let s = 0; cnt[0][s]++; for (let t = 1; t <= q.t; t++) { let u = r(), j = 0; while (j < n - 1 && u >= T[j][s]) { u -= T[j][s]; j++; } s = j; cnt[t][s]++; } }
    const ts = range(q.t + 1), last = P[q.t], emp = cnt[q.t].map((c) => c / N);
    const notes = [], stats = [];
    if (abs.length) {
      const A = absorb(T, abs, 0);
      notes.push(`H'den başlayan bir oturum sonunda ${abs.map((a, i) => `${S[a]} durumunda olasılık ${fmt(A.p[i], 4)}`).join(", ")} ile biter; soğurulmaya kadar beklenen adım ${fmt(A.steps, 4)}. ${q.t} adımdan sonra kesin dağılımda ${S[abs[0]]} payı ${fmt(last[abs[0]], 4)}, simülasyonda ${fmt(emp[abs[0]], 4)} (N = ${N}).`);
      if (q.ch === 0) notes.push("Satın alma tek soğurucu durum ve her durumdan ona bir yol var: her oturum er geç satın almayla biter (olasılık 1). Terk etmeyi modellemek için ikinci bir soğurucu durum gerekir (ikinci zincir).");
      if (q.ch === 1 && q.q === 0) notes.push("q = 0'da terk durumuna hiç gidilmez ve zincir ders örneğine döner.");
      stats.push(...A.p.map((v, i) => ({ label: `P(sonunda ${S[abs[i]]})`, value: fmt(v, 4) })), { label: "Beklenen adım", value: fmt(A.steps, 4) });
    } else {
      const pi0 = stationary(T);
      if (!pi0.every(Number.isFinite)) {
        notes.push("a = b = 1: her durum kendi başına kapalı bir sınıftır, zincir başladığı yerde kalır. Durağan dağılım tek değildir: (x, 1 − x) biçimindeki her dağılım durağandır ve hangisine varılacağını başlangıç belirler.");
        stats.push({ label: "π", value: "tek değil" });
        notes.push(`Noktalar ${N} simüle edilmiş yolun durum oranları.`);
        return { plots: [{ title: "Durum olasılıkları p_t", xlabel: "t", ylabel: "olasılık", wide: true, size: { w: 620, h: 230 }, ylim: [0, 1.02], series: range(n).map((i) => ({ name: S[i], x: ts, y: P.map((pt) => pt[i]), c: i })) }], stats, note: notes.join(" ") };
      }
      const pi = pi0, tv = P.map((pt) => 0.5 * pt.reduce((s, v, i) => s + Math.abs(v - pi[i]), 0)), mix = tv.findIndex((d) => d < 0.01);
      notes.push(`Durağan dağılım π = (${pi.map((v) => fmt(v, 4)).join(", ")}). ${q.t}. adımda toplam değişim uzaklığı ${fmt(tv[q.t], 3)}; ${mix >= 0 ? `0.01'in altına ${mix}. adımda iniyor` : "bu ufukta 0.01'in altına inmiyor"}.`);
      if (q.ch === 3 && q.eps === 0) notes.push("ε = 0'da zincir periyodiktir (A → B → C → A): p_t hiç yakınsamaz, durağan dağılım düzgün olsa da tek başlangıç durumundan ona ulaşılmaz. Küçük bir tembellik ε periyodikliği bozar ve yakınsama başlar.");
      if (q.ch === 2) { const l2 = q.a + q.b - 1; notes.push(`İkinci özdeğer λ₂ = a + b − 1 = ${fmt(l2, 3)}: uzaklık |λ₂|^t hızıyla küçülür${Math.abs(l2) === 1 ? " — |λ₂| = 1'de hiç küçülmez (a = b = 1: iki ayrı kapalı sınıf; a = b = 0: periyodik)" : ""}.`); }
      stats.push({ label: "π", value: pi.map((v) => fmt(v, 3)).join(" / ") }, { label: `TV(p_${q.t}, π)`, value: fmt(tv[q.t], 3) }, { label: "Karışma adımı (TV < 0.01)", value: mix >= 0 ? String(mix) : "—" });
    }
    notes.push(`Noktalar ${N} simüle edilmiş yolun durum oranları; örnekleme hatası ≈ 1/√N = ${fmt(1 / Math.sqrt(N), 2)}.`);
    const show = range(n);
    return {
      plots: [
        { title: "Durum olasılıkları p_t (çizgi: T^t p₀, nokta: simülasyon)", xlabel: "t", ylabel: "olasılık", wide: true, size: { w: 620, h: 230 }, ylim: [0, 1.02],
          series: [...show.map((i) => ({ name: S[i], x: ts, y: P.map((pt) => pt[i]), c: i })), ...show.map((i) => ({ name: `${S[i]} (sim.)`, x: ts, y: cnt.map((c) => c[i] / N), c: i, kind: "scatter" }))] },
      ],
      stats,
      note: notes.join(" "),
    };
  },
};
