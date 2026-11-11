// M11.2 — Ders notlarındaki perakende stok ve dinamik fiyat MDP'si için sonlu ufuklu geriye doğru dinamik programlama.
// Durumlar S0 (stok yok), S1 (az), S2 (orta), S3 (çok); eylemler yüksek fiyat (satış olasılığı pH) ve indirim (pD). Satış stoku bir
// azaltır; S0'dan ertesi gün eylemden bağımsız S1'e dönülür (yeniden stok). Günlük ödül: S0 −c_out, S1 ve S2 −1, S3 −2; sezon sonunda
// (t = h) tasfiye cezası −L·k (k = stok düzeyi). V_h(s) = terminal; V_t(s) = max_a [R(s) + γ Σ P(s'|s,a) V_{t+1}(s')].
// En iyi eylem t'ye bağlıdır (durağan olmayan politika); ufuk uzadıkça başlangıçtaki politika sonsuz ufuklu durağan politikaya oturur.
import { fmt, range } from "./plot.js";

export function solve(q) {
  const R = [-q.cout, -1, -1, -2], term = [0, -q.L, -2 * q.L, -3 * q.L], V = [term], A = [[null, null, null, null]], Qs = [];
  for (let k = 0; k < q.h - 1; k++) {
    const nx = V[0], v = Array(4), a = Array(4), qq = Array(4);
    v[0] = R[0] + q.g * nx[1]; a[0] = null; qq[0] = [v[0], v[0]];
    for (let s = 1; s < 4; s++) {
      const hi = R[s] + q.g * (q.pH * nx[s - 1] + (1 - q.pH) * nx[s]), di = R[s] + q.g * (q.pD * nx[s - 1] + (1 - q.pD) * nx[s]);
      v[s] = Math.max(hi, di); a[s] = Math.abs(hi - di) < 1e-12 ? 0.5 : hi > di ? 1 : 0; qq[s] = [hi, di];
    }
    V.unshift(v); A.unshift(a); Qs.unshift(qq);
  }
  return { V, A, Qs }; // V[0] = t = 1, V[h−1] = t = h
}
const SN = ["S0 yok", "S1 az", "S2 orta", "S3 çok"];
export default {
  params: [
    { id: "h", label: "Ufuk h (gün)", min: 2, max: 30, value: 5, int: true },
    { id: "g", label: "İndirim γ", min: 0.5, max: 1, step: 0.01, value: 1 },
    { id: "pH", label: "Yüksek fiyatta satış olasılığı", min: 0, max: 1, step: 0.05, value: 0.3 },
    { id: "pD", label: "İndirimde satış olasılığı", min: 0, max: 1, step: 0.05, value: 0.8 },
    { id: "cout", label: "Stoksuz kalma cezası", min: 0, max: 20, step: 0.5, value: 5 },
    { id: "L", label: "Sezon sonu tasfiye cezası (stok birimi başına)", min: 0, max: 10, step: 0.5, value: 4 },
  ],
  validate(q) { return q.pH > q.pD ? [{ level: "warn", msg: "Yüksek fiyatta satış olasılığı indirimden büyük: indirimin hiçbir avantajı kalmaz." }] : []; },
  run(q) {
    const { V, A } = solve(q), ts = range(q.h).map((i) => i + 1);
    const pol = (s) => ts.map((t, i) => (A[i][s] === null ? NaN : A[i][s]));
    const P = [`t = 1'de değerler: ${SN.map((n, s) => `${n.split(" ")[0]} ${fmt(V[0][s], 4)}`).join(", ")}.`];
    const sw = [1, 2, 3].map((s) => { const p = pol(s), hi = p.map((x, i) => (x === 1 ? ts[i] : null)).filter((x) => x !== null); return hi.length ? `${SN[s].split(" ")[0]}: t ≤ ${Math.max(...hi)} yüksek fiyat` : `${SN[s].split(" ")[0]}: hep indirim`; });
    P.push(`Politika zamana bağlı: ${sw.join("; ")} (son gün t = h'de eylem yok).`);
    const first = [1, 2, 3].map((s) => A[0][s]), mid = q.h > 6 ? [1, 2, 3].map((s) => A[Math.floor(q.h / 2)][s]) : null;
    if (mid && first.every((x, i) => x === mid[i])) P.push("Ufuk uzunken ilk günlerin politikası aynı kalıyor: sonsuz ufuklu durağan politikaya oturmuş; değişim yalnız sezon sonuna yakın.");
    if (q.g < 1) P.push(`γ = ${q.g}: k gün sonraki ceza ve ödüller γ^k ile küçülüyor; uzun ufukta değerler −1/(1 − γ) mertebesinde sınırlı kalıyor.`);
    if (q.cout === 0) P.push("Stoksuz kalmanın cezası yok: satmak her zaman iyi, indirim baskın.");
    const lines = [0, 1, 2, 3].map((s) => ({ name: SN[s], x: ts, y: ts.map((t, i) => V[i][s]), c: s, dots: q.h <= 12 }));
    return {
      plots: [
        { title: "V_t(s): t gününde durumun değeri", xlabel: "gün t", ylabel: "V", side: true, size: { w: 300, h: 240 }, series: lines },
        { title: "En iyi eylem (1 = yüksek fiyat, 0 = indirim)", xlabel: "gün t", ylabel: "eylem", side: true, size: { w: 300, h: 240 }, ylim: [-0.15, 1.15],
          series: [1, 2, 3].map((s) => ({ name: SN[s], x: ts, y: pol(s).map((v) => (Number.isNaN(v) ? v : v + (s - 2) * 0.05)), kind: "step", c: s })) },
      ],
      stats: [{ label: "V₁(S1)", value: fmt(V[0][1], 4) }, { label: "V₁(S3)", value: fmt(V[0][3], 4) }, { label: "t = 1'de eylemler (S1/S2/S3)", value: first.map((x) => (x === 1 ? "yüksek" : x === 0 ? "indirim" : "eşit")).join(" / ") }],
      note: P.join(" "),
    };
  },
};
