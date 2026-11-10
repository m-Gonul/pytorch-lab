// M10.6 — Doğrusal özyineleme x_t = a_t x_{t−1} + b_t'nin üç yolla hesaplanması (fp32, Math.fround ile her adımda yuvarlama):
// (0) sıralı döngü; (1) paralel tarama: (a, b) çiftleri üzerinde birleşmeli bileşim (a₂, b₂) ∘ (a₁, b₁) = (a₂a₁, a₂b₁ + b₂), Kogge–Stone
// sırasıyla (log₂T adım); (2) kümülatif çarpım ile kapalı biçim: P_t = Πa, x_t = P_t (x₀ + Σ b_k / P_k) — P_t sıfıra yaklaşınca 1/P_k
// taşar. Başvuru: float64 sıralı döngü. Ayrıca adım sayısı: sıralı T, tarama ⌈log₂T⌉.
import { fmt, range } from "./plot.js";

const f = Math.fround;
export default {
  params: [
    { id: "logT", label: "Dizi uzunluğu T = 2^k", min: 4, max: 14, value: 10, int: true },
    { id: "abar", label: "a_t ortalaması (unutma katsayısı)", min: 0.5, max: 1.0, step: 0.01, value: 0.9 },
    { id: "aspr", label: "a_t yayılımı (±)", min: 0, max: 0.09, step: 0.01, value: 0.05 },
  ],
  seed: true,
  run(q, r) {
    const T = 2 ** q.logT, a = range(T).map(() => f(Math.min(1, Math.max(0, q.abar + q.aspr * (2 * r() - 1))))), b = range(T).map(() => f(r() * 2 - 1));
    const ref = []; { let x = 0; for (let t = 0; t < T; t++) { x = a[t] * x + b[t]; ref.push(x); } }
    const seq = []; { let x = 0; for (let t = 0; t < T; t++) { x = f(f(a[t] * x) + b[t]); seq.push(x); } }
    // Kogge–Stone tarama (kapsayıcı) çiftler üzerinde
    let A = a.slice(), B = b.slice();
    for (let s = 1; s < T; s *= 2) { const A2 = A.slice(), B2 = B.slice(); for (let i = s; i < T; i++) { A2[i] = f(A[i] * A[i - s]); B2[i] = f(f(A[i] * B[i - s]) + B[i]); } A = A2; B = B2; }
    const scan = B; // x₀ = 0 olduğundan x_t = B_t
    const cp = []; { let P = 1, S = 0; for (let t = 0; t < T; t++) { P = f(P * a[t]); S = f(S + f(b[t] / P)); cp.push(f(P * S)); } }
    const scale = Math.max(1e-12, Math.sqrt(ref.reduce((s, v) => s + v * v, 0) / T)), err = (v) => Math.max(...v.map((x, i) => (Number.isFinite(x) ? Math.abs(x - ref[i]) : Infinity))) / scale;
    const eS = err(seq), eK = err(scan), eC = err(cp), Pmin = a.reduce((p, v) => p * v, 1);
    const fin = (e) => (Number.isFinite(e) ? fmt(e, 3) : "∞ (taşma/NaN)");
    const P = [`T = ${T}: en büyük göreli hata — sıralı ${fin(eS)}, paralel tarama ${fin(eK)}, kümülatif çarpım ${fin(eC)}. Adımlar: sıralı ${T}, tarama ${Math.ceil(Math.log2(T))}.`];
    P.push(`Πa_t = ${fmt(Pmin, 3)}: ${Pmin < 1e-30 ? "fp32'nin en küçük normal sayısının (1.2·10⁻³⁸) altına yaklaşıyor; 1/P_k taşıyor ve kapalı biçim çöküyor. Tarama yalnız çarpımları ve yakın geçmişin katkılarını taşıdığı için kararlı." : eC > 10 * Math.max(eS, 1e-12) ? "çarpım makul aralıkta ama küçülüyor; kapalı biçimin hatası büyümeye başladı." : "çarpım makul aralıkta; kapalı biçim de bu durumda çalışıyor."}`);
    const idx = range(Math.min(T, 512)).map((i) => Math.floor((i * T) / Math.min(T, 512)));
    return {
      plots: [{ title: "x_t: başvuru ve kümülatif çarpım yöntemi", xlabel: "t", ylabel: "x_t", wide: true, size: { w: 620, h: 220 }, series: [{ name: "başvuru (float64)", x: idx, y: idx.map((i) => ref[i]), c: 0 }, { name: "kümülatif çarpım (fp32)", x: idx, y: idx.map((i) => (Number.isFinite(cp[i]) && Math.abs(cp[i]) < 1e3 ? cp[i] : NaN)), c: 1, dash: true }] }],
      stats: [{ label: "Hata: sıralı", value: fin(eS) }, { label: "Hata: paralel tarama", value: fin(eK) }, { label: "Hata: kümülatif çarpım", value: fin(eC) }, { label: "Adım: sıralı / tarama", value: `${T} / ${Math.ceil(Math.log2(T))}` }],
      note: P.join(" "),
    };
  },
};
