// M3.2 — Tek sigmoid nöron, tek girdi x = 1, hedef y = 0 (Nielsen, Bölüm 3). Kötü bir başlangıçtan (doymuş çıkış) MSE ile çapraz entropinin öğrenme hızı.
// BP1: δᴸ = ∂C/∂a · σ′(z). MSE'de σ′ kalır (doymuşken ≈ 0), CE'de sadeleşir: δᴸ = a − y.
import { fmt, range } from "./plot.js";

const sig = (z) => 1 / (1 + Math.exp(-z));
export function train(w0, b0, loss, eta, T) {
  let w = w0, b = b0; const A = [], C = [], D = [];
  for (let t = 0; t <= T; t++) {
    const z = w + b, a = sig(z);
    const cost = loss === 0 ? 0.5 * a * a : -Math.log(1 - a); // y = 0
    const delta = loss === 0 ? a * a * (1 - a) : a; // MSE: (a − y)σ′(z), CE: a − y
    A.push(a); C.push(cost); D.push(delta);
    w -= eta * delta * 1; b -= eta * delta;
  }
  return { A, C, D };
}

export default {
  params: [
    { id: "w0", label: "Başlangıç ağırlığı w₀", min: -3, max: 3, step: 0.1, value: 2 },
    { id: "b0", label: "Başlangıç bias'ı b₀", min: -3, max: 3, step: 0.1, value: 2, help: "w₀ = b₀ = 2: z = 4, a = σ(4) ≈ 0.98 — hedef 0 iken çok emin ve çok yanlış." },
    { id: "eta", label: "Öğrenme oranı η", min: 0.01, max: 2, value: 0.15, log: true, hardMin: 1e-6 },
    { id: "T", label: "Epoch", min: 10, max: 600, value: 300, int: true },
  ],
  run(q) {
    const T = Math.round(q.T), M = train(q.w0, q.b0, 0, q.eta, T), E = train(q.w0, q.b0, 1, q.eta, T), t = range(T + 1);
    const half = (A) => { const i = A.findIndex((a) => a < 0.5 * A[0]); return i < 0 ? `> ${T}` : i; };
    return {
      plots: [
        { title: "Çıkış a = σ(w·1 + b), hedef y = 0", xlabel: "epoch", ylabel: "a", ylim: [0, 1], series: [{ name: "MSE (kuadratik maliyet)", x: t, y: M.A, c: 2 }, { name: "çapraz entropi", x: t, y: E.A, c: 0 }] },
        { title: "Çıkış hatası δᴸ = ∂C/∂z (öğrenme sinyali)", xlabel: "epoch", ylabel: "δᴸ", ylog: true, series: [{ name: "MSE: (a − y)·σ′(z)", x: t, y: M.D.map((v) => Math.max(v, 1e-12)), c: 2 }, { name: "CE: a − y", x: t, y: E.D.map((v) => Math.max(v, 1e-12)), c: 0 }] },
      ],
      stats: [
        { label: "Başlangıç çıkışı a₀", value: M.A[0] },
        { label: "Başlangıçta δᴸ: MSE / CE", value: `${fmt(M.D[0], 3)} / ${fmt(E.D[0], 3)}` },
        { label: "a'nın yarıya inmesi için epoch: MSE / CE", value: `${half(M.A)} / ${half(E.A)}` },
        { label: `Son çıkış (epoch ${T}): MSE / CE`, value: `${fmt(M.A[T], 3)} / ${fmt(E.A[T], 3)}` },
      ],
      note: "BP1: δᴸ = ∂C/∂a · σ′(zᴸ). Kuadratik maliyette ∂C/∂a = a − y, yani δᴸ = (a − y)σ′(z). Nöron doymuşsa (a ≈ 0 ya da 1) σ′(z) = a(1 − a) ≈ 0: hata büyük olsa bile öğrenme neredeyse durur — eğrinin başındaki düz plato. Çapraz entropide ∂C/∂a = (a − y)/(a(1 − a)) ve σ′ ile tam sadeleşir: δᴸ = a − y — hata ne kadar büyükse o kadar hızlı öğrenir. İnsan sezgisine uyan davranış budur. (Gizli katmanlarda BP2'deki σ′(zˡ) çarpanı kalır; onun çaresi doymayan aktivasyonlar ve iyi başlatmadır, M3.5.)",
    };
  },
};
