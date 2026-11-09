// M4.11 — ARMA(2,1) süreci: bir gerçekleşme, örnek otokorelasyon (ACF) ve kısmi otokorelasyon (PACF) ±1.96/√n bantlarıyla,
// kuramsal ACF, durağanlık (AR polinomunun kökleri birim çemberin dışında mı?), Yule–Walker kestirimi; eğilim ekleyip fark alma.
import { fmt, range } from "./plot.js";
import { cholSolve } from "./ddescent.js";

export function simulate(q, r, n) {
  const burn = 200, y = [], e = []; let y1 = 0, y2 = 0, e1 = 0;
  for (let t = 0; t < n + burn; t++) { const et = r.normal(), yt = q.phi1 * y1 + q.phi2 * y2 + et + q.theta * e1; y2 = y1; y1 = yt; e1 = et; if (t >= burn) { y.push(yt); e.push(et); } if (!Number.isFinite(yt) || Math.abs(yt) > 1e12) return { y: null }; }
  return { y };
}
export function acf(x, K) { const n = x.length, m = x.reduce((a, b) => a + b, 0) / n, c0 = x.reduce((a, v) => a + (v - m) ** 2, 0) / n; return range(K + 1).map((k) => x.slice(k).reduce((a, v, i) => a + (v - m) * (x[i] - m), 0) / n / c0); }
export function pacf(rho, K) { // Durbin–Levinson
  const out = [1], phi = [[]]; let v = 1;
  for (let k = 1; k <= K; k++) {
    const prev = phi[k - 1], a = (rho[k] - prev.reduce((s, p, j) => s + p * rho[k - 1 - j], 0)) / v;
    const cur = prev.map((p, j) => p - a * prev[k - 2 - j]); cur.push(a); phi.push(cur); v *= 1 - a * a; out.push(a);
  }
  return out;
}
// kuramsal ACF: ψ ağırlıkları (MA(∞)) üzerinden γ_k = Σ ψ_j ψ_{j+k}
export function theoryAcf(q, K) { const J = 600, psi = [1, q.phi1 + q.theta]; for (let j = 2; j < J; j++) psi.push(q.phi1 * psi[j - 1] + q.phi2 * psi[j - 2]); const g = (k) => psi.slice(0, J - k).reduce((s, p, j) => s + p * psi[j + k], 0); const g0 = g(0); return range(K + 1).map((k) => g(k) / g0); }
export function roots(q) { // 1 − φ1 z − φ2 z² = 0
  if (Math.abs(q.phi2) < 1e-12) return Math.abs(q.phi1) < 1e-12 ? [Infinity] : [Math.abs(1 / q.phi1)];
  const a = -q.phi2, b = -q.phi1, c = 1, D = b * b - 4 * a * c;
  if (D >= 0) return [Math.abs((-b + Math.sqrt(D)) / (2 * a)), Math.abs((-b - Math.sqrt(D)) / (2 * a))];
  const m = Math.sqrt(c / a); return [Math.abs(m), Math.abs(m)];
}

export default {
  params: [
    { id: "phi1", label: "φ₁ (AR, gecikme 1)", min: -2, max: 2, step: 0.05, value: 0.6 },
    { id: "phi2", label: "φ₂ (AR, gecikme 2)", min: -1, max: 1, step: 0.05, value: 0.3 },
    { id: "theta", label: "θ₁ (MA, gecikme 1)", min: -1, max: 1, step: 0.05, value: 0 },
    { id: "trend", label: "Eklenen deterministik eğilim", options: ["yok", "doğrusal (0.05·t)", "karesel (0.0005·t²)"], value: 0 },
    { id: "d", label: "Fark alma sayısı d", min: 0, max: 2, value: 0, int: true },
    { id: "n", label: "Gözlem sayısı n", min: 50, max: 2000, value: 400, int: true, log: true },
  ],
  seed: true,
  validate(q) {
    const R = roots(q), st = R.every((x) => x > 1 + 1e-9);
    return st ? [] : [{ level: "warn", msg: `AR polinomu 1 − φ₁z − φ₂z²'nin bir kökü birim çemberin içinde ya da üstünde (|z| = ${R.map((x) => fmt(x, 3)).join(", ")}): süreç durağan değil. Durağanlık bölgesi: φ₁ + φ₂ < 1, φ₂ − φ₁ < 1, |φ₂| < 1.` }];
  },
  run(q, r) {
    const n = Math.round(q.n), S = simulate(q, r, n), R = roots(q), st = R.every((x) => x > 1 + 1e-9);
    if (!S.y) return { plots: [{ title: "Süreç patladı", xlabel: "t", ylabel: "y", series: [{ name: "—", x: [0, 1], y: [0, 0] }] }], stats: [{ label: "Durum", value: "Patlayan (|y| > 10¹²): AR kökü birim çemberin içinde" }], note: "Durağan olmayan AR süreci: varyans zamanla sınırsız büyüyor. Kökleri birim çemberin dışına taşı." };
    let y = S.y.map((v, t) => v + (q.trend === 1 ? 0.05 * t : q.trend === 2 ? 0.0005 * t * t : 0));
    for (let k = 0; k < q.d; k++) y = y.slice(1).map((v, i) => v - y[i]);
    const K = 20, rho = acf(y, K), pa = pacf(rho, K), th = q.d === 0 && q.trend === 0 && st ? theoryAcf(q, K) : null, band = 1.96 / Math.sqrt(y.length);
    const A = [[1, rho[1]], [rho[1], 1]], yw = cholSolve(A, [rho[1], rho[2]]), ks = range(K + 1);
    return {
      plots: [
        { title: `Bir gerçekleşme${q.d ? ` (${q.d} kez farkı alınmış)` : ""}`, xlabel: "t", ylabel: "y", series: [{ name: "y", x: range(y.length), y, c: 0 }] },
        { title: "ACF (çubuk) ve PACF (nokta), ±1.96/√n bandı", xlabel: "gecikme k", ylabel: "korelasyon", ylim: [-1.05, 1.05], series: [{ name: "örnek ACF", x: ks, y: rho, kind: "bar", c: 0, bw: 0.4 }, { name: "örnek PACF", x: ks.slice(1), y: pa.slice(1), kind: "scatter", c: 2, r: 3.5 }, ...(th ? [{ name: "kuramsal ACF", x: ks, y: th, c: 1, dash: true }] : [])], marks: [{ y: band, label: "+1.96/√n" }, { y: -band, label: "−1.96/√n" }] },
      ],
      stats: [
        { label: "AR köklerinin modülü |z| (> 1 ise durağan)", value: R.map((x) => (Number.isFinite(x) ? fmt(x, 4) : "∞")).join(", ") },
        { label: "Yule–Walker AR(2) kestirimi (φ̂₁, φ̂₂)", value: `${fmt(yw[0], 3)}, ${fmt(yw[1], 3)}` },
        { label: "ρ₁, ρ₂ (örnek)", value: `${fmt(rho[1], 3)}, ${fmt(rho[2], 3)}` },
        ...(th ? [{ label: "ρ₁, ρ₂ (kuramsal)", value: `${fmt(th[1], 3)}, ${fmt(th[2], 3)}` }] : []),
      ],
      note: `${q.trend > 0 && q.d === 0 ? "<b>Eğilim var, fark yok:</b> ortalama zamana bağlı, seri durağan değil; örnek ACF çok yavaş söner (neredeyse her gecikmede 1'e yakın) — durağan olmayışın tipik işareti. Fark almayı dene." : q.trend > 0 && q.d < q.trend ? "Bir fark doğrusal eğilimi sabite çevirir ama karesel eğilim için iki fark gerekir." : ""} AR(p) sürecinde PACF p gecikmeden sonra (bant içinde) kesilir, ACF üstel/sönümlü salınımla söner; MA(q) sürecinde tersi: ACF q'dan sonra kesilir. Box–Jenkins model belirleme bu iki desene bakar. Yule–Walker kestirimi ${q.theta !== 0 ? "<b>MA terimi varken yanlı</b>: AR(2) varsayımı yanlış, kestirim gerçek φ'lerden sapar." : "AR(2) doğru model olduğundan gerçek φ'lere yakın (n büyüdükçe yakınsar)."}`,
    };
  },
};
