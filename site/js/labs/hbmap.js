// M1.2 — Ağır topun (momentum) kararlılık haritası: yön başına 2×2 blok
// M = [[1−ηλ, −ηβ], [λ, β]] (durum x, v; det = β). Mercek: iki yönün spiral bantlarının kesişimi.
import { fmt, range } from "./plot.js";

// Blok M(λ) = [[1−ηλ, −ηβ], [λ, β]] özdeğerleri: (tr ± √(tr² − 4β))/2, tr = 1 − ηλ + β.
export function blockRho(lam, lr, beta) {
  const tr = 1 - lr * lam + beta, disc = tr * tr - 4 * beta;
  if (disc < 0) return { rho: Math.sqrt(beta), cplx: true };   // spiral: |μ| = √β
  const s = Math.sqrt(disc);
  return { rho: Math.max(Math.abs((tr + s) / 2), Math.abs((tr - s) / 2)), cplx: false };
}

export default {
  params: [
    { id: "kappa", label: "Koşul sayısı κ = λ_max/λ_min", min: 1, max: 200, log: true, value: 25, help: "λ_max = 1'e normalize; λ_min = 1/κ." },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 4, step: 0.01, value: 2.78, hardMin: 0.001, hardMax: 20, why: "η = 0 adım atmaz." },
    { id: "beta", label: "β (sürtünmenin tersi)", min: 0, max: 0.99, step: 0.01, value: 0.44 },
    { id: "T", label: "Adım sayısı", min: 10, max: 400, step: 1, value: 150, int: true },
    { id: "x0", label: "Başlangıç (özuzayda) x₁, x₂", min: -3, max: 3, step: 0.1, value: 2.5, help: "İkinci bileşen 2.0'da sabit: dik yön x₁, yatık yön x₂." },
  ],
  validate(p) {
    const st = blockRho(1, p.lr, p.beta), sf = blockRho(1 / p.kappa, p.lr, p.beta), out = [];
    if (st.rho > 1 || sf.rho > 1) out.push({ level: "warn", msg: `Iraksama: dik yön ρ = ${fmt(st.rho)}, yatık yön ρ = ${fmt(sf.rho)} — η, 2(1+β)/λ_max sınırını aştı.` });
    return out;
  },
  run(p) {
    const l1 = 1, l2 = 1 / p.kappa;
    // --- harita: iki bloğun birlikte en kötü spektral yarıçapı
    const nx = 60, ny = 44, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const bv = ((j + 0.5) * 0.99) / ny, ev = 0.01 + ((i + 0.5) * 3.99) / nx;
      z.push(Math.min(Math.max(blockRho(l1, ev, bv).rho, blockRho(l2, ev, bv).rho), 3));
    }
    const bv = range(40).map((i) => (i * 0.99) / 39);
    const bstar = ((Math.sqrt(p.kappa) - 1) / (Math.sqrt(p.kappa) + 1)) ** 2, etastar = (2 / (1 + 1 / Math.sqrt(p.kappa))) ** 2;
    // --- yön ayrık simülasyon: her özyön kendi 1-B ağır topu
    const run1 = (lam) => {
      let x = lam === l1 ? p.x0 : 2, v = 0;
      const xs = [Math.abs(x)], r = blockRho(lam, p.lr, p.beta);
      for (let t = 1; t <= p.T; t++) {
        const g = lam * x;
        v = p.beta * v + g;
        x = x - p.lr * v;
        if (!Number.isFinite(x) || Math.abs(x) > 1e6) { xs.push(NaN); for (let k = t + 1; k <= p.T; k++) xs.push(NaN); break; }
        xs.push(Math.abs(x));
      }
      return { xs, ...r, x0: Math.abs(lam === l1 ? p.x0 : 2) };
    };
    const S = run1(l1), F = run1(l2);
    const env = (r, x0) => range(p.T + 1).map((t) => Math.max(r ** t * x0, 1e-16));
    return {
      plots: [
        { title: "Kararlılık haritası ρ(η, β): koyu = hızlı sönüm, açık = ıraksama", xlabel: "β", ylabel: "η", size: { w: 430, h: 340 },
          field: { x0: 0, x1: 0.99, y0: 0.01, y1: 4, nx, ny, z, log: true }, xlim: [0, 0.99], ylim: [0.01, 4],
          series: [
            { name: "η = 2(1+β)/λ_max (ıraksama)", x: bv, y: bv.map((b) => 2 * (1 + b)), c: 1 },
            { name: "η = (1+√β)²/λ_max (spiral üst sınırı)", x: bv, y: bv.map((b) => (1 + Math.sqrt(b)) ** 2), c: 2, dash: true },
            { name: "η = (1−√β)²/λ_min (spiral alt sınırı)", x: bv, y: bv.map((b) => p.kappa * (1 - Math.sqrt(b)) ** 2), c: 3, dash: true },
            { name: "mercek ucu (β*, η*)", x: [bstar], y: [etastar], kind: "scatter", c: 1, r: 5 },
            { name: "şimdiki (β, η)", x: [p.beta], y: [p.lr], kind: "scatter", c: 0, r: 5 },
          ] },
        { title: "Özyön başına |x| (log): noktalı çizgiler blok çarpanı ρ^t·|x₀|", xlabel: "adım", ylabel: "|x|", ylog: true,
          series: [
            { name: "dik yön |x₁| (λ = λ_max)", x: range(p.T + 1), y: S.xs.map((v) => Math.max(v, 1e-16)), c: 0 },
            { name: "dik yön teorisi ρ^t|x₁₀|", x: range(p.T + 1), y: env(S.rho, S.x0), c: 0, dash: true },
            { name: "yatık yön |x₂| (λ = λ_min)", x: range(p.T + 1), y: F.xs.map((v) => Math.max(v, 1e-16)), c: 2 },
            { name: "yatık yön teorisi ρ^t|x₂₀|", x: range(p.T + 1), y: env(F.rho, F.x0), c: 2, dash: true },
          ] },
      ],
      stats: [
        { label: "Dik yön bloğu ρ (spiral/gerçek)", value: `${fmt(S.rho)}${S.cplx ? " (spiral √β)" : ""}` },
        { label: "Yatık yön bloğu ρ (spiral/gerçek)", value: `${fmt(F.rho)}${F.cplx ? " (spiral √β)" : ""}` },
        { label: "Mercek ucu β* = ((√κ−1)/(√κ+1))²", value: bstar },
        { label: "Mercek ucu η* = (2/(√λmax+√λmin))²", value: etastar },
        { label: "1e-3'e tahmini adım ln(1e-3)/ln(ρ_max)", value: (() => { const r = Math.max(S.rho, F.rho); return r >= 1 ? "ulaşmaz" : Math.ceil(Math.log(1e-3) / Math.log(r)); })() },
      ],
      note: `Ağır top, kuadratikte her özyönde bağımsız bir <b>2×2 bloğa</b> ayrılır: M(λ) = [[1−ηλ, −ηβ], [λ, β]], det M = β, tr M = 1 − ηλ + β. Blok, ηλ &lt; 2(1+β) koşuluyla yakınsar; (1−√β)² &lt; ηλ &lt; (1+√β)² bandında kökler karmaşıktır ve sönüm çarpanı tam √β'dır (spiral). <b>Mercek</b>, dik yönün spiral üst sınırı ηλ_max = (1+√β)² ile yatık yönün spiral alt sınırı ηλ_min = (1−√β)² eğrileri arasındaki bölgedir: her iki yön de √β hızında sönümlenir. Ucu (β*, η*) bu iki eğrinin kesişimidir: √κ(1+√β) = 1−√β denkleminden β* = ((√κ−1)/(√κ+1))² ve çarpan √β* = (√κ−1)/(√κ+1) çıkar; GD'nin (κ−1)/(κ+1)'inden çok daha iyidir. Iraksama eğrisi ηλ = 2(1+β), GD'nin ηλ = 2 sınırını (1+β) kat genişletir — bedeli sınır dibindeki uzun salınımlar.`,
    };
  },
};
