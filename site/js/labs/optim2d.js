// M3.4 — Kötü koşullu 2B kuadratik üzerinde SGD / Momentum / Nesterov / RMSProp / Adam
import { fmt, linspace } from "./plot.js";
const NAMES = ["SGD", "Momentum (ağır top)", "Nesterov", "RMSProp", "Adam"];
export default {
  params: [
    { id: "m", label: "Yöntem", options: NAMES, value: 1 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.001, max: 1, log: true, value: 0.05, hardMin: 1e-6, hardMax: 10 },
    { id: "beta", label: "β (momentum / 1. moment)", min: 0, max: 0.99, step: 0.01, value: 0.9, hardMax: 0.999, on: { m: [1, 2, 4] } },
    { id: "kappa", label: "Koşul sayısı κ = λ_max/λ_min", min: 1, max: 200, log: true, value: 25 },
    { id: "rot", label: "Vadinin açısı (derece)", min: 0, max: 90, step: 1, value: 30, help: "0°: vadi eksenlere hizalı. Açı arttıkça koordinat başına uyarlama (RMSProp/Adam) avantajını kaybeder." },
    { id: "noise", label: "Gradyan gürültüsü σ (mini-batch)", min: 0, max: 2, step: 0.05, value: 0 },
    { id: "T", label: "Adım sayısı", min: 10, max: 300, step: 1, value: 100, int: true },
  ],
  seed: true,
  seedOn: { noise: 1 },
  run(p, rnd) {
    const l1 = 1, l2 = 1 / p.kappa, th = (p.rot * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
    // H = R diag(l1, l2) Rᵀ
    const H = [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]];
    const f = (x, y) => 0.5 * (H[0][0] * x * x + 2 * H[0][1] * x * y + H[1][1] * y * y);
    const g = (x, y) => [H[0][0] * x + H[0][1] * y, H[0][1] * x + H[1][1] * y];
    let x = -3, y = 2.5, v = [0, 0], m2 = [0, 0];
    const X = [x], Y = [y], L = [f(x, y)];
    for (let t = 1; t <= p.T; t++) {
      const look = p.m === 2 ? [x - p.lr * p.beta * v[0], y - p.lr * p.beta * v[1]] : [x, y];
      let G = g(look[0], look[1]);
      G = [G[0] + p.noise * rnd.normal(), G[1] + p.noise * rnd.normal()];
      let d;
      if (p.m === 0) d = G;
      else if (p.m === 1 || p.m === 2) { v = [p.beta * v[0] + G[0], p.beta * v[1] + G[1]]; d = v; }
      else if (p.m === 3) { m2 = [0.9 * m2[0] + 0.1 * G[0] ** 2, 0.9 * m2[1] + 0.1 * G[1] ** 2]; d = [G[0] / (Math.sqrt(m2[0]) + 1e-8), G[1] / (Math.sqrt(m2[1]) + 1e-8)]; }
      else { v = [p.beta * v[0] + (1 - p.beta) * G[0], p.beta * v[1] + (1 - p.beta) * G[1]]; m2 = [0.999 * m2[0] + 0.001 * G[0] ** 2, 0.999 * m2[1] + 0.001 * G[1] ** 2]; const b1 = 1 - p.beta ** t, b2 = 1 - 0.999 ** t; d = [(v[0] / b1) / (Math.sqrt(m2[0] / b2) + 1e-8), (v[1] / b1) / (Math.sqrt(m2[1] / b2) + 1e-8)]; }
      x -= p.lr * d[0]; y -= p.lr * d[1];
      if (!Number.isFinite(x) || Math.abs(x) > 1e6) { x = NaN; }
      X.push(x); Y.push(y); L.push(f(x, y));
    }
    const nx = 44, ny = 36, gx = linspace(-4, 4, nx), gy = linspace(-3.2, 3.2, ny), z = [];
    for (const yy of gy) for (const xx of gx) z.push(f(xx, yy));
    const fin = L.filter(Number.isFinite);
    const div = !Number.isFinite(L[L.length - 1]);
    const crit = 2 / l1;
    return {
      plots: [
        { title: "Yörünge (koyu = düşük kayıp)", xlabel: "w₁", ylabel: "w₂", field: { x0: -4, x1: 4, y0: -3.2, y1: 3.2, nx, ny, z, log: true }, xlim: [-4, 4], ylim: [-3.2, 3.2], series: [{ name: NAMES[p.m], x: X, y: Y, dots: true }] },
        { title: "Kayıp", xlabel: "adım", ylabel: "f", ylog: true, series: [{ name: NAMES[p.m], x: L.map((_, i) => i), y: L.map((v) => Math.max(v, 1e-16)) }] },
      ],
      stats: [
        { label: "Son kayıp", value: div ? "ıraksadı" : L[L.length - 1] },
        { label: "Kayıp < 10⁻³ olduğu ilk adım", value: (() => { const i = L.findIndex((v) => v < 1e-3); return i < 0 ? "ulaşmadı" : i; })() },
        { label: "SGD kararlılık sınırı 2/λ_max", value: crit },
        { label: "Momentum için sınır 2(1+β)/λ_max", value: (2 * (1 + p.beta)) / l1 },
      ],
      note: div ? `Iraksadı: η = ${fmt(p.lr)} en dik yöndeki kararlılık sınırını aştı (SGD için 2/λ_max = 2${p.m === 1 || p.m === 2 ? `, momentumla 2(1+β) = ${fmt(2 * (1 + p.beta))}` : ""}).` : [
        `SGD: yatık yönde (λ_min = ${fmt(l2)}) adım başına çarpan 1 − ηλ_min = ${fmt(1 - p.lr * l2)}; ${p.T} adımda bu yönde hatanın ${fmt((1 - p.lr * l2) ** p.T)} katı kalır. Koşul sayısı büyüdükçe bu çarpan 1'e yaklaşır.`,
        `Momentum: yatık yönde hız birikir (etkin adım ≈ η/(1 − β) = ${fmt(p.lr / (1 - p.beta))}); dik yönde ardışık adımlar zıt işaretli olduğu için birbirini kısmen götürür. β → 1 sönümü azaltır: uzun spiral.`,
        `Nesterov: gradyanı "bir adım ileride" (w − ηβv) aldığı için dönüş noktasını erken fark eder; aynı β'da momentumdan daha az salınır.`,
        `RMSProp: her koordinat kendi gradyan RMS'ine bölünür. Vadi eksenlere hizalıysa (açı 0°) bu tam ön koşullandırmadır; döndürülmüşse (açı ${fmt(p.rot)}°) eğrilik koordinatlar arasında karıştığı için kısmidir.`,
        `Adam: momentum (β₁ = ${fmt(p.beta)}) + RMSProp (β₂ = 0.999) + yanlılık düzeltmesi. Adım boyu kabaca η ile sınırlı (gradyan büyüklüğünden bağımsız): minimum yakınında salınım bırakabilir — η takvimi bu yüzden gerekir.`,
      ][p.m],
    };
  },
};
