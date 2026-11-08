// M2.3 — Çekirdek SVM, 2B. Dual problem: max Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼKᵢⱼ, 0 ≤ αᵢ ≤ C. Bias, çekirdeğe +1 eklenerek emilir.
// Çözüm: dual koordinat yükselişi (Hsieh ve ark. 2008). Destek vektörleri αᵢ > 0 olanlardır.
import { fmt } from "./plot.js";

export const KER = [
  { name: "doğrusal", k: (a, b) => a[0] * b[0] + a[1] * b[1] },
  { name: "polinom (1 + x·x′)³", k: (a, b) => (1 + a[0] * b[0] + a[1] * b[1]) ** 3 / 10 },
  { name: "RBF exp(−γ‖x − x′‖²)", k: (a, b, g) => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)) },
];
export function dataset(kind, n, noise, r) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const y = i % 2 ? 1 : -1; let p;
    if (kind === 0) p = [y * 1.2 + noise * r.normal(), 0.5 * y + noise * r.normal()];
    else if (kind === 1) { const t = Math.PI * r(); p = y > 0 ? [Math.cos(t) - 0.5, Math.sin(t) - 0.2] : [-Math.cos(t) + 0.5, -Math.sin(t) + 0.3]; p = [1.6 * p[0] + noise * r.normal(), 1.6 * p[1] + noise * r.normal()]; }
    else if (kind === 2) { const t = 2 * Math.PI * r(), rad = y > 0 ? 0.9 : 2.2; p = [rad * Math.cos(t) + noise * r.normal(), rad * Math.sin(t) + noise * r.normal()]; }
    else { const a = r() < 0.5 ? 1 : -1, b = a * y; p = [a * (0.4 + 1.2 * r()) + noise * r.normal(), b * (0.4 + 1.2 * r()) + noise * r.normal()]; } // XOR: y = işaret(x₁·x₂) (gürültüden önce)
    X.push(p); Y.push(y);
  }
  return { X, Y };
}
export function train(X, Y, ker, C, g, epochs = 60) {
  const n = X.length, K = X.map((a) => X.map((b) => KER[ker].k(a, b, g) + 1)), al = new Array(n).fill(0), f = new Array(n).fill(0);
  for (let ep = 0; ep < epochs; ep++) {
    let mx = 0;
    for (let i = 0; i < n; i++) {
      const G = Y[i] * f[i] - 1, na = Math.min(C, Math.max(0, al[i] - G / K[i][i])), d = na - al[i];
      if (d) { for (let j = 0; j < n; j++) f[j] += d * Y[i] * K[j][i]; al[i] = na; mx = Math.max(mx, Math.abs(d)); }
    }
    if (mx < 1e-6) break;
  }
  const dec = (x) => X.reduce((s, xi, i) => (al[i] ? s + al[i] * Y[i] * (KER[ker].k(xi, x, g) + 1) : s), 0);
  return { al, f, dec };
}

export default {
  params: [
    { id: "data", label: "Veri", options: ["iki küme (doğrusal)", "iki ay", "iç içe halkalar", "XOR"], value: 1 },
    { id: "ker", label: "Çekirdek", options: KER.map((k) => k.name), value: 2 },
    { id: "C", label: "C (hata cezası; büyük = sert marj)", min: 0.01, max: 1000, value: 1, log: true, hardMin: 1e-6 },
    { id: "g", label: "γ (RBF genişliğinin tersi)", min: 0.01, max: 50, value: 1, log: true, on: { ker: [2] }, help: "Büyük γ: her destek vektörünün etkisi dar bir tümsek — sınır noktaların etrafını sarar (aşırı uyum)." },
    { id: "noise", label: "Gürültü", min: 0, max: 0.8, step: 0.02, value: 0.15 },
    { id: "n", label: "Eğitim örneği", min: 20, max: 200, value: 100, int: true },
  ],
  seed: true,
  run(q, r) {
    const n = Math.round(q.n), { X, Y } = dataset(q.data, n, q.noise, r), V = dataset(q.data, 400, q.noise, r);
    const M = train(X, Y, q.ker, q.C, q.g);
    const R = 3.2, G = 64, z = [], b0 = [[], []], bp = [[], []];
    const grid = []; for (let j = 0; j < G; j++) { const row = []; for (let i = 0; i < G; i++) row.push(M.dec([-R + (i * 2 * R) / (G - 1), -R + (j * 2 * R) / (G - 1)])); grid.push(row); }
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) z.push(Math.max(-2, Math.min(2, grid[j][i])));
    const cross = (lev, out) => { for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const v = grid[j][i] - lev; if ((i < G - 1 && v * (grid[j][i + 1] - lev) <= 0 && grid[j][i + 1] !== grid[j][i]) || (j < G - 1 && v * (grid[j + 1][i] - lev) <= 0 && grid[j + 1][i] !== grid[j][i])) { out[0].push(-R + (i * 2 * R) / (G - 1)); out[1].push(-R + (j * 2 * R) / (G - 1)); } } };
    cross(0, b0); const bm = [[], []]; cross(1, bp); cross(-1, bm);
    const sv = M.al.map((a) => a > 1e-8), bound = M.al.map((a) => a >= q.C - 1e-8);
    const acc = (S) => S.X.reduce((s, x, i) => s + (Math.sign(M.dec(x)) === S.Y[i] ? 1 : 0), 0) / S.X.length;
    return {
      plots: [{ title: `f(x) (koyu = sınıf −1 tarafı), sınır f = 0 ve marjlar f = ±1`, xlabel: "x₁", ylabel: "x₂", size: { w: 400, h: 390 }, xlim: [-R, R], ylim: [-R, R],
        field: { x0: -R, x1: R, y0: -R, y1: R, nx: G, ny: G, z },
        series: [
          { name: "destek vektörleri (α > 0)", x: X.filter((_, i) => sv[i]).map((p) => p[0]), y: X.filter((_, i) => sv[i]).map((p) => p[1]), kind: "scatter", c: 3, r: 6.5 },
          { name: "y = −1", x: X.filter((_, i) => Y[i] < 0).map((p) => p[0]), y: X.filter((_, i) => Y[i] < 0).map((p) => p[1]), kind: "scatter", c: 0, r: 2.6 },
          { name: "y = +1", x: X.filter((_, i) => Y[i] > 0).map((p) => p[0]), y: X.filter((_, i) => Y[i] > 0).map((p) => p[1]), kind: "scatter", c: 2, r: 2.6 },
          { name: "sınır f = 0", x: b0[0], y: b0[1], kind: "scatter", c: 5, r: 1.3 },
          { name: "marj f = ±1", x: [...bp[0], ...bm[0]], y: [...bp[1], ...bm[1]], kind: "scatter", c: 4, r: 0.9 },
        ] }],
      stats: [
        { label: "Destek vektörü sayısı", value: `${sv.filter(Boolean).length} / ${n}` },
        { label: "…bunlardan α = C (marjın içinde ya da yanlış tarafta)", value: bound.filter(Boolean).length },
        { label: "Eğitim doğruluğu", value: `${fmt(acc({ X, Y }) * 100, 3)} %` },
        { label: "Test doğruluğu (400 yeni örnek)", value: `${fmt(acc(V) * 100, 3)} %` },
      ],
      note: `Mor halkalı (arkasında mor daire olan) noktalar destek vektörleridir: karar fonksiyonu f(x) = Σ αᵢyᵢk(xᵢ, x) yalnız onlara bağlıdır, diğer noktaları silmek çözümü değiştirmez. 0 < α < C olanlar tam marj üzerinde (f = ±1), α = C olanlar marjın içinde ya da yanlış taraftadır. <b>C</b> küçükse marj ihlallerine izin verilir (geniş marj, çok destek vektörü, yumuşak sınır); büyükse her hata pahalıdır (dar marj, sınır gürültüye uyar). ${q.ker === 2 ? `<b>γ = ${fmt(q.g, 3)}:</b> RBF'nin genişliği ~1/√γ. Küçük γ neredeyse doğrusal sınır, büyük γ her noktanın etrafında ada (eğitim doğruluğu ↑, test ↓).` : q.ker === 0 ? "Doğrusal çekirdekle sınır her zaman düz: ay, halka ve XOR verisinde yetersiz." : "Kübik polinom çekirdeği: sınır derece-3 bir eğri."}`,
    };
  },
};
