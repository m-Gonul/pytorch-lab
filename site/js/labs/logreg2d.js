// M1.4 — 2B lojistik regresyon: olasılık haritası, karar sınırı, kayıp seçimi (çapraz entropi vs sigmoid üstünde MSE), ayrılabilir veride ‖w‖'nin büyümesi.
import { fmt, range } from "./plot.js";

export const sig = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const softplus = (z) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))); // log(1 + e^z), kararlı
export function bce(z, y) { return softplus(z) - y * z; } // −[y log σ(z) + (1−y) log(1−σ(z))]
export function makeData(q, r) {
  const n = Math.round(q.n), X = [], Y = [];
  for (let i = 0; i < n; i++) { const y = i % 2, c = y ? q.sep / 2 : -q.sep / 2; X.push([c + q.spread * r.normal(), 0.5 * c + q.spread * r.normal()]); Y.push(y); }
  return { X, Y };
}
export function train(X, Y, q) {
  let w = [0, 0], b = 0; const n = X.length, L = [], W = [];
  for (let t = 0; t <= q.steps; t++) {
    let gw = [0, 0], gb = 0, loss = 0;
    for (let i = 0; i < n; i++) {
      const z = w[0] * X[i][0] + w[1] * X[i][1] + b, p = sig(z);
      let dz;
      if (q.loss === 0) { loss += bce(z, Y[i]); dz = p - Y[i]; } // çapraz entropi: ∂ℓ/∂z = p − y
      else { loss += (p - Y[i]) ** 2; dz = 2 * (p - Y[i]) * p * (1 - p); } // MSE(σ(z), y): σ' çarpanı
      gw[0] += dz * X[i][0]; gw[1] += dz * X[i][1]; gb += dz;
    }
    L.push(loss / n + (q.lam / 2) * (w[0] ** 2 + w[1] ** 2)); W.push(Math.hypot(...w));
    if (t === q.steps) break;
    w = [w[0] - q.lr * (gw[0] / n + q.lam * w[0]), w[1] - q.lr * (gw[1] / n + q.lam * w[1])]; b -= (q.lr * gb) / n;
  }
  return { w, b, L, W };
}

export default {
  params: [
    { id: "sep", label: "Sınıf merkezleri arası uzaklık", min: 0, max: 6, step: 0.1, value: 2.5 },
    { id: "spread", label: "Sınıf içi yayılım σ", min: 0.1, max: 2, step: 0.05, value: 0.8 },
    { id: "n", label: "Örnek sayısı", min: 10, max: 300, value: 80, int: true },
    { id: "loss", label: "Kayıp", options: ["ikili çapraz entropi (BCE)", "σ(z) üzerinde MSE"], value: 0 },
    { id: "lam", label: "L2 cezası λ", min: 0, max: 1, step: 0.005, value: 0 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 5, value: 0.5, log: true, hardMin: 1e-6 },
    { id: "steps", label: "Adım sayısı", min: 1, max: 2000, value: 300, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const { X, Y } = makeData(q, r), T = train(X, Y, { ...q, steps: Math.round(q.steps) }), { w, b } = T;
    const R = 4, N = 50, z = [];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const x0 = -R + ((i + 0.5) * 2 * R) / N, x1 = -R + ((j + 0.5) * 2 * R) / N; z.push(sig(w[0] * x0 + w[1] * x1 + b)); }
    // karar sınırı w·x + b = 0 ve p = 0.1 / 0.9 eşleri
    const line = (lev) => { const off = Math.log(lev / (1 - lev)); if (Math.abs(w[1]) > 1e-9) return { x: [-R, R], y: [-R, R].map((x) => (off - b - w[0] * x) / w[1]) }; if (Math.abs(w[0]) > 1e-9) { const x = (off - b) / w[0]; return { x: [x, x], y: [-R, R] }; } return { x: [], y: [] }; };
    const acc = X.reduce((s, x, i) => s + ((w[0] * x[0] + w[1] * x[1] + b > 0 ? 1 : 0) === Y[i] ? 1 : 0), 0) / X.length;
    const l5 = line(0.5), l1 = line(0.1), l9 = line(0.9);
    return {
      plots: [
        { title: "P(y = 1 | x) (koyu = 0'a yakın) ve karar sınırı", xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-R, R], ylim: [-R, R], field: { x0: -R, x1: R, y0: -R, y1: R, nx: N, ny: N, z },
          series: [
            { name: "sınıf 0", x: X.filter((_, i) => !Y[i]).map((v) => v[0]), y: X.filter((_, i) => !Y[i]).map((v) => v[1]), kind: "scatter", c: 0, r: 3 },
            { name: "sınıf 1", x: X.filter((_, i) => Y[i]).map((v) => v[0]), y: X.filter((_, i) => Y[i]).map((v) => v[1]), kind: "scatter", c: 2, r: 3 },
            { name: "p = 0.5 (karar sınırı)", ...l5, c: 3 },
            { name: "p = 0.1 ve 0.9", ...l1, c: 4, dash: true }, { name: "", ...l9, c: 4, dash: true },
          ] },
        { title: "Eğitim kaybı ve ‖w‖", xlabel: "adım", ylabel: "değer", ylog: true, series: [
          { name: "kayıp", x: range(T.L.length), y: T.L.map((v) => Math.max(v, 1e-12)), c: 0 },
          { name: "‖w‖ (sınırın dikliği)", x: range(T.W.length), y: T.W.map((v) => Math.max(v, 1e-3)), c: 2 },
        ] },
      ],
      stats: [
        { label: "w, b", value: `(${fmt(w[0], 3)}, ${fmt(w[1], 3)}), ${fmt(b, 3)}` },
        { label: "Eğitim doğruluğu", value: `${fmt(acc * 100, 3)} %` },
        { label: "Son kayıp", value: T.L.at(-1) },
        { label: "‖w‖: p = 0.1'den 0.9'a geçiş genişliği ≈ 4.39/‖w‖", value: T.W.at(-1) > 1e-9 ? 4.394 / T.W.at(-1) : "∞" },
      ],
      note: `Karar sınırı w·x + b = 0 bir doğrudur; ‖w‖ sınırın ne kadar keskin olduğunu belirler (p'nin 0.1'den 0.9'a çıktığı bant 2·ln 9/‖w‖ genişliğinde). ${acc === 1 && q.lam === 0 && q.loss === 0 ? "<b>Veri doğrusal ayrılabilir</b> ve ceza yok: kayıp sıfıra ancak ‖w‖ → ∞ iken iner, bu yüzden ‖w‖ adımlarla durmadan büyür (log ölçekte yavaşça) ve olasılıklar 0/1'e kayar — aşırı güven. λ > 0 bunu durdurur." : ""} ${q.loss === 1 ? "MSE(σ(z), y) seçili: gradyan 2(p − y)·p(1 − p) — model emin ama <b>yanlış</b> olduğunda (p ≈ 0, y = 1) p(1 − p) ≈ 0 ve öğrenme durur. Çapraz entropide ∂ℓ/∂z = p − y: yanlış ve emin olmak en büyük gradyanı verir. Ayrıca MSE(σ(z)) z'ye göre dışbükey değildir." : ""}`,
    };
  },
};
