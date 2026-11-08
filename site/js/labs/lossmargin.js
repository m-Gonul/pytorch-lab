// M1.4 — Sınıflandırma kayıpları marjın fonksiyonu olarak: 0-1, çapraz entropi (lojistik), sigmoid üzerinde MSE, hinge.
// Ve sayısal kararlılık: naif log(σ(z)) float32'de nerede bozulur?
import { fmt, linspace } from "./plot.js";
import { softplus } from "./logreg2d.js";

const sig = (z) => 1 / (1 + Math.exp(-z));
// naif hesap (float32 benzetimi): önce σ, sonra log
export function naiveBCE(z, y) { const f = Math.fround, p = f(1 / f(1 + f(Math.exp(f(-z))))); return -(y ? Math.log(p) : Math.log(f(1 - p))); }
export const stableBCE = (z, y) => softplus(z) - y * z;
const LOSSES = [
  { name: "0-1 kaybı [m ≤ 0]", f: (m) => (m <= 0 ? 1 : 0), d: () => 0 },
  { name: "lojistik / BCE: log(1 + e^(−m))", f: (m) => softplus(-m), d: (m) => -sig(-m) },
  { name: "σ üzerinde MSE: (1 − σ(m))²", f: (m) => (1 - sig(m)) ** 2, d: (m) => -2 * (1 - sig(m)) * sig(m) * (1 - sig(m)) },
  { name: "hinge (SVM): max(0, 1 − m)", f: (m) => Math.max(0, 1 - m), d: (m) => (m < 1 ? -1 : 0) },
];

export default {
  params: [
    { id: "hl", label: "Vurgulanan kayıp", options: LOSSES.map((L) => L.name), value: 1 },
    { id: "z", label: "İncelenen logit z (y = 1 için marj m = z)", min: -120, max: 120, step: 0.5, value: -3, help: "Sağdaki tabloda naif ve kararlı hesap karşılaştırılır. Büyük |z|'de naif hesabın ne yaptığını gör." },
    { id: "y", label: "Gerçek etiket y", options: ["0", "1"], value: 1 },
  ],
  run(q) {
    const ms = linspace(-4, 4, 161), H = LOSSES[q.hl];
    const m = q.y ? q.z : -q.z;
    const nv = naiveBCE(q.z, q.y), sv = stableBCE(q.z, q.y);
    return {
      plots: [
        { title: "Kayıp, marj m = (2y − 1)·z'nin fonksiyonu olarak", xlabel: "marj m (pozitif = doğru taraf)", ylabel: "kayıp", ylim: [-0.1, 4], series: LOSSES.map((L, i) => ({ name: L.name, x: ms, y: ms.map(L.f), c: i === 0 ? 4 : i === 1 ? 0 : i === 2 ? 2 : 1, dash: i !== q.hl })), marks: [{ x: 0, label: "" }, ...(Math.abs(m) <= 4 ? [{ x: m, label: "seçili" }] : [])] },
        { title: `${H.name}: türevi dℓ/dm (öğrenme sinyali)`, xlabel: "marj m", ylabel: "dℓ/dm", ylim: [-1.2, 0.2], series: [{ name: "dℓ/dm", x: ms, y: ms.map(H.d), c: 2 }], marks: [{ y: 0, label: "" }] },
      ],
      stats: [
        { label: `Marj m (y = ${q.y})`, value: m },
        { label: "BCE, kararlı: softplus(z) − y·z", value: sv },
        { label: "BCE, naif float32: −log σ(z) / −log(1 − σ(z))", value: Number.isFinite(nv) ? nv : "∞ (log 0)" },
        { label: "Fark", value: Number.isFinite(nv) ? Math.abs(nv - sv) : "—" },
        { label: "BCE gradyanı ∂ℓ/∂z = σ(z) − y", value: sig(q.z) - q.y },
      ],
      note: `0-1 kaybı gerçek hedeftir ama türevi her yerde 0 (ya da tanımsız): gradyanla optimize edilemez. Diğerleri onun dışbükey (MSE hariç) <b>vekil</b>leridir. Lojistik kayıp yanlış taraftaki örneklere (m < 0) neredeyse doğrusal ceza ve sabit büyüklükte gradyan (≈ −1) verir; doğru tarafta hiç sıfırlanmaz, sadece küçülür. σ üzerinde MSE ise büyük negatif marjda düzleşir: model <b>emin ve yanlış</b>ken gradyan sıfıra gider. Hinge, m ≥ 1 olunca tamamen susar (SVM: yalnız destek vektörleri önemli). Sayısal not: naif hesap önce p = σ(z)'yi float32'de yuvarlar; y = 0 iken z ≳ 17'de p tam 1'e yuvarlanır, 1 − p = 0 ve log ∞ olur; y = 1 iken z ≲ −88'de e^(−z) float32'de taşar, p = 0 olur. Kararlı biçim log σ(z) = −softplus(−z)'yi doğrudan hesaplar — PyTorch'un BCEWithLogitsLoss'u bunu yapar.`,
    };
  },
};
