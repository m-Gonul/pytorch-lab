// M5.2 — Dirac GAN (Mescheder ve ark. 2018): veri δ₀, üreteç δ_θ, ayırt edicinin logiti −ψx (D(x) = σ(−ψx)). Değer
// V(θ, ψ) = f(ψθ) + f(0), f(t) = −log(1 + e^{−t}) = log(1 − D(θ)) + log D(0): üreteç küçültür, ayırt edici büyütür.
// Doyumsuz üreteç kaybı −log D(θ) = −log σ(−ψθ).
// Eşzamanlı ve dönüşümlü gradyan adımları, R1 cezası (γ/2)·ψ² (gerçek veride ayırt edicinin x-gradyanı = ψ).
import { fmt, range } from "./plot.js";

const sig = (t) => 1 / (1 + Math.exp(-t)), fp = (t) => sig(-t); // f'(t) = σ(−t)
export function simulate(o) {
  let th = o.th0, ps = o.ps0; const T = [[th, ps]];
  for (let k = 0; k < o.steps; k++) {
    // ayırt edici: ψ yönünde yükseliş; üreteç: θ yönünde iniş
    const gPsi = (t, p) => fp(p * t) * t - o.gamma * p;               // ∂V/∂ψ − R1
    const gTh = (t, p) => (o.ns ? sig(p * t) * p : fp(p * t) * p);  // minimaks: ∂V/∂θ = σ(−ψθ)ψ;  NS: D(sahte) = σ(−ψθ), ∂(−log D)/∂θ = σ(ψθ)ψ
    if (o.alt) { ps = ps + o.lr * gPsi(th, ps); th = th - o.lr * gTh(th, ps); }
    else { const a = gPsi(th, ps), b = gTh(th, ps); ps += o.lr * a; th -= o.lr * b; }
    T.push([th, ps]);
    if (!Number.isFinite(th) || Math.abs(th) > 1e6) break;
  }
  return T;
}

export default {
  params: [
    { id: "upd", label: "Güncelleme", options: ["eşzamanlı", "dönüşümlü (önce ayırt edici)"], value: 0 },
    { id: "loss", label: "Üreteç kaybı", options: ["minimaks: log(1 − D)", "doyumsuz: −log D"], value: 0 },
    { id: "gamma", label: "R1 cezası γ", min: 0, max: 2, step: 0.05, value: 0 },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 1, value: 0.2, log: true },
    { id: "steps", label: "Adım sayısı", min: 20, max: 2000, value: 400, int: true, log: true },
    { id: "th0", label: "Başlangıç θ", min: -2, max: 2, step: 0.1, value: 1 },
  ],
  run(q) {
    const T = simulate({ th0: q.th0, ps0: 0, lr: q.lr, steps: q.steps, gamma: q.gamma, alt: q.upd === 1, ns: q.loss === 1 });
    const r = T.map(([a, b]) => Math.hypot(a, b)), last = T.at(-1), n = T.length;
    const m = Math.max(1.5, ...T.slice(0, 2000).map(([a, b]) => Math.max(Math.abs(a), Math.abs(b)))) * 1.1;
    const ev = Math.hypot(1, q.lr / 2), P = [];
    P.push(`Denge (θ, ψ) = (0, 0): üreteç veriyle aynı noktada, ayırt edici kararsız (D = ½). ${n - 1} adım sonra (θ, ψ) = (${fmt(last[0], 3)}, ${fmt(last[1], 3)}), dengeye uzaklık ${fmt(r.at(-1), 3)} (başta ${fmt(r[0], 3)}).`);
    const mid = r[Math.floor((n - 1) / 2)], trend = r.at(-1) / Math.max(mid, 1e-12);
    const verdict = r.at(-1) < 1e-3 ? "dengeye yakınsıyor" : trend < 0.8 ? "dengeye yavaşça yaklaşıyor" : trend > 1.25 ? "<b>ıraksıyor</b>" : "dengenin çevresinde dönüyor (ne yakınsıyor ne ıraksıyor)";
    P.push(`Ölçülen eğilim: yarı yoldaki uzaklık ${fmt(mid, 3)}, sonda ${fmt(r.at(-1), 3)}: yörünge ${verdict}.`);
    if (q.gamma === 0 && q.loss === 0) P.push(q.upd === 0 ? `Dengenin yakınında dinamik saf bir <b>dönmedir</b> (Jacobian'ın özdeğerleri ±i/2). Eşzamanlı gradyan adımının özdeğerlerinin modülü √(1 + η²/4) = ${fmt(ev, 4)} > 1: her adımda yörünge biraz dışarı açılır.` : "Dönüşümlü adımlar özdeğer modülünü 1'e çeker: yörünge dengenin çevresinde kapalı bir eğri üzerinde döner.");
    if (q.gamma > 0) P.push("R1 cezası (γ/2)ψ², ayırt edicinin gerçek verideki gradyanını cezalandırır ve dinamiğe <b>sönüm</b> ekler: dengenin yakınında özdeğerlerin gerçel kısmı negatifleşir ve yörünge (0, 0)'a sarmal olarak yakınsar (yeterince küçük η ile).");
    if (q.loss === 1) P.push("Doyumsuz kayıp dengede minimaksla aynı Jacobian'ı verir (σ(0) = ½), ama denge dışında üretecin adımı σ(ψθ) ile ölçeklenir. İkinci dereceden bu fark dönmeyi biraz söndürür: eşzamanlı adımlarla yörünge ıraksamak yerine sınırlı bir çevrimde kalır, dönüşümlü adımlarla çok yavaş yakınsar (bu ayarlarda 20 000 adımda uzaklık 0.045).");
    return {
      plots: [
        { title: "Faz düzlemi: (θ, ψ) yörüngesi", xlabel: "θ (üreteç)", ylabel: "ψ (ayırt edici)", xlim: [-m, m], ylim: [-m, m], size: { w: 310, h: 300 }, series: [
          { name: "yörünge", x: T.map((p) => p[0]), y: T.map((p) => p[1]), c: 0 },
          { name: "başlangıç ve denge", x: [q.th0, 0], y: [0, 0], kind: "scatter", c: 2, r: 4 }] },
        { title: "Dengeye uzaklık", xlabel: "adım", ylabel: "√(θ² + ψ²)", yzero: true, series: [{ name: "uzaklık", x: range(n), y: r, c: 3 }] },
      ],
      stats: [
        { label: "Son θ", value: last[0] },
        { label: "Son ψ", value: last[1] },
        { label: "Dengeye uzaklık: başta / sonda", value: `${fmt(r[0], 3)} / ${fmt(r.at(-1), 3)}` },
        { label: "Eşzamanlı GD özdeğer modülü √(1 + η²/4)", value: ev },
      ],
      note: P.join(" "),
    };
  },
};
