// M1.2 — f(x) = a·x²/2 üzerinde gradyan inişi: x ← x − η·a·x = (1 − ηa)·x
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "a", label: "Eğrilik a (f'' )", min: 0.1, max: 10, step: 0.1, value: 2, help: "Büyük a = dik vadi. Çok değişkenli durumda a, Hessian'ın özdeğeridir." },
    { id: "lr", label: "Öğrenme oranı η", min: 0.01, max: 1.5, step: 0.01, value: 0.3, hardMin: 0, hardMax: 10, why: "η ≤ 0 adımı yokuş yukarı ya da hiç atmaz." },
    { id: "x0", label: "Başlangıç x₀", min: -5, max: 5, step: 0.1, value: 4 },
    { id: "T", label: "Adım sayısı", min: 1, max: 60, step: 1, value: 25, int: true },
  ],
  validate(p) {
    const r = 1 - p.lr * p.a, out = [];
    if (Math.abs(r) > 1) out.push({ level: "warn", msg: `ηa = ${fmt(p.lr * p.a)} > 2: her adımda |x| ${fmt(Math.abs(r))} katına çıkar — ıraksama.` });
    else if (Math.abs(Math.abs(r) - 1) < 1e-9) out.push({ level: "warn", msg: "Sınır durumu: |1 − ηa| = 1, x ne yaklaşır ne uzaklaşır (ηa = 2'de salınım, ηa = 0'da durma)." });
    return out;
  },
  run(p) {
    const r = 1 - p.lr * p.a, xs = [p.x0];
    for (let t = 0; t < p.T; t++) xs.push(xs[t] * r);
    const f = xs.map((x) => (p.a * x * x) / 2);
    const t = range(p.T + 1);
    const regime = r >= 0 && r < 1 ? "tekdüze yakınsama (ηa < 1): x işaret değiştirmeden sıfıra iner" : r < 0 && r > -1 ? "salınımlı yakınsama (1 < ηa < 2): x her adımda işaret değiştirip sıfıra yaklaşır" : r === 0 ? "tek adımda çözüm (η = 1/a): Newton adımıyla aynı" : "ıraksama (ηa > 2)";
    return {
      plots: [
        { title: "x_t", xlabel: "adım t", ylabel: "x", series: [{ name: "x_t", x: t, y: xs, dots: true }], marks: [{ y: 0, label: "minimum" }] },
        { title: "Kayıp f(x_t)", xlabel: "adım t", ylabel: "f", ylog: true, series: [{ name: "f(x_t)", x: t, y: f.map((v) => Math.max(v, 1e-300)) }] },
      ],
      stats: [
        { label: "Daralma çarpanı |1 − ηa|", value: Math.abs(r) },
        { label: "Kararlılık sınırı η < 2/a", value: 2 / p.a },
        { label: "En hızlı η = 1/a", value: 1 / p.a },
        { label: `f(x_${p.T})`, value: f[p.T] },
      ],
      note: `Rejim: <b>${regime}</b>. Kayıp her adımda (1 − ηa)² = ${fmt(r * r)} katına iner; log ölçekte düz çizgi bu yüzden. Çok boyutta en dik yön (en büyük a) kararlılığı, en yatık yön (en küçük a) hızı belirler: koşul sayısı a_max/a_min büyüdükçe GD yavaşlar.`,
    };
  },
};
