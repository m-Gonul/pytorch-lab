// M1.2 — f(x) = a·x²/2 üzerinde gradyan inişi: x ← x − η·a·x = (1 − ηa)·x
// Kobweb diyagramı (y = x karşı y = (1−ηa)·x) ve (η, a) rejim haritası dahil.
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
    // Kobweb: (x_t, x_t)'ten y = (1−ηa)·x doğrusuna dik, oradan y = x'e yatay
    const C = 6.5, cw = { x: [p.x0], y: [p.x0] };
    let cx = p.x0;
    for (let k = 0; k < p.T; k++) {
      if (Math.abs(cx) > C) { cw.x.push(NaN); cw.y.push(NaN); break; }
      const nx = cx * r;
      cw.x.push(cx); cw.y.push(nx);                       // dik: y = x → y = (1−ηa)·x
      if (Math.abs(nx) > C) { cw.x.push(NaN); cw.y.push(NaN); break; }
      cw.x.push(nx); cw.y.push(nx);                       // yatay: geri y = x'e
      cx = nx;
    }
    // Rejim haritası: |1 − ηa| ısı haritası (η yatay, a dikey)
    const nx = 60, ny = 48, z = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const av = 0.1 + ((j + 0.5) * 9.9) / ny, ev = 0.01 + ((i + 0.5) * 1.49) / nx;
      z.push(Math.abs(1 - ev * av));
    }
    const av = range(41).map((i) => 0.2 + i * 0.245);
    return {
      plots: [
        { title: "x_t", xlabel: "adım t", ylabel: "x", series: [{ name: "x_t", x: t, y: xs, dots: true }], marks: [{ y: 0, label: "minimum" }] },
        { title: "Kayıp f(x_t)", xlabel: "adım t", ylabel: "f", ylog: true, series: [{ name: "f(x_t)", x: t, y: f.map((v) => Math.max(v, 1e-300)) }] },
        { title: "Kobweb: y = x ve y = (1 − ηa)·x", xlabel: "x_t", ylabel: "x_{t+1}", side: true, size: { w: 340, h: 340 }, xlim: [-C, C], ylim: [-C, C],
          series: [
            { name: "y = x", x: [-C, C], y: [-C, C], c: 3, dash: true },
            { name: `y = (1 − ηa)·x  (çarpan ${fmt(r)})`, x: [-C, C], y: [-C * r, C * r], c: 1 },
            { name: "adımlar", x: cw.x, y: cw.y, c: 0, dots: true },
          ] },
        { title: "Rejim haritası |1 − ηa|: koyu = hızlı yakınsama, açık = ıraksama", xlabel: "η", ylabel: "a", side: true, size: { w: 340, h: 330 },
          field: { x0: 0.01, x1: 1.5, y0: 0.1, y1: 10, nx, ny, z, log: true }, xlim: [0.01, 1.5], ylim: [0.1, 10],
          series: [
            { name: "η = 2/a (ıraksama sınırı)", x: av.map((a) => 2 / a), y: av, c: 1 },
            { name: "η = 1/a (Newton adımı)", x: av.map((a) => 1 / a), y: av, c: 2, dash: true },
            { name: "şimdiki (η, a)", x: [p.lr], y: [p.a], kind: "scatter", c: 0, r: 5 },
          ] },
      ],
      stats: [
        { label: "Daralma çarpanı |1 − ηa|", value: Math.abs(r) },
        { label: "Kararlılık sınırı η < 2/a", value: 2 / p.a },
        { label: "En hızlı η = 1/a", value: 1 / p.a },
        { label: `f(x_${p.T})`, value: f[p.T] },
        { label: `T adımda kalan oransal hata |1 − ηa|^${p.T}`, value: Math.abs(r) ** p.T },
      ],
      note: `Rejim: <b>${regime}</b>. Kayıp her adımda (1 − ηa)² = ${fmt(r * r)} katına iner; log ölçekte düz çizgi bu yüzden. Kobweb diyagramında her adım y = x doğrusundan eğri doğruya dik iner, oradan yatay gidip geri döner: çarpan |1 − ηa| < 1 iken merdiven içe, > 1 iken dışa sarar; salınımlı rejimde (1 < ηa < 2) her dik iniş karşıya geçer. Rejim haritasında eğriler η = 1/a (tek adımda çözüm) ve η = 2/a (ıraksama sınırı); slider'lar kaydıkça nokta haritada gezinir. Çok boyutta en dik yön (en büyük a) kararlılığı, en yatık yön (en küçük a) hızı belirler: koşul sayısı a_max/a_min büyüdükçe GD yavaşlar.`,
    };
  },
};
