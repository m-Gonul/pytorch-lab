// M8.2 — EMA ağırlıkları: θ̄ₜ = β θ̄ₜ₋₁ + (1 − β) θₜ. Bir parametrenin SGD yörüngesi: hedefe üstel yaklaşım (zaman sabiti τ) + gürültü σ;
// isteğe bağlı olarak hedef bir adımda sıçrar (dağılım kayması). EMA'nın etkin penceresi 1/(1 − β), k adım önceki ağırlığın payı
// (1 − β)βᵏ; başlangıçta θ̄₀ = θ₀ (ya da 0'dan başlatılıp yanlılık düzeltmesi θ̄ₜ / (1 − βᵗ)).
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "beta", label: "EMA bozunması β", options: ["0.9", "0.99", "0.995", "0.999", "0.9999"], value: 2 },
    { id: "sigma", label: "Ağırlık gürültüsü σ (SGD'nin dalgalanması)", min: 0, max: 1, step: 0.01, value: 0.3 },
    { id: "tau", label: "Yaklaşma zaman sabiti τ (adım)", min: 10, max: 2000, value: 300, log: true },
    { id: "jump", label: "Hedefin sıçradığı adım (0 = yok)", min: 0, max: 4000, value: 2500, int: true },
    { id: "init", label: "EMA başlangıcı", options: ["θ₀'dan başlat", "0'dan başlat (düzeltmesiz)", "0'dan başlat + yanlılık düzeltmesi"], value: 0 },
    { id: "T", label: "Adım", min: 500, max: 5000, value: 4000, int: true },
  ],
  seed: true,
  run(q, r) {
    const beta = [0.9, 0.99, 0.995, 0.999, 0.9999][q.beta], th = [], ema = [];
    let target = 1, x = -1, m = q.init === 0 ? x : 0;
    for (let t = 0; t < q.T; t++) {
      if (q.jump && t === q.jump) target = 0;
      x += (target - x) / q.tau; const obs = x + q.sigma * r.normal(); th.push(obs);
      m = beta * m + (1 - beta) * obs; ema.push(q.init === 2 ? m / (1 - beta ** (t + 1)) : m);
    }
    const win = 1 / (1 - beta), tail = (a) => a.slice(Math.floor(q.T * 0.4), q.jump ? q.jump : q.T), sd = (a) => { const z = tail(a), mu = z.reduce((s, v) => s + v, 0) / z.length; return Math.sqrt(z.reduce((s, v) => s + (v - mu) ** 2, 0) / z.length); };
    const sdT = sd(th), sdE = sd(ema), noise = (a) => { const z = tail(a), d = z.map((v, i) => v - (i ? z[i - 1] : v)); return Math.sqrt(d.reduce((s, v) => s + v * v, 0) / d.length); };
    let lag = null; if (q.jump) { const e0 = Math.abs(ema[q.jump - 1]), k = ema.slice(q.jump).findIndex((v) => Math.abs(v) <= e0 / 2); lag = k < 0 ? null : k; }
    const P = [`β = ${beta}: etkin pencere 1/(1 − β) = ${fmt(win, 4)} adım; ${fmt(win, 4)} adım önceki ağırlığın toplamdaki payı β^${fmt(win, 4)} ≈ ${fmt(beta ** win, 3)} (≈ 1/e). Ham parametrenin std'si ${fmt(sdT, 3)}, EMA'nınki ${fmt(sdE, 3)}.`];
    if (q.jump) P.push(lag === null ? `Hedef ${q.jump}. adımda 1'den 0'a sıçradı; EMA benzetim sonuna kadar sıçrama anındaki değeriyle yeni hedef arasındaki yolun yarısını kat edemedi: pencere çok uzun.` : `Hedef ${q.jump}. adımda 1'den 0'a sıçradı; EMA, sıçrama anındaki değeriyle yeni hedef arasındaki yolun yarısını ${lag} adımda kat etti. Gecikmenin bir kısmı parametrenin kendi yavaşlığından (yarı ömür τ ln 2 = ${fmt(q.tau * Math.LN2, 3)} adım), bir kısmı EMA'dan (pencere × ln 2 = ${fmt(win * Math.LN2, 3)} adım) gelir. Uzun pencere gürültüyü bastırır ama değişikliklere geç yetişir.`);
    if (q.init === 1) P.push(`EMA 0'dan düzeltmesiz başladı: ilk ≈ ${fmt(win, 3)} adım 0'a doğru yanlı. Düzeltme EMAₜ / (1 − βᵗ) (Adam'daki gibi) ya da EMA'yı θ₀'dan başlatmak bunu giderir.`);
    P.push("Difüzyon modellerinde örnekleme EMA ağırlıklarıyla yapılır: ağırlıkların SGD gürültüsü, örneklerde gözle görülür kalite kaybına yol açar. Kısa eğitimlerde β çok büyük seçilirse EMA henüz eğitilmemiş erken ağırlıkları taşır.");
    const ts = range(q.T), st = Math.max(1, Math.floor(q.T / 800)), sub = (a) => a.filter((_, i) => i % st === 0);
    return {
      plots: [{ title: "Parametre ve EMA'sı", xlabel: "adım", ylabel: "θ", wide: true, size: { w: 620, h: 240 }, series: [{ name: "θₜ (SGD)", x: sub(ts), y: sub(th), c: 4 }, { name: "EMA", x: sub(ts), y: sub(ema), c: 0 }], marks: q.jump ? [{ x: q.jump, label: "sıçrama" }] : [] }],
      stats: [
        { label: "Etkin pencere (adım)", value: win },
        { label: "std: ham / EMA", value: `${fmt(sdT, 3)} / ${fmt(sdE, 3)}` },
        { label: "Sıçramadan sonra yarı yola gecikme (adım)", value: q.jump ? (lag === null ? "> benzetim" : lag) : "—" },
      ],
      note: P.join(" "),
    };
  },
};
