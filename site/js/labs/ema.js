// M3.4 / M8.2 — Üstel hareketli ortalama: gecikme, gürültü bastırma, yanlılık düzeltmesi
import { fmt, range } from "./plot.js";
export default {
  params: [
    { id: "beta", label: "β (bozunma)", min: 0, max: 0.999, step: 0.001, value: 0.9, hardMax: 0.99999, why: "β ≥ 1 ortalama hiç güncellenmez." },
    { id: "noise", label: "Gürültü σ", min: 0, max: 2, step: 0.05, value: 0.8 },
    { id: "jump", label: "Sinyaldeki sıçrama", min: 0, max: 3, step: 0.1, value: 1.5 },
    { id: "T", label: "Uzunluk", min: 50, max: 1000, step: 10, value: 300, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const t = range(p.T), sig = t.map((i) => 1 + (i > p.T / 2 ? p.jump : 0) + 0.3 * Math.sin(i / 25));
    const x = sig.map((v) => v + p.noise * rnd.normal());
    let m = 0; const ema = [], corr = [];
    x.forEach((v, i) => { m = p.beta * m + (1 - p.beta) * v; ema.push(m); corr.push(m / (1 - p.beta ** (i + 1))); });
    const res = (a, from) => Math.sqrt(a.slice(from).reduce((s, v, k) => s + (v - sig[k + from]) ** 2, 0) / (a.length - from));
    const half = Math.floor(p.T / 2), target = sig[half + 1], prev = sig[half];
    const lag = corr.slice(half + 1).findIndex((v) => Math.abs(v - prev) >= 0.632 * Math.abs(target - prev));
    return {
      plots: [{ title: "Gürültülü sinyal ve EMA", xlabel: "adım", ylabel: "değer", series: [{ name: "gözlem", x: t, y: x, c: 4 }, { name: "gerçek sinyal", x: t, y: sig, c: 1, dash: true }, { name: "EMA (ham)", x: t, y: ema, c: 3 }, { name: "EMA / (1 − β^t)", x: t, y: corr, c: 0 }] }],
      stats: [{ label: "Etkin pencere 1/(1 − β)", value: 1 / (1 - p.beta) }, { label: "Gürültü std'sinin kalan oranı √((1−β)/(1+β))", value: Math.sqrt((1 - p.beta) / (1 + p.beta)) }, { label: "Sıçramaya %63 tepki gecikmesi (adım)", value: lag < 0 ? "> kalan uzunluk" : lag + 1 }, { label: "İlk 20 adımda ham EMA hatası / düzeltilmiş", value: `${fmt(res(ema.slice(0, 20), 0))} / ${fmt(res(corr.slice(0, 20), 0))}` }],
      note: `β büyüdükçe gürültü daha çok bastırılır ama sıçramaya geç tepki verilir (≈ 1/(1−β) adım): bu ödünleşim kaçınılmaz. Ham EMA sıfırdan başladığı için ilk adımlarda sıfıra doğru yanlıdır; 1 − β^t'ye bölmek (Adam'daki yanlılık düzeltmesi) bunu giderir.`,
    };
  },
};
