// M9.2 — Yoğunluk dönüşümleri: gamma, kontrast germe ve histogram eşitleme
import { fmt, range } from "./plot.js";
const W = 96, H = 72;
export default {
  params: [
    { id: "m", label: "Dönüşüm", options: ["yok (girdi)", "gamma s = r^γ", "doğrusal kontrast germe (yüzdelik)", "histogram eşitleme"], value: 3 },
    { id: "gamma", label: "γ (gamma)", min: 0.1, max: 5, log: true, value: 0.5, on: { m: [1] } },
    { id: "clip", label: "Germe yüzdeliği (%)", min: 0, max: 20, step: 0.5, value: 1, on: { m: [2] } },
    { id: "dark", label: "Girdinin karanlığı", min: 0, max: 0.9, step: 0.05, value: 0.6, help: "Sahnenin dar bir koyu aralığa sıkışmasını taklit eder." },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.02 },
  ],
  seed: true,
  run(p, rnd) {
    const img = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = 0.35 + 0.3 * (x / W) + (Math.hypot(x - 30, y - 36) < 16 ? 0.25 : 0) + (x > 60 && x < 84 && y > 14 && y < 58 ? -0.2 + 0.1 * ((x >> 2) & 1) : 0);
      v = v * (1 - p.dark) + p.noise * rnd.normal();
      img[y * W + x] = Math.min(1, Math.max(0, v));
    }
    const L = 256, q = (v) => Math.min(L - 1, Math.floor(v * L));
    const hist = (a) => { const h = new Array(L).fill(0); for (const v of a) h[q(v)]++; return h; };
    const hin = hist(img), cdf = []; let c = 0; for (const h of hin) { c += h; cdf.push(c / img.length); }
    let T;
    if (p.m === 0) T = (r) => r;
    else if (p.m === 1) T = (r) => r ** p.gamma;
    else if (p.m === 2) { const lo = cdf.findIndex((v) => v >= p.clip / 100) / (L - 1), hi = cdf.findIndex((v) => v >= 1 - p.clip / 100) / (L - 1); T = (r) => Math.min(1, Math.max(0, (r - lo) / Math.max(hi - lo, 1e-6))); }
    else { const cmin = cdf.find((v) => v > 0); T = (r) => (cdf[q(r)] - cmin) / (1 - cmin || 1); }
    const out = img.map(T), hout = hist(out);
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
    const bins = range(64).map((i) => (i + 0.5) / 64), b64 = (h) => range(64).map((i) => h.slice(i * 4, i * 4 + 4).reduce((a, b) => a + b, 0) / img.length);
    const rr = range(101).map((i) => i / 100);
    return {
      plots: [
        { title: "Girdi", img: { w: W, h: H, data: img }, side: true },
        { title: "Çıktı", img: { w: W, h: H, data: out }, side: true },
        { title: "Histogramlar (64 kova)", xlabel: "yoğunluk", ylabel: "oran", yzero: true, series: [{ name: "girdi", x: bins, y: b64(hin), kind: "bar", bw: 0.45, off: -0.5 }, { name: "çıktı", x: bins, y: b64(hout), kind: "bar", bw: 0.45, off: 0.5, c: 2 }] },
        { title: "Dönüşüm s = T(r)", xlabel: "girdi r", ylabel: "çıktı s", xlim: [0, 1], ylim: [0, 1], series: [{ name: "T", x: rr, y: rr.map(T) }, { name: "özdeşlik", x: [0, 1], y: [0, 1], c: 4, dash: true }] },
      ],
      stats: [{ label: "Girdi ortalama / std", value: `${fmt(mean(img), 3)} / ${fmt(sd(img), 3)}` }, { label: "Çıktı ortalama / std", value: `${fmt(mean(out), 3)} / ${fmt(sd(out), 3)}` }, { label: "Dolu kova sayısı (girdi → çıktı)", value: `${hin.filter(Boolean).length} → ${hout.filter(Boolean).length}` }],
      note: p.m === 3 ? "HE, girdinin birikimli dağılımını (CDF) dönüşüm olarak kullanır: çıktı histogramı yaklaşık düzgün olur. Ama kova sayısı artmaz (ayrık görüntüde düzgün dağılım ancak yaklaşık) ve düz bölgelerdeki gürültü de gerilir — CLAHE bunu kırpmayla sınırlar." : p.m === 1 ? `γ < 1 koyu tonları açar (eğim r = 0'da büyük), γ > 1 açık tonları ayırır. Şu an γ = ${fmt(p.gamma)}.` : p.m === 2 ? "Germe, alt/üst yüzdelikleri 0 ve 1'e eşleyen doğrusal bir dönüşüm; şekli korur, aralığı açar. Kırpılan uçlar doyar." : "Dönüşüm uygulanmadı.",
    };
  },
};
