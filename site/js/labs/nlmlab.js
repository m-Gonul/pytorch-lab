// M9.12 — Gauss, bilateral ve non-local means (Buades ve ark. 2005), 2-B. 48 × 48 sahne: dikey çizgili doku (periyot 6), bir kare,
// bir rampa; beyaz Gauss gürültüsü. NLM: her p için arama penceresindeki q'larda w = exp(−‖V_p − V_q‖²/h²) (yama ortalama kare farkı),
// merkez pikselin ağırlığı komşuların en büyüğüne eşitlenir (IPOL uygulaması). Ölçütler: PSNR, yöntem gürültüsü (temiz görüntüye
// uygulanınca kalan fark, RMS) ve doku karşıtlığının korunması.
import { fmt, range } from "./plot.js";

const W = 48;
export function scene() {
  const f = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let v = 0.35;
    if (y < 22 && x < 22) v = 0.3 + 0.4 * (Math.floor(x / 3) % 2);
    if (x >= 26 && x < 44 && y >= 4 && y < 20) v = 0.8;
    if (y >= 26) v = 0.15 + (0.7 * x) / (W - 1);
    if (y >= 30 && y < 44 && x >= 8 && x < 22) v = 0.9;
    f[y * W + x] = v;
  }
  return f;
}
const at = (a, x, y) => a[Math.min(W - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))];
export function gauss(a, s) { const R = Math.ceil(3 * s); return Float64Array.from(range(W * W), (i) => { const x = i % W, y = (i / W) | 0; let t = 0, n = 0; for (let j = -R; j <= R; j++) for (let k = -R; k <= R; k++) { const w = Math.exp(-(j * j + k * k) / (2 * s * s)); t += w * at(a, x + k, y + j); n += w; } return t / n; }); }
export function bilateral(a, ss, sr) { const R = Math.ceil(2 * ss); return Float64Array.from(range(W * W), (i) => { const x = i % W, y = (i / W) | 0, c = a[i]; let t = 0, n = 0; for (let j = -R; j <= R; j++) for (let k = -R; k <= R; k++) { const v = at(a, x + k, y + j), w = Math.exp(-(j * j + k * k) / (2 * ss * ss) - ((v - c) ** 2) / (2 * sr * sr)); t += w * v; n += w; } return t / n; }); }
export function nlm(a, P, S, h) {
  const p = (P - 1) / 2, s = (S - 1) / 2, h2 = h * h, np = P * P;
  return Float64Array.from(range(W * W), (i) => {
    const x = i % W, y = (i / W) | 0; let t = 0, n = 0, wmax = 0;
    for (let dy = -s; dy <= s; dy++) for (let dx = -s; dx <= s; dx++) {
      if (!dx && !dy) continue;
      const qx = x + dx, qy = y + dy; if (qx < 0 || qy < 0 || qx >= W || qy >= W) continue;
      let d = 0; for (let j = -p; j <= p; j++) for (let k = -p; k <= p; k++) d += (at(a, x + k, y + j) - at(a, qx + k, qy + j)) ** 2;
      const w = Math.exp(-(d / np) / h2); t += w * a[qy * W + qx]; n += w; if (w > wmax) wmax = w;
    }
    t += (wmax || 1) * a[i]; n += wmax || 1;
    return t / n;
  });
}
const psnr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m += (a[i] - b[i]) ** 2; return 10 * Math.log10(1 / (m / a.length)); };
const METHODS = ["Gauss", "bilateral", "non-local means"];
export default {
  params: [
    { id: "m", label: "Süzgeç", options: METHODS, value: 2 },
    { id: "gs", label: "Gauss σ", min: 0.5, max: 3, step: 0.1, value: 1, on: { m: [0] } },
    { id: "ss", label: "Bilateral σ_s", min: 0.5, max: 3, step: 0.1, value: 1.5, on: { m: [1] } },
    { id: "sr", label: "Bilateral σ_r", min: 0.02, max: 1, value: 0.15, log: true, on: { m: [1] } },
    { id: "P", label: "NLM yama", options: ["3 × 3", "5 × 5", "7 × 7"], value: 1, on: { m: [2] } },
    { id: "S", label: "NLM arama penceresi", options: ["7 × 7", "11 × 11", "15 × 15", "21 × 21"], value: 2, on: { m: [2] } },
    { id: "h", label: "NLM h (≈ gürültü σ'nın katı)", min: 0.02, max: 0.5, value: 0.1, log: true, on: { m: [2] } },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.2, step: 0.005, value: 0.08 },
  ],
  seed: true,
  run(q, r) {
    // Not: yöntem gürültüsü bu parçalı sabit sahnede bilateral için çok küçüktür (σ_r bütün kenarları dışlar); NLM rampayı biraz düzleştirir.
    const f = scene(), g = f.map((v) => v + q.noise * r.normal()), P = 2 * q.P + 3, S = [7, 11, 15, 21][q.S];
    const D = (a) => (q.m === 0 ? gauss(a, q.gs) : q.m === 1 ? bilateral(a, q.ss, q.sr) : nlm(a, P, S, q.h));
    const out = D(g), mn = D(f), mnr = f.map((v, i) => v - mn[i]), rms = Math.sqrt(mnr.reduce((s, v) => s + v * v, 0) / mnr.length);
    const stripe = (a) => { let hi = 0, lo = 0, nh = 0, nl = 0; for (let y = 4; y < 18; y++) for (let x = 4; x < 18; x++) { if (Math.floor(x / 3) % 2) { hi += a[y * W + x]; nh++; } else { lo += a[y * W + x]; nl++; } } return hi / nh - lo / nl; };
    const cost = q.m === 2 ? W * W * (S * S - 1) * P * P : 0;
    const Pn = [`PSNR ${fmt(psnr(g, f), 3)} → ${fmt(psnr(out, f), 3)} dB. Doku karşıtlığı (temizde 0.40) ${fmt(stripe(out), 3)}; yöntem gürültüsü RMS ${fmt(rms, 3)}.`];
    Pn.push(q.m === 0 ? "Gauss her yerde aynı çekirdek: gürültüyle birlikte doku ve kenarlar da yumuşar; yöntem gürültüsünde bütün kenarlar ve doku görünür." : q.m === 1 ? `Bilateral, değer farkı σ_r'yi aşan komşuları dışlar: kenarlar korunur. ${q.sr <= 0.2 ? "Dokunun karşıtlığı (0.40) σ_r'nin epey üstünde: çizgiler ayrı tutuluyor." : "σ_r dokunun karşıtlığına (0.40) yaklaştı: çizgiler birbirine karışıyor ve bilateral Gauss'a dönüşüyor."}${q.sr < q.noise ? " σ_r gürültüden küçük: komşuların çoğu 'farklı' sayılıyor, gürültü az azalıyor." : ""}` : `NLM, ${S} × ${S} pencerede benzer ${P} × ${P} yamalara sahip pikselleri ortalar: çizgili dokuda aynı fazdaki çizgiler birbirine benzer ve konumdan bağımsız ağırlık alır${stripe(out) > 0.33 ? ", doku korunur" : "; ama bu gürültü ve h'de doku kısmen düzleşiyor"}. ${fmt(cost / 1e6, 3)} milyon çarpım: σ = 1 Gauss'un (7 × 7) ${fmt(cost / (49 * W * W), 3)} katı.`);
    if (q.m === 2 && rms > 0.015) Pn.push(`Yöntem gürültüsünün çoğu rampadadır: temiz rampada komşu yamalar yalnız eğim kadar farklıdır ve h'ye göre benzer sayılır; NLM rampayı biraz düzleştirir. Parçalı sabit bu sahnede bilateral'in yöntem gürültüsü (σ_r = 0.15'te 0.009) daha küçüktür; NLM'nin üstünlüğü tekrarlayan doku ve yüksek gürültüde görülür.`);
    if (q.m === 2 && q.h < q.noise * 0.6) Pn.push("h, gürültüye göre küçük: gürültülü yamalar bile birbirine benzemez görünür, ağırlıkların çoğu ≈ 0 ve süzgeç az iş görür.");
    return {
      plots: [
        { title: `Çıkış: ${METHODS[q.m]}`, img: { w: W, h: W, data: Array.from(out, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Yöntem gürültüsü f − D(f) (0.5 = sıfır, ×4)", img: { w: W, h: W, data: Array.from(mnr, (v) => Math.max(0, Math.min(1, 0.5 + 4 * v))) }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "PSNR girdi → çıkış", value: `${fmt(psnr(g, f), 3)} → ${fmt(psnr(out, f), 3)} dB` }, { label: "Doku karşıtlığı (temiz 0.40)", value: stripe(out) }, { label: "Yöntem gürültüsü RMS", value: rms }],
      note: Pn.join(" "),
    };
  },
};
