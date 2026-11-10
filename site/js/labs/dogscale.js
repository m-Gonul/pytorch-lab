// M9.14 — Ölçek uzayı ve DoG uç noktaları (küçük bir SIFT dedektörü). 96 × 96 sahne: std'si 2, 3.5 ve 6 olan üç Gauss leke,
// bir dikdörtgen (kenar ve köşeler), isteğe bağlı gürültü; sahne merkez çevresinde z kadar büyütülebilir. Oktav başına s + 3 Gauss
// görüntüsü (σ0 = 1.6, k = 2^{1/s}), s + 2 DoG, 3 × 3 × 3 uç nokta, kontrast eşiği |D| ≥ c ve kenar testi Tr²/Det < (r+1)²/r.
// Ayrıca leke merkezlerinde ölçek-normalize σ²∇²L ile normalize edilmemiş ∇²L'nin σ'ya göre eğrisi.
import { fmt, range } from "./plot.js";

const W = 96, BLOBS = [[26, 26, 2], [66, 30, 3.5], [36, 66, 6]];
export function scene(z, r, sn) {
  const c = W / 2, f = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const u = c + (x - c) / z, v = c + (y - c) / z;
    let val = 0.2;
    if (u >= 58 && u < 84 && v >= 56 && v < 82) val = 0.55;
    for (const [bx, by, t] of BLOBS) val += 0.6 * Math.exp(-((u - bx) ** 2 + (v - by) ** 2) / (2 * t * t));
    f[y * W + x] = val + sn * r.normal();
  }
  return f;
}
export function blur(a, n, s) {
  if (s <= 0) return Float64Array.from(a);
  const R = Math.ceil(3 * s), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / s) ** 2)), S = w.reduce((p, q) => p + q, 0), t = new Float64Array(n * n), o = new Float64Array(n * n);
  const at = (b, x, y) => b[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(a, x + j - R, y); t[y * n + x] = v / S; }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let v = 0; for (let j = 0; j <= 2 * R; j++) v += w[j] * at(t, x, y + j - R); o[y * n + x] = v / S; }
  return o;
}
export function detect(f, s, c, r, octaves = 3, s0 = 1.6, dbl = false) {
  const k = 2 ** (1 / s), kps = [];
  let n = W, img = f, sc0 = 1;
  if (dbl) { // Lowe: girdiyi 2 kat büyüt (bilineer), varsayılan bulanıklık 0.5 → 1.0; bu "−1. oktav" σ < σ0 ölçekleri yakalar.
    n = 2 * W; sc0 = 0.5; img = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const u = Math.min(W - 1.001, x / 2), v = Math.min(W - 1.001, y / 2), x0 = Math.floor(u), y0 = Math.floor(v), a = u - x0, b = v - y0; img[y * n + x] = (1 - a) * (1 - b) * f[y0 * W + x0] + a * (1 - b) * f[y0 * W + x0 + 1] + (1 - a) * b * f[(y0 + 1) * W + x0] + a * b * f[(y0 + 1) * W + x0 + 1]; }
  }
  let base = blur(img, n, Math.sqrt(s0 * s0 - (dbl ? 1 : 0.25)));
  let rejC = 0, rejE = 0, cand = 0;
  for (let o = 0; o < octaves; o++) {
    const G = [base]; for (let i = 1; i < s + 3; i++) { const sp = s0 * k ** (i - 1), sn = s0 * k ** i; G.push(blur(G[i - 1], n, Math.sqrt(sn * sn - sp * sp))); }
    const D = range(s + 2).map((i) => G[i + 1].map((v, j) => v - G[i][j]));
    for (let i = 1; i <= s; i++) for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
      const v = D[i][y * n + x]; let mx = true, mn = true;
      for (let di = -1; di <= 1 && (mx || mn); di++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { if (!di && !dy && !dx) continue; const u = D[i + di][(y + dy) * n + x + dx]; if (u >= v) mx = false; if (u <= v) mn = false; }
      if (!mx && !mn) continue;
      cand++;
      if (Math.abs(v) < c) { rejC++; continue; }
      const d = D[i], dxx = d[y * n + x + 1] + d[y * n + x - 1] - 2 * v, dyy = d[(y + 1) * n + x] + d[(y - 1) * n + x] - 2 * v, dxy = (d[(y + 1) * n + x + 1] - d[(y + 1) * n + x - 1] - d[(y - 1) * n + x + 1] + d[(y - 1) * n + x - 1]) / 4;
      const tr = dxx + dyy, det = dxx * dyy - dxy * dxy;
      if (det <= 0 || (tr * tr) / det >= ((r + 1) ** 2) / r) { rejE++; continue; }
      kps.push({ x: x * 2 ** o * sc0, y: y * 2 ** o * sc0, sigma: s0 * k ** i * 2 ** o * sc0, v });
    }
    const nb = new Float64Array((n / 2) * (n / 2)); for (let y = 0; y < n / 2; y++) for (let x = 0; x < n / 2; x++) nb[y * (n / 2) + x] = G[s][2 * y * n + 2 * x];
    base = nb; n = n / 2;
  }
  return { kps, cand, rejC, rejE };
}
export default {
  params: [
    { id: "s", label: "Oktav başına aralık s", min: 1, max: 5, value: 3, int: true },
    { id: "c", label: "Kontrast eşiği |D| (görüntü [0, 1])", min: 0.002, max: 0.08, value: 0.01, log: true },
    { id: "r", label: "Kenar oranı r", min: 1.5, max: 30, step: 0.5, value: 10 },
    { id: "z", label: "Sahne büyütme z", min: 0.6, max: 1.6, step: 0.05, value: 1 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.08, step: 0.005, value: 0.01 },
    { id: "dbl", label: "Başlangıç", options: ["görüntüyü olduğu gibi kullan", "önce 2 kat büyüt (Lowe)"], value: 1 },
  ],
  seed: true,
  run(q, r) {
    const f = scene(q.z, r, q.noise), { kps, cand, rejC, rejE } = detect(f, q.s, q.c, q.r, 4, 1.6, q.dbl === 1), cc = W / 2;
    const blobs = BLOBS.map(([bx, by, t]) => ({ x: cc + (bx - cc) * q.z, y: cc + (by - cc) * q.z, t: t * q.z }));
    const found = blobs.map((b) => { const near = kps.filter((p) => Math.hypot(p.x - b.x, p.y - b.y) <= Math.max(2.5, 0.8 * b.t)); if (!near.length) return null; return near.reduce((a, p) => (Math.abs(p.v) > Math.abs(a.v) ? p : a)); });
    // Ölçek eğrisi: en büyük lekenin merkezinde σ² ∇²L ve ∇²L
    const sig = range(25).map((i) => 1 + i * 0.5), b0 = blobs[2], cx = Math.round(b0.x), cy = Math.round(b0.y);
    const lap = sig.map((sg) => { const L = blur(f, W, sg), at = (x, y) => L[y * W + x]; return at(cx + 1, cy) + at(cx - 1, cy) + at(cx, cy + 1) + at(cx, cy - 1) - 4 * at(cx, cy); });
    const nrm = lap.map((v, i) => -v * sig[i] ** 2), raw = lap.map((v) => -v), pk = sig[nrm.indexOf(Math.max(...nrm))];
    const levels = [...new Set(kps.map((p) => p.sigma.toFixed(2)))].sort((a, b) => a - b);
    const P = [`${cand} uç nokta adayı: ${rejC} düşük kontrast, ${rejE} kenar/eyer testiyle elendi, ${kps.length} anahtar nokta kaldı.`];
    P.push(`Lekeler (std ${blobs.map((b) => fmt(b.t, 3)).join(", ")}): bulunan ölçekler ${found.map((p) => (p ? fmt(p.sigma, 3) : "—")).join(", ")}. Gauss bir leke için ölçek-normalize LoG'un tepesi σ = t'de olur. D(σ) = L(kσ) − L(σ), yaklaşık √k·σ'daki LoG'a karşılık geldiği için bulunan σ'lar t'nin biraz altındaki ızgara değerleridir (σ0·k^i).`);
    P.push(`En büyük lekenin merkezinde σ²∇²L'nin tepesi σ = ${pk} (leke std'si ${fmt(b0.t, 3)}); normalize edilmemiş ∇²L ise σ büyüdükçe sürekli azalır ve ölçek seçemez.`);
    if (!q.dbl && !found[0]) P.push("En küçük leke bulunamadı: DoG'un mutlak değeri en alt ölçek düzeyinde (σ0 = 1.6) en büyük, yani uç nokta erişilebilen ölçeklerin altında kalıyor. Lowe girdiyi önce 2 kat büyütür: σ < σ0 ölçekleri de taranır.");
    if (q.z !== 1) P.push(`Sahne ${q.z} kat büyütüldü: bulunan ölçekler de yaklaşık ${q.z} kat değişir (ölçek kovaryansı).`);
    const img = Array.from(f, (v) => Math.max(0, Math.min(1, v)));
    return {
      plots: [
        { title: "Anahtar noktalar (renk: ölçek düzeyi)", img: { w: W, h: W, data: img }, side: true, size: { w: 300, h: 290 }, series: levels.slice(0, 6).map((lv, i) => ({ name: `σ ≈ ${lv}`, x: kps.filter((p) => p.sigma.toFixed(2) === lv).map((p) => p.x + 0.5), y: kps.filter((p) => p.sigma.toFixed(2) === lv).map((p) => W - p.y - 0.5), kind: "scatter", c: i % 6, r: 2 + Number(lv) })) },
        { title: "En büyük lekenin merkezinde Laplasyen yanıtı", xlabel: "σ", ylabel: "−∇²L (ölçekli)", side: true, size: { w: 300, h: 290 }, series: [{ name: "σ²∇²L (normalize)", x: sig, y: nrm.map((v) => v / Math.max(...nrm)), c: 0 }, { name: "∇²L (ham)", x: sig, y: raw.map((v) => v / Math.max(...raw)), c: 1, dash: true }], marks: [{ x: b0.t, label: "leke std" }] },
      ],
      stats: [{ label: "Aday / kontrast / kenar / kalan", value: `${cand} / ${rejC} / ${rejE} / ${kps.length}` }, { label: "Lekelerde bulunan ölçek", value: found.map((p) => (p ? fmt(p.sigma, 3) : "—")).join(", ") }, { label: "Normalize LoG tepesi (en büyük leke)", value: pk }],
      note: P.join(" "),
    };
  },
};
