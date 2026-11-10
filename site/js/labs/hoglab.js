// M9.14 — HOG (Dalal & Triggs 2005), scikit-image'in hog() tanımıyla: merkezi fark gradyanı ([−1, 0, 1], kenarlarda 0), işaretsiz
// yön (0–180°), hücre başına büyüklük ağırlıklı yön histogramı (bölmeler arası enterpolasyon yok, hücre alanına bölünür), 2 × 2 hücrelik
// bloklar (adım 1 hücre) ve blok normalizasyonu. 64 × 128 pencerede yapay bir yaya silueti; aydınlatma değişimi (kazanç, gölge
// rampası) altında betimleyicinin özgününkine kosinüs benzerliği ve doğrusal şablon puanının oranı normalizasyon türüne göre ölçülür.
import { fmt, range } from "./plot.js";

const WW = 64, HH = 128;
export function person(r, sn) {
  const f = new Float64Array(WW * HH);
  for (let y = 0; y < HH; y++) for (let x = 0; x < WW; x++) {
    let v = 0.55 + 0.08 * Math.sin(x / 3.1 + y / 5.3);
    if ((x - 32) ** 2 + (y - 22) ** 2 <= 64) v = 0.2;
    if (x >= 22 && x < 42 && y >= 31 && y < 76) v = 0.25;
    if (x >= 16 && x < 22 && y >= 34 && y < 70) v = 0.3;
    if (x >= 42 && x < 48 && y >= 34 && y < 70) v = 0.3;
    if (y >= 76 && y < 118 && ((x >= 23 && x < 31 && x - 23 >= (y - 76) * -0.05) || (x >= 33 && x < 41))) v = 0.22;
    f[y * WW + x] = v + sn * r.normal();
  }
  return f;
}
export function hog(img, cell, bins, signed, norm) {
  const gx = new Float64Array(WW * HH), gy = new Float64Array(WW * HH);
  for (let y = 0; y < HH; y++) for (let x = 1; x < WW - 1; x++) gx[y * WW + x] = img[y * WW + x + 1] - img[y * WW + x - 1];
  for (let y = 1; y < HH - 1; y++) for (let x = 0; x < WW; x++) gy[y * WW + x] = img[(y + 1) * WW + x] - img[(y - 1) * WW + x];
  const cx = Math.floor(WW / cell), cy = Math.floor(HH / cell), range_ = signed ? 360 : 180, H = range(cy * cx).map(() => new Float64Array(bins));
  for (let y = 0; y < cy * cell; y++) for (let x = 0; x < cx * cell; x++) {
    const i = y * WW + x, m = Math.hypot(gx[i], gy[i]); let a = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI; a = ((a % range_) + range_) % range_;
    const b = Math.min(bins - 1, Math.floor(a / (range_ / bins)));
    H[Math.floor(y / cell) * cx + Math.floor(x / cell)][b] += m / (cell * cell);
  }
  const by = cy - 1, bx = cx - 1, out = [];
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    let v = [...H[j * cx + i], ...H[j * cx + i + 1], ...H[(j + 1) * cx + i], ...H[(j + 1) * cx + i + 1]];
    const eps = 1e-5;
    if (norm === 1) { const n = Math.sqrt(v.reduce((s, t) => s + t * t, 0) + eps * eps); v = v.map((t) => t / n); }
    else if (norm === 2) { let n = Math.sqrt(v.reduce((s, t) => s + t * t, 0) + eps * eps); v = v.map((t) => Math.min(0.2, t / n)); n = Math.sqrt(v.reduce((s, t) => s + t * t, 0) + eps * eps); v = v.map((t) => t / n); }
    else if (norm === 3) { const n = v.reduce((s, t) => s + Math.abs(t), 0) + eps; v = v.map((t) => Math.sqrt(t / n)); }
    out.push(...v);
  }
  return { H, cx, cy, desc: out };
}
const NORMS = ["yok", "L2", "L2-Hys (Dalal–Triggs)", "L1-sqrt"];
export default {
  params: [
    { id: "cell", label: "Hücre", options: ["4 × 4", "8 × 8", "16 × 16"], value: 1 },
    { id: "bins", label: "Yön bölmesi", options: ["6", "9", "18"], value: 1 },
    { id: "signed", label: "Yön", options: ["işaretsiz (0–180°)", "işaretli (0–360°)"], value: 0 },
    { id: "norm", label: "Blok normalizasyonu", options: NORMS, value: 2 },
    { id: "gain", label: "Aydınlatma kazancı", min: 0.3, max: 2, step: 0.05, value: 0.6 },
    { id: "shadow", label: "Gölge rampası (soldan sağa çarpan)", min: 0, max: 0.8, step: 0.05, value: 0.5 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.05, step: 0.002, value: 0.01 },
  ],
  seed: true,
  run(q, r) {
    const cell = [4, 8, 16][q.cell], bins = [6, 9, 18][q.bins], f = person(r, q.noise);
    const g = f.map((v, i) => v * q.gain * (1 - q.shadow * ((i % WW) / (WW - 1))));
    const A = hog(f, cell, bins, q.signed === 1, q.norm), B = hog(g, cell, bins, q.signed === 1, q.norm);
    const dot = A.desc.reduce((s, v, i) => s + v * B.desc[i], 0), nA = A.desc.reduce((s, v) => s + v * v, 0), cos = dot / Math.sqrt(nA * B.desc.reduce((s, v) => s + v * v, 0));
    // Doğrusal sınıflandırıcı (SVM) gibi: şablon w = özgün betimleyici; puan oranı w·d(aydınlatılmış) / w·d(özgün).
    const ratio = dot / nA;
    // HOG glifi: her hücrede her bölme için, kenar yönünde (gradyana dik) uzunluğu histogramla orantılı bir çizgi.
    const gl = new Float64Array(WW * HH), mx = Math.max(...B.H.flatMap((h) => Array.from(h))) || 1;
    for (let cyy = 0; cyy < B.cy; cyy++) for (let cxx = 0; cxx < B.cx; cxx++) {
      const h = B.H[cyy * B.cx + cxx], c0x = cxx * cell + cell / 2, c0y = cyy * cell + cell / 2;
      for (let b = 0; b < bins; b++) { const ang = (((b + 0.5) * (q.signed ? 360 : 180)) / bins + 90) * (Math.PI / 180), L = (cell / 2) * (h[b] / mx); for (let t = -L; t <= L; t += 0.25) { const X = Math.round(c0x + t * Math.cos(ang)), Y = Math.round(c0y + t * Math.sin(ang)); if (X >= 0 && Y >= 0 && X < WW && Y < HH) gl[Y * WW + X] = Math.min(1, gl[Y * WW + X] + 0.5 * (h[b] / mx) + 0.2); } }
    }
    const P = [`Betimleyici boyutu: ${(B.cx - 1) * (B.cy - 1)} blok × 4 hücre × ${bins} bölme = ${B.desc.length}.`, `Aydınlatma (kazanç ${q.gain}, gölge ${q.shadow}) altında, ${NORMS[q.norm]} normalizasyonuyla: kosinüs benzerliği ${fmt(cos, 4)}, doğrusal şablon puanının oranı ${fmt(ratio, 4)} (1 = puan değişmedi).`];
    P.push(q.norm === 0 ? `Normalizasyon yok: kosinüs ölçeği zaten affeder, ama doğrusal bir sınıflandırıcının puanı kazançla birlikte ölçeklenir (oran ${fmt(ratio, 3)}): sabit bir SVM eşiği aydınlatmaya bağımlı olur.` : `Blok normalizasyonu her bloğun yerel kazancını siler: şablon puanı aydınlatmadan büyük ölçüde bağımsızlaşır (oran ${fmt(ratio, 3)}). Bedeli: zayıf gradyanlı (düz) blokları da güçlü bloklarla aynı boya getirir; gölge rampası düz arka planda gerçek bir gradyan eklediği için orada betimleyici değişir ve kosinüs normalizasyonsuz duruma göre biraz düşebilir.`);
    if (q.norm === 2) P.push("L2-Hys, L2 normalizasyonundan sonra bileşenleri 0.2'de kırpıp yeniden normalize eder: tek bir çok güçlü kenarın bloğa egemen olmasını engeller (SIFT'teki 0.2 eşiğiyle aynı fikir).");
    return {
      plots: [
        { title: "Aydınlatılmış pencere (64 × 128)", img: { w: WW, h: HH, data: Array.from(g, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 200, h: 340 }, series: [] },
        { title: "HOG glifleri (kenar yönleri)", img: { w: WW, h: HH, data: Array.from(gl) }, side: true, size: { w: 200, h: 340 }, series: [] },
      ],
      stats: [{ label: "Boyut", value: B.desc.length }, { label: "Kosinüs benzerliği (özgün ↔ aydınlatılmış)", value: cos }, { label: "Şablon puanı oranı", value: ratio }, { label: "Hücre ızgarası", value: `${B.cx} × ${B.cy}` }],
      note: P.join(" "),
    };
  },
};
