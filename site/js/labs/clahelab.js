// M9.12 — Global HE, AHE ve CLAHE (OpenCV kuralı): 96 × 96, 8 bit sahne: sol yarı karanlık (40) üzerinde ±8 düşük karşıtlıklı
// diskler, sağ yarı parlak (200) üzerinde ±8 kareler, ortada düz gri (128) bir bant + gürültü. Döşeme başına histogram, kırpma eşiği
// max(1, ⌊clip · alan / 256⌋), fazlayı eşit dağıtma (artan kalan adımlı), LUT = round(255 · cdf / alan), dört LUT'un bilineer karışımı.
import { fmt, range } from "./plot.js";

const W = 96;
// OpenCV'nin cvRound'u gibi yarıyı çifte yuvarlama (saturate_cast<uchar>).
export const rhe = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 + 1e-9 ? f + 1 : d < 0.5 - 1e-9 ? f : f % 2 === 0 ? f : f + 1; };
export function scene(r, sn) {
  const f = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let v = x < 40 ? 40 : x >= 56 ? 200 : 128;
    if (x < 40 && [[20, 20], [20, 48], [20, 76]].some(([cx, cy]) => (x - cx) ** 2 + (y - cy) ** 2 <= 49)) v += 8;
    if (x >= 56 && [[76, 20], [76, 48], [76, 76]].some(([cx, cy]) => Math.abs(x - cx) <= 6 && Math.abs(y - cy) <= 6)) v -= 8;
    f[y * W + x] = Math.max(0, Math.min(255, Math.round(v + sn * r.normal())));
  }
  return f;
}
export function tileLUT(vals, clip) {
  const h = new Array(256).fill(0); vals.forEach((v) => h[v]++);
  const area = vals.length;
  if (clip > 0) {
    const lim = Math.max(1, Math.floor((clip * area) / 256)); let ex = 0;
    for (let i = 0; i < 256; i++) if (h[i] > lim) { ex += h[i] - lim; h[i] = lim; }
    const batch = Math.floor(ex / 256); let res = ex - batch * 256;
    for (let i = 0; i < 256; i++) h[i] += batch;
    if (res > 0) { const step = Math.max(Math.floor(256 / res), 1); for (let i = 0; i < 256 && res > 0; i += step, res--) h[i]++; }
  }
  let s = 0; const sc = 255 / area; return h.map((n) => { s += n; return rhe(Math.fround(s * sc)); });
}
export function clahe(f, g, clip, interp) {
  const ts = W / g, luts = [];
  for (let ty = 0; ty < g; ty++) for (let tx = 0; tx < g; tx++) { const v = []; for (let y = ty * ts; y < (ty + 1) * ts; y++) for (let x = tx * ts; x < (tx + 1) * ts; x++) v.push(f[y * W + x]); luts.push(tileLUT(v, clip)); }
  return Float64Array.from(range(W * W), (i) => {
    const x = i % W, y = (i / W) | 0, v = f[i];
    if (!interp) return luts[Math.floor(y / ts) * g + Math.floor(x / ts)][v];
    const fx = x / ts - 0.5, fy = y / ts - 0.5, x1 = Math.floor(fx), y1 = Math.floor(fy), ax = fx - x1, ay = fy - y1;
    const L = (tx, ty) => luts[Math.min(g - 1, Math.max(0, ty)) * g + Math.min(g - 1, Math.max(0, tx))][v];
    return (1 - ay) * ((1 - ax) * L(x1, y1) + ax * L(x1 + 1, y1)) + ay * ((1 - ax) * L(x1, y1 + 1) + ax * L(x1 + 1, y1 + 1));
  });
}
const METHODS = ["global HE", "AHE (döşemeli, kırpmasız)", "CLAHE"];
export default {
  params: [
    { id: "m", label: "Yöntem", options: METHODS, value: 2 },
    { id: "g", label: "Döşeme ızgarası g × g", options: ["2 × 2", "4 × 4", "8 × 8"], value: 1, on: { m: [1, 2] } },
    { id: "clip", label: "Kırpma sınırı (clipLimit)", min: 1, max: 40, step: 0.5, value: 2, on: { m: [2] } },
    { id: "interp", label: "Döşemeler arası", options: ["enterpolasyon yok", "bilineer enterpolasyon"], value: 1, on: { m: [1, 2] } },
    { id: "noise", label: "Gürültü σ (gri seviye)", min: 0, max: 8, step: 0.5, value: 2 },
  ],
  seed: true,
  run(q, r) {
    const f = scene(r, q.noise), g = [2, 4, 8][q.g];
    const out = q.m === 0 ? (() => { const L = tileLUT(Array.from(f), 0); return f.map((v) => L[v]); })() : clahe(f, g, q.m === 2 ? q.clip : 0, q.interp === 1).map((v) => rhe(v));
    const reg = (x0, x1, y0, y1, cond) => { const v = []; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (cond(x, y)) v.push(y * W + x); return v; };
    const mean = (a, idx) => idx.reduce((s, i) => s + a[i], 0) / idx.length, std = (a, idx) => { const m = mean(a, idx); return Math.sqrt(idx.reduce((s, i) => s + (a[i] - m) ** 2, 0) / idx.length); };
    const inD = reg(14, 27, 42, 55, (x, y) => (x - 20) ** 2 + (y - 48) ** 2 <= 16), outD = reg(2, 38, 30, 66, (x, y) => (x - 20) ** 2 + (y - 48) ** 2 >= 100);
    const flat = reg(44, 52, 4, 92, () => true);
    const cD = (a) => mean(a, inD) - mean(a, outD);
    // Düz bantta gürültü kestirimi: yatay komşu farklarının RMS'i / √2 (yavaş değişime duyarsız; döşeme sınırı sütunu hariç).
    const ts0 = W / g, nz = (a) => { let s = 0, n = 0; for (let y = 4; y < 92; y++) for (let x = 42; x < 54; x++) { if (q.m > 0 && (x + 1) % ts0 === 0) continue; s += (a[y * W + x + 1] - a[y * W + x]) ** 2; n++; } return Math.sqrt(s / n / 2); };
    let jump = 0, nj = 0, ts = W / g; if (q.m > 0) for (let y = 4; y < 92; y++) for (let k = 1; k < g; k++) { const x = k * ts; if (x > 40 && x < 56) { jump += Math.abs(out[y * W + x] - out[y * W + x - 1]); nj++; } }
    const P = [`Karanlık bölgedeki disk karşıtlığı ${fmt(cD(f), 3)} → ${fmt(cD(out), 3)} gri seviye; düz bantta gürültü (komşu farklarından) ${fmt(nz(f), 3)} → ${fmt(nz(out), 3)}.`];
    P.push(q.m === 0 ? "Global HE tek bir LUT kullanır: histogramın büyük tepeleri (40, 128, 200) yayılır ama karanlık ve parlak bölgelerdeki ±8'lik ayrıntılar yerel olarak az kazanır." : q.m === 1 ? "AHE her döşemeye kendi CDF'ini uygular: karanlık bölgedeki ayrıntı çok güçlenir, ama düz bantta histogram tek bir tepe olduğu için CDF dikleşir ve gürültü katlanarak büyür." : `CLAHE histogramı kırpma eşiğinde keser ve fazlayı bütün bölmelere dağıtır: CDF'in eğimi, yani karşıtlık kazancı sınırlanır. clipLimit = ${q.clip}; küçük değer HE'den az, büyük değer AHE'ye yakın davranır.`);
    if (q.m > 0 && g === 8) P.push(`Döşemeler 12 piksel: disklerden (çap 14) küçük. Bazı döşemeler tamamen bir diskin içinde kalır ve diski 'zemin' sayar; karşıtlık kazanmak yerine kaybolur. Döşeme, korunmak istenen ayrıntıdan büyük olmalı.`);
    if (q.m > 0) P.push(q.interp ? "Bilineer enterpolasyon dört komşu döşemenin LUT'larını karıştırır: döşeme sınırlarında süreklilik." : `Enterpolasyon yok: her döşeme kendi LUT'unu kullanır ve sınırlarda basamaklar oluşur (düz bantta döşeme sınırındaki ortalama sıçrama ${fmt(nj ? jump / nj : 0, 3)}).`);
    const lt = tileLUT(Array.from(reg(0, W / g, 0, W / g, () => true).map((i) => f[i])), q.m === 2 ? q.clip : 0);
    return {
      plots: [
        { title: "Girdi", img: { w: W, h: W, data: Array.from(f, (v) => v / 255) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `${METHODS[q.m]}${q.m ? `, ${g} × ${g}` : ""}`, img: { w: W, h: W, data: Array.from(out, (v) => Math.max(0, Math.min(1, v / 255))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Sol üst döşemenin dönüşüm fonksiyonu T(r)", xlabel: "r", ylabel: "s", wide: true, size: { w: 620, h: 170 }, series: [{ name: q.m === 2 ? "CLAHE LUT" : "kırpmasız LUT", x: range(256), y: lt, c: 0 }, { name: "kimlik", x: [0, 255], y: [0, 255], c: 4, dash: true }], xlim: [0, 255], ylim: [0, 255] },
      ],
      stats: [{ label: "Disk karşıtlığı (karanlık bölge)", value: `${fmt(cD(f), 3)} → ${fmt(cD(out), 3)}` }, { label: "Düz bant gürültüsü (komşu farkı)", value: `${fmt(nz(f), 3)} → ${fmt(nz(out), 3)}` }, { label: "Döşeme sınırında sıçrama (düz bant)", value: q.m > 0 ? jump / Math.max(nj, 1) : 0 }],
      note: P.join(" "),
    };
  },
};
