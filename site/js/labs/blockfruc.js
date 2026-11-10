// M9.17 — Blok eşleme ve kare hızı artırma (FRUC). 64 × 64 sahne: dokulu sabit arka plan üzerinde, dokulu bir kare nesne kare başına
// (dx, dy) hareket ediyor. Kare 0 ve kare 1 verilir; amaç t = 0.5'teki kareyi üretmek (gerçeği bilinen: nesne yolun ortasında).
// Yöntemler: kare tekrarı (en yakın kare), kare ortalaması, blok hareket telafili (BMC) FRUC. BMC: kare 1'in her B × B bloğu için kare 0'da
// ±R arama penceresinde en iyi eşleşme (tam arama ya da üç adımlı arama; SAD, SSD ya da ortalama/kontrast düzeltmeli SAD); blok,
// iki karedeki karşılıklarının ortalaması olarak ara karede yolun ortasına yapıştırılır. Örtüşen bölgeler ortalanır; hiçbir bloğun düşmediği
// delikler kare ortalamasıyla doldurulur. PSNR gerçek ara kareye karşı ölçülür (parlaklık farkı varsa ara karede yarısı).
import { fmt, range } from "./plot.js";

const N = 64;
function texture(r, n) { const B = range(n).map(() => [64 * r(), 64 * r(), 1.5 + 3 * r(), (r() < 0.5 ? -1 : 1) * (0.1 + 0.2 * r())]); return (x, y) => { let v = 0; for (const [cx, cy, s, a] of B) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)); return v; }; }
function scene(r) {
  const bg = texture(r, 60), ob = texture(r, 25);
  return (t, dx, dy) => { const I = new Float64Array(N * N), ox = 18 + t * dx, oy = 22 + t * dy; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const lx = x - ox, ly = y - oy; I[y * N + x] = lx >= 0 && lx < 20 && ly >= 0 && ly < 20 ? 0.75 + ob(lx * 3.2, ly * 3.2) * 0.8 : 0.4 + bg(x, y); } return I; };
}
const at = (I, x, y) => I[Math.max(0, Math.min(N - 1, y)) * N + Math.max(0, Math.min(N - 1, x))];
function cost(I0, I1, bx, by, B, vx, vy, m) {
  let s = 0, m0 = 0, m1 = 0;
  if (m === 2) { for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) { m0 += at(I0, bx + x + vx, by + y + vy); m1 += at(I1, bx + x, by + y); } m0 /= B * B; m1 /= B * B; }
  for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) { const d = at(I1, bx + x, by + y) - m1 - (at(I0, bx + x + vx, by + y + vy) - m0); s += m === 1 ? d * d : Math.abs(d); }
  return s;
}
export function search(I0, I1, bx, by, B, R, m, tss) {
  let evals = 0; const c = (vx, vy) => { evals++; return cost(I0, I1, bx, by, B, vx, vy, m); };
  if (!tss) { let best = [0, 0], bc = Infinity; for (let vy = -R; vy <= R; vy++) for (let vx = -R; vx <= R; vx++) { const k = c(vx, vy); if (k < bc - 1e-12 || (Math.abs(k - bc) <= 1e-12 && Math.hypot(vx, vy) < Math.hypot(...best))) { bc = k; best = [vx, vy]; } } return { v: best, evals }; }
  let cx = 0, cy = 0, step = 2 ** Math.floor(Math.log2(Math.max(1, R))), bc = c(0, 0);
  while (step >= 1) { let nb = [cx, cy]; for (const [ax, ay] of [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]) { const vx = cx + ax * step, vy = cy + ay * step; if (Math.abs(vx) > R || Math.abs(vy) > R) continue; const k = c(vx, vy); if (k < bc) { bc = k; nb = [vx, vy]; } } [cx, cy] = nb; step >>= 1; }
  return { v: [cx, cy], evals };
}
const psnr = (A, B) => { let s = 0; for (let i = 0; i < A.length; i++) s += (A[i] - B[i]) ** 2; return 10 * Math.log10(1 / (s / A.length)); };
export default {
  params: [
    { id: "meth", label: "Ara kare yöntemi", options: ["kare tekrarı", "kare ortalaması", "blok hareket telafili (BMC)"], value: 2 },
    { id: "dx", label: "Nesne hızı: x (piksel/kare)", min: -12, max: 12, value: 6, int: true },
    { id: "dy", label: "Nesne hızı: y (piksel/kare)", min: -12, max: 12, value: 4, int: true },
    { id: "B", label: "Blok boyutu B", options: ["4 × 4", "8 × 8", "16 × 16"], value: 1, on: { meth: [2] } },
    { id: "R", label: "Arama aralığı ±R", min: 1, max: 15, value: 7, int: true, on: { meth: [2] } },
    { id: "m", label: "Benzerlik ölçütü", options: ["SAD", "SSD", "ortalaması çıkarılmış SAD"], value: 0, on: { meth: [2] } },
    { id: "tss", label: "Arama", options: ["tam arama", "üç adımlı arama"], value: 0, on: { meth: [2] } },
    { id: "light", label: "Kare 1'de parlaklık farkı", min: -0.2, max: 0.2, step: 0.01, value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.01 },
  ],
  seed: true,
  run(q, r) {
    const S = scene(r), Bs = [4, 8, 16][q.B], nz = () => q.noise * r.normal();
    const I0 = S(0, q.dx, q.dy).map((v) => v + nz()), I1 = S(1, q.dx, q.dy).map((v) => v + q.light + nz()), T = S(0.5, q.dx, q.dy).map((v) => v + q.light / 2);
    let O, extra = "", st = [];
    if (q.meth === 0) O = Float64Array.from(I0);
    else if (q.meth === 1) O = I0.map((v, i) => (v + I1[i]) / 2);
    else {
      const acc = new Float64Array(N * N), w = new Float64Array(N * N); let evals = 0, nb = 0, okv = 0, nObj = 0;
      for (let by = 0; by < N; by += Bs) for (let bx = 0; bx < N; bx += Bs) {
        const { v, evals: e } = search(I0, I1, bx, by, Bs, q.R, q.m, q.tss === 1); evals += e; nb++;
        // gerçek: bloğun çoğu nesnedeyse (kare 1'de) hareket −(dx, dy), değilse 0
        const ox = 18 + q.dx, oy = 22 + q.dy; let inObj = 0; for (let y = by; y < by + Bs; y++) for (let x = bx; x < bx + Bs; x++) if (x - ox >= 0 && x - ox < 20 && y - oy >= 0 && y - oy < 20) inObj++;
        const tv = inObj > (Bs * Bs) / 2 ? [-q.dx, -q.dy] : [0, 0]; if (inObj > (Bs * Bs) / 2) nObj++; if (v[0] === tv[0] && v[1] === tv[1]) okv++;
        // ara karede blok, kare 1'deki konumundan v/2 kadar kare 0'a doğru kayar: (bx + v/2, by + v/2)
        for (let y = 0; y < Bs; y++) for (let x = 0; x < Bs; x++) { const X = Math.round(bx + x + v[0] / 2), Y = Math.round(by + y + v[1] / 2); if (X < 0 || Y < 0 || X >= N || Y >= N) continue; acc[Y * N + X] += (at(I1, bx + x, by + y) + at(I0, bx + x + v[0], by + y + v[1])) / 2; w[Y * N + X]++; }
      }
      let holes = 0, over = 0; O = new Float64Array(N * N); for (let i = 0; i < N * N; i++) { if (w[i] === 0) { holes++; O[i] = (I0[i] + I1[i]) / 2; } else { if (w[i] > 1) over++; O[i] = acc[i] / w[i]; } }
      st = [{ label: "Blok başına aday değerlendirme", value: fmt(evals / nb, 3) }, { label: "Doğru vektör (blok)", value: `${okv} / ${nb}` }, { label: "Delik / örtüşme (piksel)", value: `${fmt((100 * holes) / (N * N), 3)}% / ${fmt((100 * over) / (N * N), 3)}%` }];
      extra = ` Blok başına ${fmt(evals / nb, 3)} aday değerlendirildi (tam arama ${(2 * q.R + 1) ** 2}); blokların ${okv}/${nb}'ünde vektör gerçek hareketle aynı. Ara karede piksellerin ${fmt((100 * holes) / (N * N), 3)}%'i delik (kare ortalamasıyla dolduruldu), ${fmt((100 * over) / (N * N), 3)}%'i örtüşme (ortalandı).`;
      if (Math.max(Math.abs(q.dx), Math.abs(q.dy)) > q.R) extra += " Nesne hızı arama aralığını aşıyor: doğru vektör adaylar arasında yok.";
      if (q.light !== 0 && q.m !== 2) extra += " Parlaklık farkı SAD/SSD'yi yanıltıyor; ortalaması çıkarılmış ölçüt bunu siler.";
    }
    const pT = psnr(O, T), err = O.map((v, i) => Math.min(1, 4 * Math.abs(v - T[i])));
    const names = ["Kare tekrarı", "Kare ortalaması", "BMC"];
    const note = `${names[q.meth]}: gerçek ara kareye göre PSNR ${fmt(pT, 3)} dB.` + (q.meth === 1 ? " Nesne iki konumda yarı saydam görünür (hayalet)." : q.meth === 0 ? " Nesne yarım kare geride: hareket sarsıntılı görünür." : "") + extra;
    return {
      plots: [
        { title: "Kare 0", img: { w: N, h: N, data: Array.from(I0, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 200, h: 200 }, series: [] },
        { title: "Üretilen ara kare (t = 0.5)", img: { w: N, h: N, data: Array.from(O, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 200, h: 200 }, series: [] },
        { title: "Gerçeğe göre mutlak hata (×4)", img: { w: N, h: N, data: Array.from(err) }, side: true, size: { w: 200, h: 200 }, series: [] },
      ],
      stats: [{ label: "PSNR (gerçek ara kareye)", value: `${fmt(pT, 3)} dB` }, ...st],
      note,
    };
  },
};
