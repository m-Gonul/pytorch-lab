// M9.17 — Lucas–Kanade optik akışı (Lucas & Kanade 1981; piramit: Bouguet 2001). 96 × 96 sahne üç bölgeden oluşur: solda dokulu
// (analitik Gauss lekeleri toplamı), sağ üstte yalnız x'te değişen dikey çizgiler (açıklık problemi), sağ altta düz bölge. İkinci kare
// sahnenin analitik olarak kaydırılmış/döndürülmüş hâlidir (enterpolasyon hatası yok), isteğe bağlı kazanç ve gürültüyle. Akış 8 piksel
// aralıklı ızgarada kestirilir: her noktada w × w pencerede AᵀA m = −Aᵀ I_t; AᵀA'nın en küçük özdeğeri eşiğin altındaysa nokta reddedilir.
// Yineleme: tahmin edilen akışla ikinci kare bilineer örneklenir, kalan zamansal fark yeniden çözülür. Piramit: 2 × 2 ortalamayla küçültme.
import { fmt, range } from "./plot.js";

const N = 96;
function sceneFn(r) {
  const B = range(40).map(() => [4 + 40 * r(), 4 + 88 * r(), 2.5 + 3.5 * r(), (r() < 0.5 ? -1 : 1) * (0.15 + 0.25 * r())]);
  return (x, y) => {
    if (x >= 48 && y < 48) return 0.5 + 0.3 * Math.sin((2 * Math.PI * x) / 10);
    if (x >= 48) return 0.5;
    let v = 0.5; for (const [cx, cy, s, a] of B) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)); return v;
  };
}
const regionOf = (x, y) => (x < 48 ? 0 : y < 48 ? 1 : 2);
function sample(I, n, x, y) { x = Math.max(0, Math.min(n - 1.001, x)); y = Math.max(0, Math.min(n - 1.001, y)); const x0 = Math.floor(x), y0 = Math.floor(y), a = x - x0, b = y - y0, i = y0 * n + x0; return (1 - a) * (1 - b) * I[i] + a * (1 - b) * I[i + 1] + (1 - a) * b * I[i + n] + a * b * I[i + n + 1]; }
function down(I, n) { const m = n >> 1, O = new Float64Array(m * m); for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { const i = 2 * y * n + 2 * x; O[y * m + x] = (I[i] + I[i + 1] + I[i + n] + I[i + n + 1]) / 4; } return O; }
function grads(I, n) { const gx = new Float64Array(n * n), gy = new Float64Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const xm = Math.max(0, x - 1), xp = Math.min(n - 1, x + 1), ym = Math.max(0, y - 1), yp = Math.min(n - 1, y + 1); gx[y * n + x] = (I[y * n + xp] - I[y * n + xm]) / (xp - xm); gy[y * n + x] = (I[yp * n + x] - I[ym * n + x]) / (yp - ym); } return [gx, gy]; }
// Tek seviye, tek nokta, yinelemeli LK. Döndürür: [u, v, λmin / w²].
function lkPoint(I1, I2, gx, gy, n, px, py, u0, v0, half, iters) {
  let Sxx = 0, Sxy = 0, Syy = 0; const pts = [];
  for (let dy = -half; dy <= half; dy++) for (let dx = -half; dx <= half; dx++) { const x = Math.max(0, Math.min(n - 1, Math.round(px) + dx)), y = Math.max(0, Math.min(n - 1, Math.round(py) + dy)), i = y * n + x; Sxx += gx[i] ** 2; Sxy += gx[i] * gy[i]; Syy += gy[i] ** 2; pts.push([x, y, i]); }
  const tr = Sxx + Syy, det = Sxx * Syy - Sxy * Sxy, lmin = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  let u = u0, v = v0;
  if (det > 1e-12) for (let k = 0; k < iters; k++) {
    let bx = 0, by = 0; for (const [x, y, i] of pts) { const It = sample(I2, n, x + u, y + v) - I1[i]; bx += gx[i] * It; by += gy[i] * It; }
    const du = -(Syy * bx - Sxy * by) / det, dv = -(-Sxy * bx + Sxx * by) / det; u += du; v += dv; if (Math.hypot(du, dv) < 1e-3) break;
  }
  return [u, v, lmin / pts.length];
}
export function lkPyr(I1, I2, n, pts, win, levels, iters) {
  const P1 = [I1], P2 = [I2]; for (let l = 1; l < levels; l++) { P1.push(down(P1[l - 1], n >> (l - 1))); P2.push(down(P2[l - 1], n >> (l - 1))); }
  const G = P1.map((I, l) => grads(I, n >> l)), half = (win - 1) >> 1;
  return pts.map(([px, py]) => { let u = 0, v = 0, lm = 0; for (let l = levels - 1; l >= 0; l--) { const s = 2 ** l, m = n >> l; [u, v, lm] = lkPoint(P1[l], P2[l], G[l][0], G[l][1], m, px / s, py / s, u, v, half, iters); if (l > 0) { u *= 2; v *= 2; } } return [u, v, lm]; });
}
export default {
  params: [
    { id: "type", label: "Hareket", options: ["öteleme (45°)", "merkez çevresinde dönme"], value: 0 },
    { id: "mag", label: "Yer değiştirme büyüklüğü (piksel; dönmede 40 piksel yarıçapta)", min: 0.2, max: 12, step: 0.1, value: 1.5 },
    { id: "win", label: "Pencere w × w", min: 3, max: 15, step: 2, value: 7, int: true },
    { id: "levels", label: "Piramit seviyesi", min: 1, max: 4, value: 1, int: true },
    { id: "iters", label: "Seviye başına yineleme", min: 1, max: 10, value: 1, int: true },
    { id: "tau", label: "En küçük özdeğer eşiği τ (λmin / w²)", min: 1e-6, max: 1e-2, value: 1e-4, log: true },
    { id: "gain", label: "İkinci karede parlaklık kazancı", min: 0.7, max: 1.3, step: 0.01, value: 1 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.1, step: 0.005, value: 0.01 },
  ],
  seed: true,
  validate(q) { return q.win % 2 ? [] : [{ level: "error", msg: "Pencere w × w: w tek olmalı." }]; },
  run(q, r) {
    const f = sceneFn(r), c = (N - 1) / 2, th = q.mag / 40, ux = q.mag * Math.SQRT1_2;
    const flow = (x, y) => (q.type === 0 ? [ux, ux] : [Math.cos(th) * (x - c) - Math.sin(th) * (y - c) + c - x, Math.sin(th) * (x - c) + Math.cos(th) * (y - c) + c - y]);
    const back = (x, y) => (q.type === 0 ? [x - ux, y - ux] : [Math.cos(th) * (x - c) + Math.sin(th) * (y - c) + c, -Math.sin(th) * (x - c) + Math.cos(th) * (y - c) + c]);
    const I1 = new Float64Array(N * N), I2 = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { I1[y * N + x] = f(x, y) + q.noise * r.normal(); const [bx, by] = back(x, y); I2[y * N + x] = q.gain * f(bx, by) + q.noise * r.normal(); }
    const pts = []; for (let y = 8; y < N - 4; y += 8) for (let x = 8; x < N - 4; x += 8) if (x !== 48 && y !== 48) pts.push([x, y]);
    const est = lkPyr(I1, I2, N, pts, q.win, q.levels, q.iters);
    const reg = [[], [], []], qx = [], qy = [], rx = [], ry = [], tx = [], ty = [], S = 1.5;
    pts.forEach(([x, y], k) => {
      const [u, v, lm] = est[k], [tu, tv] = flow(x, y), g = regionOf(x, y), ok = lm >= q.tau;
      const [ix, iy] = [x + 0.5, N - y - 0.5];
      tx.push(ix, ix + S * tu, NaN); ty.push(iy, iy - S * tv, NaN);
      if (ok) { qx.push(ix, ix + S * u, NaN); qy.push(iy, iy - S * v, NaN); } else { rx.push(ix); ry.push(iy); }
      const nrm = [1, 0], normalErr = Math.abs((u - tu) * nrm[0] + (v - tv) * nrm[1]);
      reg[g].push({ ok, epe: Math.hypot(u - tu, v - tv), nErr: normalErr });
    });
    const st = reg.map((a) => { const A = a.filter((o) => o.ok); return { acc: A.length / a.length, epe: A.length ? A.reduce((s, o) => s + o.epe, 0) / A.length : NaN, nErr: A.length ? A.reduce((s, o) => s + o.nErr, 0) / A.length : NaN }; });
    const names = ["dokulu", "çizgili", "düz"];
    const part = (g) => (st[g].acc > 0 ? `EPE ${fmt(st[g].epe, 3)} piksel, noktaların ${fmt(100 * st[g].acc, 3)}%'i kabul` : "bütün noktalar reddedildi (λmin < τ)");
    const P = [`Dokulu bölge: ${part(0)}. Çizgili bölge: ${part(1)}${st[1].acc > 0 ? `; gradyan yönündeki (x) bileşenin hatası yalnız ${fmt(st[1].nErr, 3)}` : ""}. Düz bölge: ${part(2)}.`];
    if (st[1].acc > 0.5) P.push("Çizgili bölgede AᵀA neredeyse tekil (I_y ≈ 0) ama eşik onu geçirdi: dikey bileşen gürültüden belirleniyor, açıklık problemi.");
    if (q.levels === 1 && q.mag > 3) P.push("Tek seviyede yer değiştirme doğrusallaştırmanın geçerli olduğu aralığı (leke boyutu birkaç piksel) aşıyor; piramit seviyesi ekle.");
    if (st[2].acc > 0) P.push("Düz bölgede kabul edilen noktalar var: gürültü sahte gradyan üretip λmin'i eşiğin üstüne çıkarıyor, ama akış rastgele.");
    if (q.gain !== 1) P.push(`Parlaklık kazancı ${q.gain}: I_t'ye hareketten bağımsız bir terim ekleniyor, parlaklık sabitliği bozuluyor.`);
    return {
      plots: [{ title: "Kare 1 üzerinde gerçek akış, LK tahmini ve reddedilen noktalar (oklar 1.5 kat)", img: { w: N, h: N, data: Array.from(I1, (v) => Math.max(0, Math.min(1, v))) }, wide: true, size: { w: 420, h: 420 }, series: [{ name: "gerçek akış", x: tx, y: ty, c: 4 }, { name: "LK", x: qx, y: qy, c: 2 }, { name: "reddedildi", x: rx, y: ry, kind: "scatter", c: 1, r: 3 }] }],
      stats: names.map((nm, g) => ({ label: `EPE / kabul: ${nm}`, value: `${Number.isFinite(st[g].epe) ? fmt(st[g].epe, 3) : "—"} / ${fmt(100 * st[g].acc, 3)}%` })),
      note: P.join(" "),
    };
  },
};
