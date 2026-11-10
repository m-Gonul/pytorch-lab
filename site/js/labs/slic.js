// M9.11 — SLIC süper pikselleri (Achanta ve ark. 2012, ders slaytlarının adımları): s = √(n_tp/n_sp) aralıklı ızgarada başlat,
// merkezi 3 × 3'te en küçük gradyana taşı, her merkez için 2s × 2s penceredeki piksellere D = √((d_c/c)² + (d_s/s)²) uzaklığıyla
// ata, merkezleri güncelle. Ölçütler: sınır geri çağırma (gerçek sınırın 1 piksel içinde süper piksel sınırı), ulaşılabilir
// bölütleme doğruluğu (ASA) ve isteğe bağlı SLIC + k-means (süper piksel ortalamalarını k = 5 kümeye ayırıp saflık).
import { fmt, range } from "./plot.js";
import { scene, kmeans } from "./segcluster.js";

const W = 64;
export function slic(z, nsp, c, iters) {
  const n = W * W, s = Math.sqrt(n / nsp), centers = [];
  const gm = (x, y) => { const g = (a, b) => z[Math.min(W - 1, Math.max(0, b)) * W + Math.min(W - 1, Math.max(0, a))]; let t = 0; for (let d = 0; d < 3; d++) t += (g(x + 1, y)[d] - g(x - 1, y)[d]) ** 2 + (g(x, y + 1)[d] - g(x, y - 1)[d]) ** 2; return t; };
  for (let y = s / 2; y < W; y += s) for (let x = s / 2; x < W; x += s) {
    let bx = Math.round(x), by = Math.round(y), bg = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = Math.round(x) + dx, Y = Math.round(y) + dy; if (X < 0 || Y < 0 || X >= W || Y >= W) continue; const g = gm(X, Y); if (g < bg) { bg = g; bx = X; by = Y; } }
    centers.push([...z[by * W + bx], bx, by]);
  }
  const L = new Int32Array(n).fill(-1), dist = new Float64Array(n);
  const res = [];
  for (let it = 0; it < iters; it++) {
    dist.fill(Infinity);
    centers.forEach((m, i) => {
      for (let y = Math.max(0, Math.floor(m[4] - s)); y < Math.min(W, Math.ceil(m[4] + s)); y++) for (let x = Math.max(0, Math.floor(m[3] - s)); x < Math.min(W, Math.ceil(m[3] + s)); x++) {
        const p = z[y * W + x], dc = Math.hypot(p[0] - m[0], p[1] - m[1], p[2] - m[2]), ds = Math.hypot(x - m[3], y - m[4]), D = Math.hypot(dc / c, ds / s);
        if (D < dist[y * W + x]) { dist[y * W + x] = D; L[y * W + x] = i; }
      }
    });
    const acc = centers.map(() => [0, 0, 0, 0, 0, 0]);
    for (let i = 0; i < n; i++) { const l = L[i]; if (l < 0) continue; const a = acc[l]; z[i].forEach((v, d) => (a[d] += v)); a[3] += i % W; a[4] += (i / W) | 0; a[5]++; }
    let E = 0; centers.forEach((m, i) => { const a = acc[i]; if (!a[5]) return; const nm = [a[0] / a[5], a[1] / a[5], a[2] / a[5], a[3] / a[5], a[4] / a[5]]; E += Math.hypot(...nm.map((v, d) => v - m[d])); centers[i] = nm; });
    res.push(E);
  }
  return { L, centers, s, res };
}
// segcluster sahnesi + gökyüzüne yakın renkte, düşük karşıtlıklı bir disk (sınıf 4); kenarlar 3 × 3 ortalamayla yumuşatılır.
export function hardScene(r, sn) {
  const zr = Object.assign(() => 0.5, { normal: () => 0 }), { z, gt } = scene(zr, 0), g = Array.from(gt), c = z.map((v) => v.slice());
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if ((x - 52) ** 2 + (y - 46) ** 2 <= 64 && g[y * W + x] === 0) { g[y * W + x] = 4; c[y * W + x] = c[y * W + x].map((v, d) => v + [0.12, -0.05, -0.08][d]); }
  const b = c.map((_, i) => { const x = i % W, y = (i / W) | 0, a = [0, 0, 0]; let k = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= W) continue; c[Y * W + X].forEach((v, d) => (a[d] += v)); k++; } return a.map((v) => Math.max(0, Math.min(1, v / k + sn * r.normal()))); });
  return { z: b, gt: g };
}
export default {
  params: [
    { id: "nsp", label: "Süper piksel sayısı n_sp", min: 9, max: 256, value: 64, int: true, log: true },
    { id: "c", label: "Renk ölçeği c (büyük: daha kompakt)", min: 0.02, max: 2, value: 0.15, log: true },
    { id: "it", label: "İterasyon", min: 1, max: 10, value: 5, int: true },
    { id: "km", label: "Sonra", options: ["yalnız SLIC", "süper piksel ortalamalarını k-means ile 5 kümeye ayır"], value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.15, step: 0.005, value: 0.04 },
  ],
  seed: true,
  run(q, r) {
    const { z, gt } = hardScene(r, q.noise), { L, centers, s, res } = slic(z, q.nsp, q.c, q.it), n = W * W;
    const bnd = (lab, i) => { const x = i % W, y = (i / W) | 0; return (x < W - 1 && lab[i] !== lab[i + 1]) || (y < W - 1 && lab[i] !== lab[i + W]); };
    let gtB = 0, hit = 0;
    for (let i = 0; i < n; i++) if (bnd(gt, i)) { gtB++; const x = i % W, y = (i / W) | 0; let ok = false; for (let dy = -1; dy <= 1 && !ok; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < W && Y < W && bnd(L, Y * W + X)) { ok = true; break; } } if (ok) hit++; }
    const K = centers.length, ov = range(K).map(() => [0, 0, 0, 0, 0]); for (let i = 0; i < n; i++) if (L[i] >= 0) ov[L[i]][gt[i]]++;
    const asa = ov.reduce((t, o) => t + Math.max(...o), 0) / n, sizes = ov.map((o) => o.reduce((a, b) => a + b, 0)).filter((v) => v > 0), msz = sizes.reduce((a, b) => a + b, 0) / sizes.length, ssd = Math.sqrt(sizes.reduce((a, v) => a + (v - msz) ** 2, 0) / sizes.length);
    const mean = centers.map((m) => m.slice(0, 3));
    let final = L, P = [`Izgara aralığı s = ${fmt(s, 3)}; ${sizes.length} süper piksel (ortalama ${fmt(msz, 3)} ± ${fmt(ssd, 3)} piksel). Sınır geri çağırma ${fmt(100 * hit / gtB, 3)}%, ulaşılabilir doğruluk (ASA) ${fmt(100 * asa, 3)}%.`];
    P.push(q.c < 0.08 ? "c küçük: renk uzaklığı baskın, süper pikseller sınırlara sıkı yapışır ama biçimleri düzensizleşir (gürültüye de uyarlar)." : q.c > 0.6 ? "c büyük: konum baskın, süper pikseller ızgaraya benzer kompakt kareler olur ve sınırları kaçırır." : "Orta c: sınırlara yapışma ile düzenli biçim arasında denge.");
    let pur = null;
    if (q.km === 1) {
      const used = mean.map((m, i) => i).filter((i) => ov[i].some((v) => v > 0)), km = kmeans(used.map((i) => mean[i]), Math.min(5, used.length), r, 30, true), map = new Int32Array(K).fill(0);
      used.forEach((i, j) => (map[i] = km.lab[j]));
      final = L.map((l) => map[l]);
      let sm = 0; for (let j = 0; j < 5; j++) { const cc = [0, 0, 0, 0, 0]; final.forEach((l, i) => { if (l === j) cc[gt[i]]++; }); sm += Math.max(...cc); } pur = sm / n;
      P.push(`SLIC + k-means: ${used.length} süper piksel ortalaması k-means++ ile 5 kümeye ayrıldı (sahnede 5 bölge); saflık ${fmt(100 * pur, 3)}%. Kümeleme ${n} piksel yerine ${used.length} vektörle yapıldı (alıştırma 6.1).`);
    }
    const meanImg = []; for (let i = 0; i < n; i++) { const l = L[i] < 0 ? 0 : L[i]; meanImg.push(...mean[l]); }
    const over = []; for (let i = 0; i < n; i++) { if (bnd(final, i)) over.push(1, 1, 1); else over.push(...z[i]); }
    return {
      plots: [
        { title: q.km ? "SLIC + k-means bölgelerinin sınırları" : "Süper piksel sınırları (beyaz)", img: { w: W, h: W, rgb: true, data: over }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Süper piksellerin ortalama rengiyle", img: { w: W, h: W, rgb: true, data: meanImg }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Merkezlerin değişimi E (iterasyon başına)", xlabel: "iterasyon", ylabel: "E", wide: true, size: { w: 620, h: 140 }, series: [{ name: "E", x: range(res.length).map((i) => i + 1), y: res, c: 0 }] },
      ],
      stats: [{ label: "Süper piksel sayısı", value: sizes.length }, { label: "Sınır geri çağırma", value: `${fmt(100 * hit / gtB, 3)}%` }, { label: "ASA", value: `${fmt(100 * asa, 3)}%` }, { label: "Boyut std / ortalama", value: ssd / msz }, ...(pur !== null ? [{ label: "SLIC + k-means saflık", value: `${fmt(100 * pur, 3)}%` }] : [])],
      note: P.join(" "),
    };
  },
};
