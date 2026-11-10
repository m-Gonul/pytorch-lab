// M9.9 — Tam renkli işleme: bileşen bazlı mı, vektör (ya da HSI yoğunluğu) bazlı mı? 72 × 72 sahne: koyu zemin üzerinde
// toplamsal karışan kırmızı (soldan sağa gölgeli), yeşil, mavi diskler, turuncu bir çeldirici kare ve altta gri bir rampa; Gauss gürültüsü. Dört deney: (0) histogram eşitleme,
// RGB kanallarına ayrı ayrı mı HSI yoğunluğuna mı; (1) k × k ortalama, RGB'de mi yalnız I'da mı; (2) Sobel büyüklüklerinin kanal
// toplamı mı Di Zenzo vektör gradyanı mı; (3) kırmızı bölge bölütleme: Öklid küresi, Mahalanobis elipsoidi ya da kutu.
import { fmt, range } from "./plot.js";
import { rgb2hsi, hsi2rgb } from "./colorconv.js";

const W = 72;
export function scene(r, sn, dark) {
  const c = new Float64Array(3 * W * W), gt = new Uint8Array(W * W);
  const D = [{ x: 25, y: 25, col: [0.75, 0.1, 0.1] }, { x: 47, y: 25, col: [0.1, 0.65, 0.15] }, { x: 36, y: 44, col: [0.1, 0.2, 0.75] }];
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let v = [0.06, 0.06, 0.06], n = 0, who = -1;
    D.forEach((d, k) => { if ((x - d.x) ** 2 + (y - d.y) ** 2 <= 15 * 15) { const sh = k === 0 ? 0.55 + (0.7 * (x - 10)) / 30 : 1; v = v.map((a, i) => a + sh * d.col[i]); n++; who = k; } });
    if (x >= 56 && x < 69 && y >= 44 && y < 58) v = [0.8, 0.36, 0.1];
    if (y >= 62) v = [1, 1, 1].map(() => 0.05 + (0.9 * x) / (W - 1));
    if (n === 1 && who === 0 && y < 62) gt[y * W + x] = 1;
    for (let i = 0; i < 3; i++) c[3 * (y * W + x) + i] = Math.max(0, Math.min(1, dark * Math.min(1, v[i]) + sn * r.normal()));
  }
  return { c, gt };
}
const he = (vals) => { const h = new Array(256).fill(0); vals.forEach((v) => h[Math.round(v * 255)]++); let s = 0; const T = h.map((n) => (s += n) / vals.length); return vals.map((v) => T[Math.round(v * 255)]); };
const box = (a, k) => { const h = (k - 1) / 2, o = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let s = 0, n = 0; for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) { const X = x + i, Y = y + j; if (X >= 0 && Y >= 0 && X < W && Y < W) { s += a[Y * W + X]; n++; } } o[y * W + x] = s / n; } return o; };
const chan = (c, i) => Float64Array.from(range(W * W), (k) => c[3 * k + i]);
const merge = (R, G, B) => { const o = new Float64Array(3 * W * W); for (let k = 0; k < W * W; k++) { o[3 * k] = R[k]; o[3 * k + 1] = G[k]; o[3 * k + 2] = B[k]; } return o; };
const toHSI = (c) => range(W * W).map((k) => rgb2hsi(c[3 * k], c[3 * k + 1], c[3 * k + 2]));
const fromHSI = (H, S, I) => { const o = new Float64Array(3 * W * W); for (let k = 0; k < W * W; k++) { const v = hsi2rgb(H[k], S[k], I[k]); for (let i = 0; i < 3; i++) o[3 * k + i] = Math.max(0, Math.min(1, v[i])); } return o; };
const sob = (a) => { const at = (x, y) => a[Math.min(W - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))], gx = new Float64Array(W * W), gy = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { gx[y * W + x] = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)) / 8; gy[y * W + x] = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)) / 8; } return [gx, gy]; };
export function dizenzo(gs) { // gs: [[gx, gy] × kanallar] → en büyük değişim hızı F
  const n = gs[0][0].length, F = new Float64Array(n);
  for (let k = 0; k < n; k++) { let xx = 0, yy = 0, xy = 0; for (const [gx, gy] of gs) { xx += gx[k] * gx[k]; yy += gy[k] * gy[k]; xy += gx[k] * gy[k]; } F[k] = Math.sqrt(0.5 * (xx + yy + Math.sqrt((xx - yy) ** 2 + 4 * xy * xy))); }
  return F;
}
const hdiff = (a, b) => { const d = Math.abs(a - b) % 360; return Math.min(d, 360 - d); };
const MODES = ["histogram eşitleme", "k × k ortalama ile düzleştirme", "renk gradyanı", "renk bölütleme"];
export default {
  params: [
    { id: "mode", label: "Deney", options: MODES, value: 0 },
    { id: "how", label: "Yöntem", options: ["RGB kanallarına ayrı ayrı", "HSI yoğunluğuna (H, S korunur)"], value: 1, on: { mode: [0, 1] } },
    { id: "k", label: "Çekirdek k", options: ["3", "5", "7", "9"], value: 1, on: { mode: [1] } },
    { id: "seg", label: "Bölütleme ölçütü", options: ["Öklid küresi", "Mahalanobis elipsoidi", "kutu (± t·std)"], value: 1, on: { mode: [3] } },
    { id: "D0", label: "Eşik D₀ (Öklid'de RGB birimi, diğerlerinde std)", min: 0.02, max: 4, step: 0.01, value: 2.5, on: { mode: [3] } },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.15, step: 0.005, value: 0.03 },
  ],
  seed: true,
  run(q, r) {
    const dark = q.mode === 0 ? 0.45 : 1, { c, gt } = scene(r, q.noise, dark), img = (a, rgb) => ({ w: W, h: W, rgb, data: Array.from(a) });
    let left = img(c, true), right, P = [], stats = [], lt = "Girdi", rt = "";
    if (q.mode === 0 || q.mode === 1) {
      const hs = toHSI(c), H = hs.map((h) => h.H), Sa = hs.map((h) => h.S), I = hs.map((h) => h.I);
      const k = 2 * q.k + 3;
      const viaRGB = q.mode === 0 ? merge(...[0, 1, 2].map((i) => he(Array.from(chan(c, i))))) : merge(...[0, 1, 2].map((i) => box(chan(c, i), k)));
      const viaI = q.mode === 0 ? fromHSI(H, Sa, he(I)) : fromHSI(H, Sa, Array.from(box(Float64Array.from(I), k)));
      const out = q.how === 0 ? viaRGB : viaI, ho = toHSI(out), mask = range(W * W).filter((i) => Sa[i] > 0.2 && I[i] > 0.05);
      const dH = mask.reduce((s, i) => s + hdiff(H[i], ho[i].H), 0) / mask.length, dRI = range(3 * W * W).reduce((s, i) => s + Math.abs(viaRGB[i] - viaI[i]), 0) / (3 * W * W);
      right = img(out, true); rt = `${MODES[q.mode]}: ${q.how === 0 ? "RGB'de ayrı ayrı" : "yalnız HSI yoğunluğu"}`;
      stats.push({ label: "Ortalama ton değişimi (S > 0.2 piksellerde, °)", value: dH }, { label: "Ortalama yoğunluk I (girdi → çıkış)", value: `${fmt(I.reduce((s, v) => s + v, 0) / I.length, 3)} → ${fmt(ho.reduce((s, h) => s + h.I, 0) / ho.length, 3)}` }, { label: "İki yöntemin farkı (ortalama |Δ|·255)", value: dRI * 255 });
      P.push(q.mode === 0 ? (q.how === 0 ? `Kanallar ayrı ayrı eşitlenince her kanal farklı bir dönüşüm alır: renkler kayar (ortalama ton değişimi ${fmt(dH, 3)}°). Gri rampa bile renklenebilir.` : `Yalnız I eşitlenince ton ve doygunluk korunur (ortalama ton değişimi ${fmt(dH, 3)}°; sıfırdan farkı RGB küpüne kırpmadan ve gürültüden). Görüntü aydınlanır ama renkler biraz soluk görünebilir: ders bunu doygunluğu artırarak düzeltiyor. Koyu ve gürültülü zeminde ise ton ve doygunluk gürültüden gelir; I'yı germek bu renk gürültüsünü görünür kılar (zemin rengarenk benekleniyor).`)
        : (q.how === 0 ? `Kanalları ayrı ayrı ortalamak, renk vektörlerini ortalamakla aynıdır: komşu renkler karışır, disk sınırlarında ara renkler oluşur.` : `Yalnız I'yı ortalamak H ve S'ye dokunmaz: sınırlarda renkler karışmaz, yalnız parlaklık yumuşar. İki yöntemin ortalama farkı ${fmt(dRI * 255, 3)} gri seviye; çekirdek büyüdükçe artar.`));
    } else if (q.mode === 2) {
      const gs = [0, 1, 2].map((i) => sob(chan(c, i))), sum = Float64Array.from(range(W * W), (k) => gs.reduce((s, [gx, gy]) => s + Math.hypot(gx[k], gy[k]), 0)), F = dizenzo(gs);
      const ms = Math.max(...sum), mf = Math.max(...F);
      left = img(sum.map((v) => v / ms), false); lt = "Kanal Sobel büyüklüklerinin toplamı"; right = img(F.map((v) => v / mf), false); rt = "Di Zenzo vektör gradyanı F_θ";
      // Ders örneği: üç kanalda aynı yönlü kenar (d) ile bir kanalı dik (h).
      const mk = (dirs) => dirs.map((d) => [Float64Array.of(d === 0 ? 1 : 0), Float64Array.of(d === 1 ? 1 : 0)]);
      const Fd = dizenzo(mk([0, 0, 0]))[0], Fh = dizenzo(mk([0, 0, 1]))[0];
      stats.push({ label: "Ders örneği, kanal toplamı (d) / (h)", value: "3 / 3" }, { label: "Ders örneği, Di Zenzo (d) / (h)", value: `${fmt(Fd, 4)} / ${fmt(Fh, 4)}` }, { label: "Görüntüde iki haritanın ilintisi", value: (() => { const a = Array.from(sum, (v) => v / ms), b = Array.from(F, (v) => v / mf), ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length; let sab = 0, saa = 0, sbb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; }); return sab / Math.sqrt(saa * sbb); })() });
      P.push(`Ders örneğinde üç kanalın kenarı aynı yöndeyse (d) Di Zenzo √3 = ${fmt(Fd, 4)}, biri dikse (h) √2 = ${fmt(Fh, 4)} verir; kanal toplamı ikisinde de 3'tür ve farkı göremez. Görüntüde iki harita çoğunlukla benzer: yalnız kenar bulmak için kanal toplamı yeterlidir, büyüklük ve yönün doğruluğu önemliyse vektör gradyanı gerekir.`);
    } else {
      const S = range(W * W).filter((i) => gt[i]), smp = S.filter((_, j) => j % 7 === 0), z = (i) => [c[3 * i], c[3 * i + 1], c[3 * i + 2]];
      const a = [0, 1, 2].map((d) => smp.reduce((s, i) => s + z(i)[d], 0) / smp.length);
      const Cv = [0, 1, 2].map((p) => [0, 1, 2].map((qq) => smp.reduce((s, i) => s + (z(i)[p] - a[p]) * (z(i)[qq] - a[qq]), 0) / (smp.length - 1)));
      const inv = (m) => { const [[A, B, C], [Dd, E, Fq], [G, H, I]] = m, det = A * (E * I - Fq * H) - B * (Dd * I - Fq * G) + C * (Dd * H - E * G); return [[E * I - Fq * H, C * H - B * I, B * Fq - C * E], [Fq * G - Dd * I, A * I - C * G, C * Dd - A * Fq], [Dd * H - E * G, B * G - A * H, A * E - B * Dd]].map((row) => row.map((v) => v / det)); };
      const Ci = inv(Cv), sd = [0, 1, 2].map((d) => Math.sqrt(Cv[d][d]));
      const inside = (i) => { const d = z(i).map((v, j) => v - a[j]); if (q.seg === 0) return Math.hypot(...d) <= q.D0 * Math.sqrt(sd.reduce((s, v) => s + v * v, 0) / 3); if (q.seg === 1) { const m = [0, 1, 2].reduce((s, p) => s + d[p] * [0, 1, 2].reduce((t, qq) => t + Ci[p][qq] * d[qq], 0), 0); return Math.sqrt(Math.max(m, 0)) <= q.D0; } return d.every((v, j) => Math.abs(v) <= q.D0 * sd[j]); };
      const m = Uint8Array.from(range(W * W), (i) => (inside(i) ? 1 : 0));
      let tp = 0, fp = 0, fn = 0; for (let i = 0; i < W * W; i++) { if (m[i] && gt[i]) tp++; else if (m[i]) fp++; else if (gt[i]) fn++; }
      const Pp = tp / Math.max(tp + fp, 1), Rr = tp / Math.max(tp + fn, 1);
      right = img(m, false); rt = `Maske: ${["Öklid küresi", "Mahalanobis elipsoidi", "kutu"][q.seg]}`;
      stats.push({ label: "Ortalama renk a", value: a.map((v) => fmt(v, 3)).join(", ") }, { label: "Kanal std", value: sd.map((v) => fmt(v, 3)).join(", ") }, { label: "Kesinlik / duyarlılık", value: `${fmt(Pp, 3)} / ${fmt(Rr, 3)}` }, { label: "F1", value: (2 * Pp * Rr) / Math.max(Pp + Rr, 1e-12) });
      P.push(`Örnekler kırmızı bölgenin ${smp.length} pikselinden. Eşik ${q.seg === 0 ? "Öklid'de ortalama std'nin" : "std biriminde"} ${q.D0} katı. ${q.seg === 0 ? "Küre bütün yönlerde aynı toleransı verir; kanalların yayılması farklıysa ya sızıntı ya kayıp olur." : q.seg === 1 ? "Mahalanobis, örneklerin kovaryansına uyan bir elipsoit kullanır: yayılmanın büyük olduğu yönde daha toleranslıdır." : "Kutu, her eksende ayrı std ile sınırlar; hesap en ucuzdur, köşeleri elipsoitten biraz geniştir."}`);
    }
    return {
      plots: [{ title: lt, img: left, side: true, size: { w: 300, h: 290 }, series: [] }, { title: rt, img: right, side: true, size: { w: 300, h: 290 }, series: [] }],
      stats,
      note: P.join(" "),
    };
  },
};
