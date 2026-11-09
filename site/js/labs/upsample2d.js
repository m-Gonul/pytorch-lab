// M4.10 — Bir görüntüyü f kat büyütmenin yolları: en yakın komşu, bilineer interpolasyon (F.interpolate, align_corners=False),
// bilineer çekirdekle başlatılmış transposed convolution (k = 2f, s = f, p = f/2) ve k'si f'ye bölünmeyen bir transposed
// convolution (dama tahtası). Yüksek çözünürlüklü gerçek görüntüye göre hata.
import { fmt, range } from "./plot.js";
import { kernel } from "./tconv1d.js";

const SCENES = ["disk", "eğik kenar", "sinüs çizgileri"];
function truth(type, X, Y) { // X, Y ∈ [0, 1)
  if (type === 0) return Math.hypot(X - 0.5, Y - 0.48) < 0.3 ? 1 : 0;
  if (type === 1) return X + 0.4 * Y > 0.7 ? 1 : 0;
  return 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * (X + 0.5 * Y));
}
export function lowres(type, n, ss = 8) { // her düşük çözünürlüklü piksel, alt-örneklerin ortalaması
  return range(n).map((i) => range(n).map((j) => { let s = 0; for (let a = 0; a < ss; a++) for (let b = 0; b < ss; b++) s += truth(type, (j + (b + 0.5) / ss) / n, (i + (a + 0.5) / ss) / n); return s / (ss * ss); }));
}
export function nearest(A, f) { const n = A.length; return range(n * f).map((i) => range(n * f).map((j) => A[Math.floor(i / f)][Math.floor(j / f)])); }
export function bilinear(A, f) { // align_corners=False: kaynak koordinat (i + 0.5)/f − 0.5, kenarda kırpılır
  const n = A.length, cl = (v) => Math.min(n - 1, Math.max(0, v));
  return range(n * f).map((i) => range(n * f).map((j) => {
    const y = Math.max(0, (i + 0.5) / f - 0.5), x = Math.max(0, (j + 0.5) / f - 0.5), y0 = Math.floor(y), x0 = Math.floor(x), dy = y - y0, dx = x - x0;
    const g = (a, b) => A[cl(a)][cl(b)];
    return (1 - dy) * ((1 - dx) * g(y0, x0) + dx * g(y0, x0 + 1)) + dy * ((1 - dx) * g(y0 + 1, x0) + dx * g(y0 + 1, x0 + 1));
  }));
}
export function tconv2d(A, W, s, p) {
  const n = A.length, k = W.length, full = (n - 1) * s + k, O = range(full).map(() => new Array(full).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) O[i * s + u][j * s + v] += A[i][j] * W[u][v];
  return O.slice(p, full - p).map((r) => r.slice(p, full - p));
}
const outer = (a, b) => a.map((x) => b.map((y) => x * y));

export default {
  params: [
    { id: "scene", label: "Görüntü", options: SCENES, value: 0 },
    { id: "n", label: "Düşük çözünürlük n × n", min: 4, max: 16, value: 8, int: true },
    { id: "f", label: "Büyütme çarpanı f", options: ["2", "4", "8"], value: 1 },
    { id: "m", label: "Yöntem", options: ["en yakın komşu", "bilineer interpolasyon", "transposed conv, bilineer çekirdek (k = 2f)", "transposed conv, düz çekirdek (k = f + 1)"], value: 3 },
  ],
  run(q) {
    const f = [2, 4, 8][q.f], A = lowres(q.scene, q.n), N = q.n * f;
    let U, note;
    if (q.m === 0) { U = nearest(A, f); note = "En yakın komşu her pikseli f × f bloğa kopyalar: basamaklı kenarlar."; }
    else if (q.m === 1) { U = bilinear(A, f); note = "Bilineer interpolasyon dört komşunun ağırlıklı ortalaması; kenarda görüntü sınırı kırpılarak tekrarlanır."; }
    else if (q.m === 2) { const w = kernel(0, 2 * f, f); U = tconv2d(A, outer(w, w), f, f / 2); note = `Bilineer çekirdekli transposed conv (k = ${2 * f}, s = ${f}, p = ${f / 2}): iç bölgede bilineer interpolasyonla <b>aynı</b>; yalnız en dış yarım pikselde katkı eksik kaldığı için değer düşer (F.interpolate kenarı tekrarlar). FCN'in yukarı örnekleme katmanları böyle başlatılır, sonra öğrenilir.`; }
    else { const k = f + 1, w = range(k).map(() => f / k); /* eksen başına ortalama katkı k/f: ortalama parlaklık korunur */ U = tconv2d(A, outer(w, w), f, Math.floor((k - f) / 2)); U = U.slice(0, N).map((r) => r.slice(0, N)); while (U.length < N) U.push(new Array(N).fill(0)); note = `<b>Dama tahtası:</b> k = ${f + 1} adım ${f}'ye bölünmüyor; bazı çıkış satır/sütunlarına iki girdi pikselinin katkısı düşüyor, bazılarına bir. Düz bir bölgede bile parlaklık ızgara biçiminde dalgalanıyor (Odena ve ark. 2016). Çare: k'yı s'nin katı seçmek ya da önce interpolasyonla büyütüp sonra normal konvolüsyon (resize-conv) uygulamak.`; }
    const T = range(N).map((i) => range(N).map((j) => { let s = 0; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) s += truth(q.scene, (j + (b + 0.5) / 4) / N, (i + (a + 0.5) / 4) / N); return s / 16; }));
    const err = Math.sqrt(U.flat().reduce((s, v, k) => s + (v - T.flat()[k]) ** 2, 0) / (N * N));
    const cl = (M) => M.flat().map((v) => Math.min(1, Math.max(0, v))), errIn = (() => { let s = 0, c = 0; for (let i = f; i < N - f; i++) for (let j = f; j < N - f; j++) { s += (U[i][j] - T[i][j]) ** 2; c++; } return Math.sqrt(s / c); })();
    return {
      plots: [
        { title: `Girdi ${q.n} × ${q.n}`, img: { w: q.n, h: q.n, data: cl(A) }, side: true, size: { w: 310, h: 300 } },
        { title: `Büyütülmüş ${N} × ${N}`, img: { w: N, h: N, data: cl(U) }, side: true, size: { w: 310, h: 300 } },
        { title: "Gerçek yüksek çözünürlük", img: { w: N, h: N, data: cl(T) }, side: true, size: { w: 310, h: 300 } },
      ],
      stats: [
        { label: "RMSE (gerçeğe göre, tüm görüntü)", value: err },
        { label: "RMSE (kenardan f piksel içeride)", value: errIn },
        { label: "Çıkış boyutu", value: `${N} × ${N}` },
      ],
      note: `${note} Hiçbir sabit interpolasyon kaybolan ayrıntıyı geri getiremez (keskin kenarlar bulanık kalır); FCN'in ve U-Net'in öğrenilen yukarı örnekleme katmanları ve atlama bağlantıları ayrıntıyı daha sığ, yüksek çözünürlüklü haritalardan alır.`,
    };
  },
};
