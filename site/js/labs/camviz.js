// M4.8 — Küresel ortalama havuzlama (GAP) + doğrusal sınıflandırıcı → sınıf aktivasyon haritası (CAM; Zhou ve ark. 2016).
// Sabit bir konvolüsyon katmanı (yönelimli çizgi filtreleri ya da rastgele filtreler) + ReLU, GAP, softmax regresyonu; yalnız
// görüntü etiketleriyle eğitilir. CAM_c(x, y) = Σ_k w_ck f_k(x, y): sınıf skorunun uzamsal dağılımı, nesnenin yerini gösterir.
import { fmt, range } from "./plot.js";

const N = 24, CLS = ["yatay çubuk", "dikey çubuk", "çapraz çubuk"];
export function draw(cls, cx, cy, img) {
  for (let t = -4; t <= 4; t++) for (let w = 0; w < 2; w++) {
    const [x, y] = cls === 0 ? [cx + t, cy + w] : cls === 1 ? [cx + w, cy + t] : [cx + t + w, cy + t];
    if (x >= 0 && x < N && y >= 0 && y < N) img[y * N + x] = 1;
  }
}
export function filters(kind, r) {
  const F = [];
  if (kind === 0) {
    for (let o = 0; o < 4; o++) F.push(range(25).map((k) => { const i = Math.floor(k / 5) - 2, j = (k % 5) - 2, d = [Math.abs(i), Math.abs(j), Math.abs(i - j) / Math.SQRT2, Math.abs(i + j) / Math.SQRT2][o]; return d < 0.75 ? 1 : -0.25; }));
  } else for (let o = 0; o < 6; o++) { const w = range(25).map(() => r.normal()), m = w.reduce((a, b) => a + b, 0) / 25; F.push(w.map((v) => v - m)); }
  return F;
}
export function fmaps(img, F) {
  return F.map((w) => { const out = new Float64Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let s = 0; for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) { const yy = y + i, xx = x + j; if (yy >= 0 && yy < N && xx >= 0 && xx < N) s += w[(i + 2) * 5 + j + 2] * img[yy * N + xx]; } out[y * N + x] = Math.max(0, s); } return out; });
}
function sample(r, noise, cls) { const img = new Float64Array(N * N).map(() => noise * r.normal()), c = cls ?? Math.floor(3 * r()); draw(c, 5 + Math.floor(r() * 14), 5 + Math.floor(r() * 14), img); return { img, c }; }
const gap = (M) => M.map((m) => m.reduce((a, b) => a + b, 0) / (N * N));
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

export function train(q, r) {
  const F = filters(q.filt, r), n = Math.round(q.n), data = range(n).map(() => { const s = sample(r, q.noise); return { g: gap(fmaps(s.img, F)), c: s.c }; });
  const K = F.length, mu = range(K).map((k) => data.reduce((a, d) => a + d.g[k], 0) / n), sd = range(K).map((k) => Math.sqrt(data.reduce((a, d) => a + (d.g[k] - mu[k]) ** 2, 0) / n) + 1e-9);
  const X = data.map((d) => d.g.map((v, k) => (v - mu[k]) / sd[k])), W = range(3).map(() => range(K).map(() => 0)), b = [0, 0, 0];
  for (let it = 0; it < 400; it++) {
    const gW = range(3).map(() => range(K).map(() => 0)), gb = [0, 0, 0];
    X.forEach((x, i) => { const p = softmax(W.map((w, c) => w.reduce((s, v, k) => s + v * x[k], b[c]))); p.forEach((pc, c) => { const d = pc - (c === data[i].c ? 1 : 0); gb[c] += d / n; for (let k = 0; k < K; k++) gW[c][k] += (d * x[k]) / n; }); });
    for (let c = 0; c < 3; c++) { b[c] -= 0.5 * gb[c]; for (let k = 0; k < K; k++) W[c][k] -= 0.5 * (gW[c][k] + 1e-3 * W[c][k]); }
  }
  // standartlaştırmayı ağırlıklara kat: logit_c = Σ_k (W_ck/sd_k)·GAP_k + sabit → CAM ağırlıkları w_ck = W_ck/sd_k
  const w = W.map((row) => row.map((v, k) => v / sd[k])), b0 = b.map((bc, c) => bc - W[c].reduce((s, v, k) => s + (v * mu[k]) / sd[k], 0));
  const predict = (img) => { const M = fmaps(img, F), g = gap(M); return { M, p: softmax(w.map((wc, c) => wc.reduce((s, v, k) => s + v * g[k], b0[c]))) }; };
  return { F, w, b0, predict, acc: data.filter((d) => { const p = softmax(w.map((wc, c) => wc.reduce((s, v, k) => s + v * d.g[k], b0[c]))); return p.indexOf(Math.max(...p)) === d.c; }).length / n };
}
const cam = (M, wc) => { const out = new Float64Array(N * N); M.forEach((m, k) => { for (let i = 0; i < N * N; i++) out[i] += wc[k] * m[i]; }); return out; };
const norm01 = (a) => { const v = Array.from(a, (x) => Math.max(0, x)), hi = Math.max(...v); return v.map((x) => (hi > 0 ? x / hi : 0)); };
const peak = (a) => { let i = 0; a.forEach((v, k) => { if (v > a[i]) i = k; }); return [i % N, Math.floor(i / N)]; };

export default {
  params: [
    { id: "cls", label: "Test görüntüsündeki nesne", options: CLS, value: 1 },
    { id: "px", label: "Nesnenin yatay konumu", min: 5, max: 18, value: 7, int: true },
    { id: "py", label: "Nesnenin dikey konumu", min: 5, max: 18, value: 15, int: true },
    { id: "dis", label: "İkinci (başka sınıftan) nesne", options: ["yok", "var (karşı köşede)"], value: 1 },
    { id: "noise", label: "Arka plan gürültüsü σ", min: 0, max: 1, step: 0.05, value: 0.3 },
    { id: "n", label: "Eğitim görüntüsü sayısı (yalnız sınıf etiketi, konum yok)", min: 12, max: 600, value: 150, int: true, log: true },
    { id: "filt", label: "Konvolüsyon filtreleri (sabit)", options: ["4 yönelimli çizgi filtresi", "6 rastgele filtre"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const T = train(q, r), img = new Float64Array(N * N).map(() => q.noise * r.normal()), oc = (q.cls + 1) % 3, ox = 23 - q.px, oy = 23 - q.py;
    draw(q.cls, q.px, q.py, img); if (q.dis === 1) draw(oc, ox, oy, img);
    const P = T.predict(img), pred = P.p.indexOf(Math.max(...P.p)), C1 = cam(P.M, T.w[q.cls]), C2 = cam(P.M, T.w[oc]), [mx, my] = peak(C1);
    let tc = 0; const test = range(150).map(() => sample(r, q.noise)); test.forEach((s) => { const p = T.predict(s.img).p; if (p.indexOf(Math.max(...p)) === s.c) tc++; });
    const imgN = (() => { const lo = Math.min(...img), hi = Math.max(...img); return Array.from(img, (v) => (v - lo) / (hi - lo || 1)); })();
    return {
      plots: [
        { title: "Test görüntüsü", img: { w: N, h: N, data: imgN }, side: true, size: { w: 310, h: 300 } },
        { title: `CAM: "${CLS[q.cls]}"`, img: { w: N, h: N, data: norm01(C1) }, side: true, size: { w: 310, h: 300 } },
        { title: `CAM: "${CLS[oc]}"`, img: { w: N, h: N, data: norm01(C2) }, side: true, size: { w: 310, h: 300 } },
      ],
      stats: [
        { label: "Eğitim / test doğruluğu (tek nesneli görüntüler)", value: `${fmt(T.acc * 100, 3)}% / ${fmt((tc / 150) * 100, 3)}%` },
        { label: "Bu görüntü için olasılıklar", value: CLS.map((c, i) => `${c} ${fmt(P.p[i], 3)}`).join(" · ") },
        { label: `"${CLS[q.cls]}" CAM'inin tepesi → nesnenin merkezi`, value: `(${mx}, ${my}) → (${q.px}, ${q.py}); uzaklık ${fmt(Math.hypot(mx - q.px, my - q.py), 3)} piksel` },
      ],
      note: `Sınıflandırıcı yalnız "bu görüntüde hangi çubuk var" etiketiyle eğitildi; hiçbir konum bilgisi görmedi. Yine de GAP doğrusal olduğu için skor uzamsal olarak ayrışır: logit_c = Σ_k w_ck · ortalama(f_k) + b = <b>ortalama(CAM_c)</b> + b. CAM_c, her pikselin c sınıfının skoruna katkısıdır; nesnenin bulunduğu yerde parlar. ${q.dis === 1 ? `İki nesneli görüntüde her sınıfın CAM'i kendi nesnesini gösteriyor: model "${CLS[pred]}" dedi (${fmt(P.p[pred], 3)}), ama haritalar iki nesneyi de ayrı ayrı yerelleştiriyor.` : ""} ${q.filt === 1 ? "Rastgele filtrelerle öznitelikler yönelime daha az seçici: doğruluk ve haritanın keskinliği düşer." : ""} Gerçek ağlarda son konvolüsyon haritaları düşük çözünürlüklüdür (ör. 7×7), CAM görüntü boyutuna büyütülür; Grad-CAM aynı fikri GAP olmayan mimarilere gradyanlarla genelleştirir.`,
    };
  },
};
