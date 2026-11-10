// M9.18 — Arka plan çıkarma yöntemlerinin karşılaştırılması. 64 × 64, 160 karelik yapay video: dokulu sabit arka plan, sağ üstte
// salınan "ağaç" bölgesi (dinamik arka plan: dokusu yatayda sinüsle kayar), gürültü, isteğe bağlı yavaş aydınlatma değişimi ve kamera
// titremesi; iki ön plan nesnesi (soldan sağa yürüyen kare, kare 90'da duran ve öyle kalan ikinci nesne). Yer gerçeği maskesi nesnelerin
// pikselleri. Yöntemler: kare farkı |I_t − I_{t−1}| > T; üstel hareketli ortalama B ← (1 − α)B + αI, |I − B| > T; Stauffer–Grimson
// (piksel başına K = 3 Gauss, eşleşme 2.5σ, ağırlık sırası w/σ, toplam ağırlığı 0.7'yi geçen ilk bileşenler arka plan); ViBe (N örnek,
// yarıçap R, #min = 2, φ, uzamsal yayılım). Ölçüt: kare 30–159 arası ön plan F1'i.
import { fmt, range } from "./plot.js";

export const W = 64, TT = 160;
export function makeVideo(r, o) {
  const B = range(70).map(() => [64 * r(), 64 * r(), 1.5 + 3 * r(), (r() < 0.5 ? -1 : 1) * (0.06 + 0.12 * r())]);
  const bgF = (x, y) => { let v = 0.45; for (const [cx, cy, s, a] of B) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)); return v; };
  const frames = [], masks = [];
  for (let t = 0; t < TT; t++) {
    const I = new Float64Array(W * W), M = new Uint8Array(W * W), jx = o.jit ? Math.round(o.jit * r.normal()) : 0, jy = o.jit ? Math.round(o.jit * r.normal()) : 0, gain = 1 + o.illum * Math.sin((Math.PI * t) / TT);
    const ax = -8 + t * 0.6, ay = 50, bx = Math.max(40, 70 - t * 0.33), by = 8;
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const X = x + jx, Y = y + jy; let v;
      if (X >= 40 && Y < 26) v = 0.45 + 0.18 * Math.sin(0.9 * (X + o.dyn * 3 * Math.sin(0.35 * t + 0.2 * Y)) ) * Math.cos(0.6 * Y);
      else v = bgF(X, Y);
      if (x >= ax && x < ax + 10 && y >= ay && y < ay + 12) { v = 0.8 + 0.05 * Math.sin(x + y); M[y * W + x] = 1; }
      if (o.stop && t >= 20 && x >= bx && x < bx + 9 && y >= by + 30 && y < by + 39) { v = 0.15 + 0.05 * Math.cos(x - y); M[y * W + x] = 1; }
      I[y * W + x] = gain * v + o.noise * r.normal();
    }
    frames.push(I); masks.push(M);
  }
  return { frames, masks };
}
export function vibe(frames, r, p, onFrame) {
  const N = p.N, S = new Float64Array(W * W * N), out = [];
  const nb = (x, y) => { let nx, ny; do { nx = x + Math.floor(r() * 3) - 1; ny = y + Math.floor(r() * 3) - 1; } while ((nx === x && ny === y) || nx < 0 || ny < 0 || nx >= W || ny >= W); return [nx, ny]; };
  const F0 = frames[0];
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) for (let k = 0; k < N; k++) { const [nx, ny] = nb(x, y); S[(y * W + x) * N + k] = F0[ny * W + nx]; }
  for (let t = 0; t < frames.length; t++) {
    const I = frames[t], M = new Uint8Array(W * W);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, v = I[i]; let c = 0;
      for (let k = 0; k < N && c < p.min; k++) if (Math.abs(v - S[i * N + k]) < p.R) c++;
      if (c >= p.min) {
        if (Math.floor(r() * p.phi) === 0) S[i * N + Math.floor(r() * N)] = v;
        if (p.diff && Math.floor(r() * p.phi) === 0) { const [nx, ny] = nb(x, y); S[(ny * W + nx) * N + Math.floor(r() * N)] = v; }
      } else M[i] = 1;
    }
    out.push(M); if (onFrame) onFrame(t, S);
  }
  return out;
}
function gmm(frames, alpha, K) {
  const n = W * W, mu = new Float64Array(n * K), vr = new Float64Array(n * K).fill(0.01), w = new Float64Array(n * K), out = [];
  for (let i = 0; i < n; i++) { mu[i * K] = frames[0][i]; w[i * K] = 1; for (let k = 1; k < K; k++) { mu[i * K + k] = 0.5; w[i * K + k] = 0; vr[i * K + k] = 0.05; } }
  for (const I of frames) {
    const M = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const b = i * K, x = I[i];
      // arka plan bileşenleri: w/σ'ya göre sırala, toplam ağırlık 0.7'yi geçene kadar
      const ord = range(K).sort((a, c) => w[b + c] / Math.sqrt(vr[b + c]) - w[b + a] / Math.sqrt(vr[b + a]));
      let acc = 0, bgSet = new Set(); for (const k of ord) { bgSet.add(k); acc += w[b + k]; if (acc > 0.7) break; }
      let m = -1; for (const k of ord) if (Math.abs(x - mu[b + k]) < 2.5 * Math.sqrt(vr[b + k])) { m = k; break; }
      M[i] = m >= 0 && bgSet.has(m) ? 0 : 1;
      for (let k = 0; k < K; k++) w[b + k] = (1 - alpha) * w[b + k] + (k === m ? alpha : 0);
      if (m >= 0) { const rho = alpha / Math.max(w[b + m], alpha); mu[b + m] += rho * (x - mu[b + m]); vr[b + m] = Math.max(1e-4, vr[b + m] + rho * ((x - mu[b + m]) ** 2 - vr[b + m])); }
      else { const lo = ord[K - 1]; mu[b + lo] = x; vr[b + lo] = 0.01; w[b + lo] = alpha; }
      let s = 0; for (let k = 0; k < K; k++) s += w[b + k]; for (let k = 0; k < K; k++) w[b + k] /= s;
    }
    out.push(M);
  }
  return out;
}
export function f1(masks, gts, t0 = 30) { let tp = 0, fp = 0, fn = 0; for (let t = t0; t < masks.length; t++) for (let i = 0; i < W * W; i++) { const a = masks[t][i], g = gts[t][i]; if (a && g) tp++; else if (a) fp++; else if (g) fn++; } const P = tp / Math.max(1, tp + fp), R = tp / Math.max(1, tp + fn); return { P, R, F: (2 * P * R) / Math.max(1e-12, P + R) }; }
export default {
  params: [
    { id: "meth", label: "Yöntem", options: ["kare farkı", "üstel hareketli ortalama", "Gauss karışımı (Stauffer–Grimson)", "ViBe"], value: 3 },
    { id: "T", label: "Eşik T", min: 0.02, max: 0.3, step: 0.01, value: 0.08, on: { meth: [0, 1] } },
    { id: "alpha", label: "Öğrenme oranı α", min: 0.001, max: 0.3, value: 0.02, log: true, on: { meth: [1, 2] } },
    { id: "R", label: "ViBe yarıçapı R", min: 0.02, max: 0.3, step: 0.01, value: 0.08, on: { meth: [3] } },
    { id: "phi", label: "ViBe alt örnekleme φ", min: 1, max: 64, value: 16, int: true, log: true, on: { meth: [3] } },
    { id: "dyn", label: "Dinamik arka plan (ağaç salınımı)", min: 0, max: 2, step: 0.1, value: 1 },
    { id: "illum", label: "Yavaş aydınlatma değişimi", min: 0, max: 0.4, step: 0.02, value: 0.1 },
    { id: "jit", label: "Kamera titremesi σ (piksel)", min: 0, max: 1.5, step: 0.1, value: 0 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.08, step: 0.005, value: 0.02 },
    { id: "show", label: "Gösterilen kare", min: 1, max: 159, value: 120, int: true },
  ],
  seed: true,
  run(q, r) {
    const V = makeVideo(r, { dyn: q.dyn, illum: q.illum, jit: q.jit, noise: q.noise, stop: true });
    let Ms;
    if (q.meth === 0) Ms = V.frames.map((I, t) => { const M = new Uint8Array(W * W); if (t) for (let i = 0; i < W * W; i++) M[i] = Math.abs(I[i] - V.frames[t - 1][i]) > q.T ? 1 : 0; return M; });
    else if (q.meth === 1) { const B = Float64Array.from(V.frames[0]); Ms = V.frames.map((I) => { const M = new Uint8Array(W * W); for (let i = 0; i < W * W; i++) { M[i] = Math.abs(I[i] - B[i]) > q.T ? 1 : 0; B[i] = (1 - q.alpha) * B[i] + q.alpha * I[i]; } return M; }); }
    else if (q.meth === 2) Ms = gmm(V.frames, q.alpha, 3);
    else Ms = vibe(V.frames, r, { N: 20, R: q.R, min: 2, phi: q.phi, diff: true });
    const sc = f1(Ms, V.masks), perF = range(TT).map((t) => f1([Ms[t]], [V.masks[t]], 0).F);
    const regionFP = (x0, x1, y0, y1) => { let c = 0, n = 0; for (let t = 30; t < TT; t++) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const i = y * W + x; if (!V.masks[t][i]) { n++; c += Ms[t][i]; } } return c / n; };
    const fpTree = regionFP(40, 64, 0, 26), stopIn = range(TT - 100).map((k) => { const t = 100 + k; let a = 0, g = 0; for (let i = 0; i < W * W; i++) { const x = i % W, y = (i / W) | 0; if (x >= 40 && x < 49 && y >= 38 && y < 47) { g++; a += Ms[t][i]; } } return a / Math.max(1, g); });
    const stopKeep = stopIn.reduce((a, b) => a + b, 0) / stopIn.length;
    const names = ["Kare farkı", "Üstel ortalama", "GMM", "ViBe"];
    const P = [`${names[q.meth]}: kare 30–159'da ön plan kesinliği ${fmt(sc.P, 3)}, duyarlılığı ${fmt(sc.R, 3)}, F1 ${fmt(sc.F, 3)}. Ağaç bölgesinde arka plan piksellerinin ${fmt(100 * fpTree, 3)}%'i yanlışlıkla ön plan.`];
    P.push(`Kare 20'de gelip 90'da duran nesnenin kare 100–159'da ön plan olarak kalan oranı ${fmt(stopKeep, 3)}: ${stopKeep < 0.3 ? "model duran nesneyi arka plana kattı." : stopKeep > 0.8 ? "nesne hâlâ ön plan." : "nesne arka plana kısmen katıldı."}`);
    if (q.meth === 0) P.push("Kare farkı yalnız değişen pikselleri bulur: tekdüze renkli nesnenin içi ve duran nesne kaybolur.");
    const show = q.show;
    return {
      plots: [
        { title: `Kare ${show}`, img: { w: W, h: W, data: Array.from(V.frames[show], (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 200, h: 200 }, series: [] },
        { title: "Ön plan maskesi", img: { w: W, h: W, data: Array.from(Ms[show]) }, side: true, size: { w: 200, h: 200 }, series: [] },
        { title: "Yer gerçeği", img: { w: W, h: W, data: Array.from(V.masks[show]) }, side: true, size: { w: 200, h: 200 }, series: [] },
        { title: "Kare başına F1", xlabel: "kare", ylabel: "F1", wide: true, size: { w: 620, h: 170 }, ylim: [0, 1], series: [{ name: names[q.meth], x: range(TT), y: perF, c: 0 }], marks: [{ x: 90, label: "nesne durur" }, { x: show, label: "gösterilen" }] },
      ],
      stats: [{ label: "Kesinlik / duyarlılık / F1", value: `${fmt(sc.P, 3)} / ${fmt(sc.R, 3)} / ${fmt(sc.F, 3)}` }, { label: "Ağaç bölgesinde yanlış ön plan", value: `${fmt(100 * fpTree, 3)}%` }, { label: "Duran nesne: kalan ön plan oranı", value: fmt(stopKeep, 3) }],
      note: P.join(" "),
    };
  },
};
