// M4.13 — Hizalamayı öğrenmek: kaynak tokenları x_1..x_n, hedef y_t = x_{π(t)} (π: özdeşlik, ters, komşu takası, döndürme ya
// da sabit rastgele permütasyon). Kodlayıcı h_i = E[x_i] + P[i] (bağlamsız: yalnız token + konum), kod çözücü durumu s_t = Q[t]
// (yalnız adım). Skor fonksiyonları: nokta s·h, genel sᵀWh, toplamsal (Bahdanau) vᵀtanh(W₁h + W₂s) ve slayttaki "tek nöronlu
// yoğun katman" a·s + b·h. α = softmax_i(e), c_t = Σ α_i h_i, logit = O c_t + o. Çapraz entropi, gradyanlar elle, Adam.
import { fmt, range } from "./plot.js";

const PERM = [(n) => range(n), (n) => range(n).map((t) => n - 1 - t), (n) => range(n).map((t) => (t ^ 1) < n ? t ^ 1 : t), (n) => range(n).map((t) => (t + n - 1) % n)];
export function perm(task, n, r) { if (task < 4) return PERM[task](n); const p = range(n); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; }

const mat = (a, b, s, r) => range(a).map(() => range(b).map(() => s * r.normal()));
const zeros = (M) => M.map((row) => row.map(() => 0));
export function init(V, n, d, r) {
  return { E: mat(V, d, 0.5, r), P: mat(n, d, 0.5, r), Q: mat(n, d, 0.5, r), W: mat(d, d, 0.3, r), W1: mat(d, d, 0.3, r), W2: mat(d, d, 0.3, r), v: [mat(1, d, 0.3, r)[0]], a: [mat(1, d, 0.3, r)[0]], b: [mat(1, d, 0.3, r)[0]], O: mat(V, d, 0.3, r), o: [range(V).map(() => 0)] };
}
const dot = (x, y) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; };
const mv = (M, x) => M.map((row) => dot(row, x));
const mtv = (M, g) => { const o = new Array(M[0].length).fill(0); M.forEach((row, i) => { for (let j = 0; j < row.length; j++) o[j] += row[j] * g[i]; }); return o; };
const addOuter = (G, u, w, s = 1) => { for (let i = 0; i < u.length; i++) for (let j = 0; j < w.length; j++) G[i][j] += s * u[i] * w[j]; };
const addTo = (g, x, s = 1) => { for (let i = 0; i < g.length; i++) g[i] += s * x[i]; };

// tek örnek: ileri + (istenirse) geri; G verilirse gradyanları biriktirir
export function step(M, x, y, kind, G) {
  const n = x.length, d = M.E[0].length, H = x.map((tok, i) => M.E[tok].map((e, k) => e + M.P[i][k]));
  let loss = 0, correct = 0; const A = [];
  for (let t = 0; t < n; t++) {
    const s = M.Q[t]; let e, U = null;
    if (kind === 0) e = H.map((h) => dot(s, h));
    else if (kind === 1) { const Ws = mtv(M.W, s); e = H.map((h) => dot(Ws, h)); }
    else if (kind === 2) { const w2s = mv(M.W2, s); U = H.map((h) => mv(M.W1, h).map((z, k) => Math.tanh(z + w2s[k]))); e = U.map((u) => dot(M.v[0], u)); }
    else { const as = dot(M.a[0], s); e = H.map((h) => as + dot(M.b[0], h)); }
    const m = Math.max(...e), ex = e.map((z) => Math.exp(z - m)), Z = ex.reduce((p, q) => p + q, 0), al = ex.map((z) => z / Z); A.push(al);
    const c = new Array(d).fill(0); H.forEach((h, i) => addTo(c, h, al[i]));
    const lg = M.O.map((row, j) => dot(row, c) + M.o[0][j]), lm = Math.max(...lg), le = lg.map((z) => Math.exp(z - lm)), LZ = le.reduce((p, q) => p + q, 0), p = le.map((z) => z / LZ);
    loss -= Math.log(p[y[t]] + 1e-12); if (p.indexOf(Math.max(...p)) === y[t]) correct++;
    if (!G) continue;
    const g = p.slice(); g[y[t]] -= 1;
    addOuter(G.O, g, c); addTo(G.o[0], g);
    const dc = mtv(M.O, g), da = H.map((h) => dot(dc, h)), sa = dot(al, da), de = al.map((ai, i) => ai * (da[i] - sa));
    const dH = H.map((_, i) => dc.map((z) => al[i] * z)), ds = new Array(d).fill(0);
    if (kind === 0) H.forEach((h, i) => { addTo(dH[i], s, de[i]); addTo(ds, h, de[i]); });
    else if (kind === 1) { const Ws = mtv(M.W, s); H.forEach((h, i) => { addOuter(G.W, s, h, de[i]); addTo(dH[i], Ws, de[i]); addTo(ds, mv(M.W, h), de[i]); }); }
    else if (kind === 2) H.forEach((h, i) => { const u = U[i]; addTo(G.v[0], u, de[i]); const dz = u.map((uk, k) => de[i] * M.v[0][k] * (1 - uk * uk)); addOuter(G.W1, dz, h); addOuter(G.W2, dz, s); addTo(dH[i], mtv(M.W1, dz)); addTo(ds, mtv(M.W2, dz)); });
    else H.forEach((h, i) => { addTo(G.a[0], s, de[i]); addTo(G.b[0], h, de[i]); addTo(dH[i], M.b[0], de[i]); addTo(ds, M.a[0], de[i]); });
    x.forEach((tok, i) => { addTo(G.E[tok], dH[i]); addTo(G.P[i], dH[i]); }); addTo(G.Q[t], ds);
  }
  return { loss: loss / n, acc: correct / n, A };
}

export function train(o, r) {
  const pi = perm(o.task, o.n, r), M = init(o.V, o.n, o.d, r), keys = Object.keys(M), mo = Object.fromEntries(keys.map((k) => [k, zeros(M[k])])), vo = Object.fromEntries(keys.map((k) => [k, zeros(M[k])]));
  const sample = () => { const x = range(o.n).map(() => Math.floor(r() * o.V)); return [x, pi.map((j) => x[j])]; }, curve = [];
  for (let it = 1; it <= o.steps; it++) {
    const G = Object.fromEntries(keys.map((k) => [k, zeros(M[k])])); let L = 0, Acc = 0;
    for (let b = 0; b < 16; b++) { const [x, y] = sample(), s = step(M, x, y, o.kind, G); L += s.loss / 16; Acc += s.acc / 16; }
    if (it === 1 || it % Math.max(1, Math.floor(o.steps / 40)) === 0 || it === o.steps) curve.push([it, L, Acc]);
    const c1 = 1 - 0.9 ** it, c2 = 1 - 0.999 ** it;
    for (const k of keys) M[k].forEach((row, i) => row.forEach((_, j) => { const g = G[k][i][j] / 16; mo[k][i][j] = 0.9 * mo[k][i][j] + 0.1 * g; vo[k][i][j] = 0.999 * vo[k][i][j] + 0.001 * g * g; row[j] -= (o.lr * mo[k][i][j]) / c1 / (Math.sqrt(vo[k][i][j] / c2) + 1e-8); }));
  }
  let acc = 0, al = 0, rowvar = 0; const T = 200;
  for (let k = 0; k < T; k++) { const [x, y] = sample(), s = step(M, x, y, o.kind, null); acc += s.acc / T; al += s.A.filter((a, t) => a.indexOf(Math.max(...a)) === pi[t]).length / o.n / T; const mean = range(o.n).map((i) => s.A.reduce((q, a) => q + a[i], 0) / o.n); rowvar += Math.sqrt(s.A.reduce((q, a) => q + a.reduce((u, ai, i) => u + (ai - mean[i]) ** 2, 0), 0) / (o.n * o.n)) / T; }
  const [x, y] = sample(), ex = step(M, x, y, o.kind, null);
  return { pi, curve, acc, al, rowvar, ex, x, y };
}

export default {
  params: [
    { id: "kind", label: "Skor fonksiyonu", options: ["nokta: s·h (Luong dot)", "genel: sᵀWh (Luong general)", "toplamsal: vᵀtanh(W₁h + W₂s) (Bahdanau)", "tek nöron: a·s + b·h (tanh'sız, slayttaki tarif)"], value: 2 },
    { id: "task", label: "Hizalama π", options: ["özdeşlik", "ters (y_t = x_{n−1−t})", "komşu takası", "bir kaydırma", "sabit rastgele permütasyon"], value: 1 },
    { id: "n", label: "Dizi uzunluğu n", min: 3, max: 10, value: 6, int: true },
    { id: "V", label: "Sözlük boyu V", min: 4, max: 30, value: 10, int: true },
    { id: "d", label: "Gizli boyut d", min: 2, max: 24, value: 12, int: true },
    { id: "steps", label: "Eğitim adımı (batch 16)", min: 10, max: 800, value: 300, int: true, log: true },
    { id: "lr", label: "Adam öğrenme oranı", min: 0.001, max: 0.2, value: 0.03, log: true },
  ],
  seed: true,
  validate(q) { return q.d * q.d * q.n * q.n * q.steps > 2e7 ? [{ level: "warn", msg: "Büyük ayar: eğitim birkaç saniye sürebilir." }] : []; },
  run(q, r) {
    const R = train(q, r), last = R.curve.at(-1);
    const P = [`Eğitim sonrası 200 yeni dizide token doğruluğu ${fmt(R.acc, 3)} (rastgele tahmin 1/V = ${fmt(1 / q.V, 2)}); dikkat argmax'ının doğru kaynak konumuna (π(t)) düştüğü adımların oranı ${fmt(R.al, 3)}.`];
    if (q.kind === 3) P.push(`<b>Tek nöronlu skor</b>: e_{t,i} = a·s_t + b·h_i. a·s_t terimi i'ye bağlı değil ve softmax'ta sadeleşiyor, yani ağırlıklar kod çözücünün adımına bağlı olamaz: her t için aynı α satırı (satırlar arası standart sapma ${fmt(R.rowvar, 3)}). Model hangi konumun okunacağını adım adım değiştiremez; hiçbir hizalamayı, özdeşliği bile öğrenemez: her adım aynı karışımı okur ve doğruluk 1/V'nin biraz üstünde kalır. Bahdanau'nun hizalama modeli bu yüzden tanh'lı bir gizli katman içerir.`);
    else P.push(`α satırları adımdan adıma değişiyor (satırlar arası standart sapma ${fmt(R.rowvar, 3)}): kod çözücünün durumu s_t, hangi kaynak konumun okunacağını seçiyor.`);
    if (q.kind === 0 && q.d < q.n) P.push(`Nokta skorunda d = ${q.d} < n = ${q.n}: konum vektörlerinin s_t ile ayrıştırılabilmesi için yeterli boyut olmayabilir.`);
    return {
      plots: [
        { title: `Dikkat ağırlıkları α (örnek dizi; satır: hedef adımı t, sütun: kaynak i)`, xlabel: "kaynak konumu i", ylabel: "hedef adımı t", size: { w: 310, h: 300 }, img: { w: q.n, h: q.n, data: R.ex.A.flat() }, side: true, series: [] },
        { title: "Eğitim", xlabel: "adım", ylabel: "kayıp / doğruluk", yzero: true, series: [
          { name: "kayıp (CE)", x: R.curve.map((c) => c[0]), y: R.curve.map((c) => c[1]), c: 3 },
          { name: "batch doğruluğu", x: R.curve.map((c) => c[0]), y: R.curve.map((c) => c[2]), c: 1 }] },
      ],
      stats: [
        { label: "Token doğruluğu (200 yeni dizi)", value: R.acc },
        { label: "Hizalama doğruluğu: argmax α = π(t)", value: R.al },
        { label: "α satırları arası std (0: her adım aynı)", value: R.rowvar },
        { label: "Son batch kaybı", value: last[1] },
        { label: "Örnek: kaynak → hedef", value: `${R.x.join(" ")} → ${R.y.join(" ")}` },
      ],
      note: P.join(" "),
    };
  },
};
