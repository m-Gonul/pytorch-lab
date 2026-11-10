// M9.8 — İkili morfoloji. 64 × 64 temiz desen: delikli dikdörtgen, ince köprüyle bağlı iki disk, 1 piksellik çıkıntılı L,
// küçük bir kare; isteğe bağlı tuz (arka planda 1) ve biber (ön planda 0) gürültüsü. İşlemler ders kitabı tanımlarıyla:
// erozyon {z | (B)_z ⊆ A}, genişleme {z | (B̂)_z ∩ A ≠ ∅}; görüntü dışı arka plan (0) sayılır.
import { fmt, range } from "./plot.js";

export const W = 64;
export function clean() {
  const a = new Uint8Array(W * W), set = (x, y) => { if (x >= 0 && y >= 0 && x < W && y < W) a[y * W + x] = 1; };
  for (let y = 6; y < 26; y++) for (let x = 6; x < 30; x++) set(x, y);
  for (let y = 13; y < 17; y++) for (let x = 15; x < 19; x++) a[y * W + x] = 0;
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if ((x - 42) ** 2 + (y - 12) ** 2 <= 49 || (x - 56) ** 2 + (y - 22) ** 2 <= 30) set(x, y);
  for (let k = 0; k < 10; k++) { set(47 + k, 16 + k); set(48 + k, 16 + k); }
  for (let y = 36; y < 58; y++) for (let x = 8; x < 14; x++) set(x, y);
  for (let y = 52; y < 58; y++) for (let x = 14; x < 30; x++) set(x, y);
  for (let x = 30; x < 38; x++) set(x, 54);
  for (let y = 40; y < 43; y++) for (let x = 44; x < 47; x++) set(x, y);
  for (let y = 46; y < 58; y++) for (let x = 50; x < 60; x++) set(x, y);
  return a;
}
export function se(shape, n) {
  const h = (n - 1) / 2, o = [];
  for (let j = -h; j <= h; j++) for (let i = -h; i <= h; i++) {
    const ok = shape === 0 ? true : shape === 1 ? i === 0 || j === 0 : shape === 2 ? i * i + j * j <= h * h + 0.5 : shape === 3 ? j === 0 : i === 0;
    if (ok) o.push([i, j]);
  }
  return o;
}
const get = (a, x, y) => (x < 0 || y < 0 || x >= W || y >= W ? 0 : a[y * W + x]);
export const erode = (a, B) => Uint8Array.from(range(W * W), (k) => (B.every(([i, j]) => get(a, (k % W) + i, ((k / W) | 0) + j)) ? 1 : 0));
export const dilate = (a, B) => Uint8Array.from(range(W * W), (k) => (B.some(([i, j]) => get(a, (k % W) - i, ((k / W) | 0) - j)) ? 1 : 0));
export const open = (a, B) => dilate(erode(a, B), B), close = (a, B) => erode(dilate(a, B), B);
function label(a, val, conn8) {
  const lab = new Int32Array(W * W).fill(-1); let n = 0; const touch = [];
  for (let s = 0; s < W * W; s++) if (a[s] === val && lab[s] < 0) {
    let border = false; const st = [s]; lab[s] = n;
    while (st.length) { const k = st.pop(), x = k % W, y = (k / W) | 0; if (x === 0 || y === 0 || x === W - 1 || y === W - 1) border = true; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { if (!(dx || dy) || (!conn8 && dx && dy)) continue; const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= W) continue; const m = Y * W + X; if (a[m] === val && lab[m] < 0) { lab[m] = n; st.push(m); } } }
    touch.push(border); n++;
  }
  return { lab, n, touch };
}
export const comps = (a) => label(a, 1, true).n;
export const holes = (a) => label(a, 0, false).touch.filter((t) => !t).length;
export function fill(a) { const L = label(a, 0, false); return Uint8Array.from(a, (v, k) => (v || !L.touch[L.lab[k]] ? 1 : 0)); }
export function skeleton(a, B) {
  const S = new Uint8Array(W * W); let e = a, k = 0;
  while (e.some((v) => v) && k < 40) { const o = open(e, B); e.forEach((v, i) => { if (v && !o[i]) S[i] = 1; }); e = erode(e, B); k++; }
  return { S, k };
}
const OPS = ["erozyon A ⊖ B", "genişleme A ⊕ B", "açma A ∘ B", "kapama A • B", "açma, sonra kapama", "sınır A − (A ⊖ B)", "delik doldurma", "iskelet (Lantuéjoul)"];
const SHAPES = ["kare", "artı (+)", "disk", "yatay çizgi", "dikey çizgi"];
export default {
  params: [
    { id: "op", label: "İşlem", options: OPS, value: 4 },
    { id: "shape", label: "Yapı elemanı", options: SHAPES, value: 0 },
    { id: "n", label: "Boyut", options: ["3", "5", "7", "9"], value: 0 },
    { id: "p", label: "Gürültü olasılığı (tuz + biber)", min: 0, max: 0.2, step: 0.01, value: 0.05 },
  ],
  seed: true,
  run(q, r) {
    const c = clean(), n = 2 * q.n + 3, B = se(q.shape, n);
    const a = c.map((v) => (r() < q.p ? 1 - v : v));
    let out, extra = "";
    if (q.op === 0) out = erode(a, B); else if (q.op === 1) out = dilate(a, B); else if (q.op === 2) out = open(a, B); else if (q.op === 3) out = close(a, B);
    else if (q.op === 4) out = close(open(a, B), B);
    else if (q.op === 5) { const e = erode(a, B); out = a.map((v, i) => (v && !e[i] ? 1 : 0)); }
    else if (q.op === 6) out = fill(a);
    else { const s = skeleton(a, B); out = s.S; extra = ` İskelet ${s.k} erozyon adımında tamamlandı.`; }
    const err = (x) => x.reduce((s, v, i) => s + (v !== c[i] ? 1 : 0), 0), cnt = (x) => x.reduce((s, v) => s + v, 0);
    const P = [`|A| = ${cnt(a)} → ${cnt(out)} piksel; bileşen ${comps(a)} → ${comps(out)} (temiz desende ${comps(c)}); delik ${holes(a)} → ${holes(out)} (temizde ${holes(c)}).`];
    if (q.op <= 4) P.push(`Temiz desenden farklı piksel: girdi ${err(a)}, çıkış ${err(out)}.`);
    P.push([
      `Erozyon, B'nin tamamen sığdığı konumları tutar: ${n} genişlikli ${SHAPES[q.shape]} ile ${n}'den ince yapılar (köprü, çıkıntı, küçük kare) silinir; ön plandaki biber delikleri de büyür.`,
      "Genişleme, B̂'nin A'ya değdiği konumları tutar: nesneler kalınlaşır, delikler ve boşluklar kapanır, tuz noktaları B büyüklüğünde lekelere döner.",
      "Açma = erozyon + genişleme: B'nin sığdığı ötelemelerin birleşimi. B'den küçük tuz noktaları, ince köprü ve çıkıntı gider; kalan nesneler yaklaşık eski boyutundadır.",
      "Kapama = genişleme + erozyon: B'den küçük delikler ve dar boşluklar kapanır, nesneler eski boyutuna döner. Asıl delik B'den büyükse kalır.",
      "Açma tuzu, ardından kapama biberi temizler: slaytlardaki gürültü giderici. İnce yapılar (köprü, çıkıntı) da açmada kaybolur.",
      "Sınır çıkarma: A'dan erozyonunu çıkarmak, B'nin kalınlığında bir iç sınır bırakır (3 × 3'te bir piksel).",
      "Delik doldurma: kenara değmeyen arka plan bileşenleri (4-komşulukla) doldurulur; Gonzalez'in X_k = (X_{k−1} ⊕ B) ∩ Aᶜ yinelemesinin kenardan başlatılmış eşdeğeri. Gürültüdeki biber noktaları da delik sayılıp dolar.",
      "İskelet: S(A) = ∪_k (A ⊖ kB) − (A ⊖ kB) ∘ B. Her erozyon basamağında açmanın kaybettiği pikseller iskelete eklenir; gürültü iskelete çok sayıda dal ekler (budama gerekir). Bu tanımın ayrık ızgaradaki sonucu bağlı olmak zorunda değildir; bağlı ve bir piksel kalınlığında iskelet için inceltme (thinning) kullanılır.",
    ][q.op] + extra);
    const img = (x) => ({ w: W, h: W, data: Array.from(x, (v) => (v ? 1 : 0)) });
    return {
      plots: [
        { title: "Girdi A (beyaz ön plan)", img: img(a), side: true, size: { w: 300, h: 290 }, series: [] },
        { title: OPS[q.op], img: img(out), side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "Ön plan piksel", value: `${cnt(a)} → ${cnt(out)}` }, { label: "Bağlı bileşen (8)", value: `${comps(a)} → ${comps(out)}` }, { label: "Delik", value: `${holes(a)} → ${holes(out)}` }, { label: "Temizden farklı piksel", value: q.op <= 4 ? `${err(a)} → ${err(out)}` : "—" }],
      note: P.join(" "),
    };
  },
};
