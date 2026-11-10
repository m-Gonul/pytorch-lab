// M9.11 — Tekil nokta ve çizgi tespiti. 64 × 64 sahne: gürültülü gri zemin, 6 tekil nokta (3 güçlü, 3 zayıf), dört yönde
// 1 piksellik parlak çizgiler (0°, 45°, 90°, 135°) ve bir basamak kenarı. Nokta: 8-komşulu Laplasyen |R| ≥ T. Çizgi: Laplasyenin
// mutlak değeri ya da yalnız pozitif kısmı (negatif merkezli çekirdeğin −1 katı), ya da yönlü Z1–Z4 çekirdekleri.
import { fmt, range } from "./plot.js";

const W = 64;
export function scene(r, sn) {
  const f = new Float64Array(W * W).fill(0.3), pts = [[10, 10, 0.5], [22, 8, 0.5], [52, 12, 0.5], [12, 24, 0.12], [40, 6, 0.12], [56, 28, 0.12]];
  const lines = [];
  for (let x = 4; x < 30; x++) { f[34 * W + x] = 0.7; lines.push([x, 34, 0]); }
  for (let k = 0; k < 20; k++) { f[(38 + k) * W + 4 + k] = 0.7; lines.push([4 + k, 38 + k, 1]); }
  for (let y = 34; y < 60; y++) { f[y * W + 34] = 0.7; lines.push([34, y, 2]); }
  for (let k = 0; k < 20; k++) { f[(38 + k) * W + 60 - k] = 0.7; lines.push([60 - k, 38 + k, 3]); }
  for (let y = 16; y < 30; y++) for (let x = 30; x < 50; x++) f[y * W + x] = 0.55;
  pts.forEach(([x, y, a]) => { f[y * W + x] += a; });
  return { f: f.map((v) => v + sn * r.normal()), pts, lines };
}
const Z = [[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]];
const LAP8 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]];
const conv = (f, K) => Float64Array.from(range(W * W), (i) => { const x = i % W, y = (i / W) | 0; if (x < 1 || y < 1 || x > W - 2 || y > W - 2) return 0; let s = 0; for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) s += K[j][k] * f[(y + j - 1) * W + x + k - 1]; return s; });
const MODES = ["nokta: |Laplasyen| ≥ T", "çizgi: |Laplasyen|", "çizgi: Laplasyenin pozitif kısmı", "çizgi: yönlü çekirdek Z_k"];
const DIRS = ["yatay (0°)", "+45°", "dikey (90°)", "−45°"];
export default {
  params: [
    { id: "mode", label: "Tespit", options: MODES, value: 0 },
    { id: "dir", label: "Z_k yönü", options: DIRS, value: 0, on: { mode: [3] } },
    { id: "T", label: "Eşik T (max yanıtın oranı)", min: 0.02, max: 0.9, step: 0.01, value: 0.7 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.08, step: 0.005, value: 0.02 },
  ],
  seed: true,
  run(q, r) {
    const { f, pts, lines } = scene(r, q.noise);
    let R;
    if (q.mode <= 2) { const L = conv(f, LAP8); R = q.mode === 2 ? L.map((v) => Math.max(0, -v)) : L.map(Math.abs); }
    else R = conv(f, Z[q.dir]).map(Math.abs);
    const mx = Math.max(...R), det = Uint8Array.from(R, (v) => (v >= q.T * mx ? 1 : 0)), n = det.reduce((s, v) => s + v, 0);
    const near = (x, y, d) => { for (let j = -d; j <= d; j++) for (let k = -d; k <= d; k++) { const X = x + k, Y = y + j; if (X >= 0 && Y >= 0 && X < W && Y < W && det[Y * W + X]) return true; } return false; };
    const P = [], stats = [{ label: "Tespit edilen piksel", value: n }];
    if (q.mode === 0) {
      const strong = pts.filter((p) => p[2] > 0.3 && near(p[0], p[1], 0)).length, weak = pts.filter((p) => p[2] < 0.3 && near(p[0], p[1], 0)).length;
      const onLine = lines.filter(([x, y]) => det[y * W + x]).length;
      stats.push({ label: "Güçlü / zayıf nokta bulundu", value: `${strong}/3, ${weak}/3` }, { label: "Çizgi piksellerinde yanıt", value: onLine });
      P.push(`Güçlü noktaların ${strong}'ü, zayıfların ${weak}'ü bulundu; ${onLine} çizgi pikseli de eşiği aştı. Tek bir nokta 8-komşulu Laplasyende 8a, bir çizgi pikseli ise ancak 6a (yan komşuların ikisi de çizgide) yanıt verir; eşik 8a ile 6a arasındaysa yalnız noktalar kalır.`);
    } else {
      const byDir = [0, 1, 2, 3].map((d) => { const pl = lines.filter((l) => l[2] === d); return pl.filter(([x, y]) => det[y * W + x]).length / pl.length; });
      let thick = 0, cnt = 0; for (const [x, y, d] of lines) { if (x < 8 || x > 56 || y < 36 || y > 58) continue; if (d === 2) { let s = 0; for (let k = -2; k <= 2; k++) s += det[y * W + x + k]; thick += s; cnt++; } }
      stats.push({ label: "Çizgi pikseli bulunma oranı (0°, 45°, 90°, −45°)", value: byDir.map((v) => fmt(v, 2)).join(", ") }, { label: "Dikey çizginin ortalama kalınlığı (piksel)", value: cnt ? thick / cnt : 0 });
      P.push(`Dikey çizgi ortalama ${fmt(cnt ? thick / cnt : 0, 3)} piksel kalınlığında tespit edildi.`);
      P.push(q.mode === 1 ? "Mutlak değer, Laplasyenin çizginin iki yanındaki negatif yanıtını da alır: çizgiler kalınlaşır (çift çizgi etkisi), basamak kenarının iki yanı da yanıt verir." : q.mode === 2 ? "Yalnız pozitif kısım (parlak çizgide −∇²f > 0) çizginin kendisini verir: daha ince çizgiler; kenarın yalnız parlak tarafı kalır." : `Z_k yalnız kendi yönündeki 1 piksellik çizgilere güçlü yanıt verir. Öbür üç yöndeki tek piksellik bir çizgide çizginin geçtiği üç katsayı (−1, 2, −1) birbirini götürür ve yanıt sıfırdır; yalnız çizgi uçları, kesişimler ve kenarlar yanıt bırakır.`);
    }
    return {
      plots: [
        { title: "Girdi", img: { w: W, h: W, data: Array.from(f, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `${MODES[q.mode]}${q.mode === 3 ? ", " + DIRS[q.dir] : ""}: eşiklenmiş`, img: { w: W, h: W, data: Array.from(det) }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats,
      note: P.join(" "),
    };
  },
};
