// M9.15 — Şablon eşleme: ham korelasyon Σw·f, SSD ve normalize korelasyon katsayısı γ (Gonzalez/ders slaytlarının tanımı;
// cv2.TM_CCOEFF_NORMED ile aynı). 72 × 72 dokulu sahnede şablon (11 × 11 çapraz desen) sol üstte; sağ yarıya isteğe bağlı
// aydınlatma kazancı/ofseti uygulanır ve şablonun ikinci bir kopyası oraya yerleştirilir; ayrıca parlak, düz bir "çeldirici" kare var.
import { fmt, range } from "./plot.js";

const W = 72, T = 11, h = (T - 1) / 2;
export function scene(r, sn, gain, bias, dist) {
  const f = new Float64Array(W * W), tpl = new Float64Array(T * T);
  for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) tpl[j * T + i] = (Math.abs(i - h) <= 1 || Math.abs(j - h) <= 1 ? 0.75 : 0.3) + 0.1 * ((i + 2 * j) % 3 === 0 ? 1 : 0);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) f[y * W + x] = 0.4 + 0.12 * Math.sin(x / 2.3) * Math.cos(y / 3.1);
  const put = (cx, cy) => { for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) f[(cy - h + j) * W + cx - h + i] = tpl[j * T + i]; };
  put(16, 18); put(54, 50);
  for (let y = 44; y < 62; y++) for (let x = 10; x < 28; x++) f[y * W + x] = dist;
  for (let y = 0; y < W; y++) for (let x = 36; x < W; x++) f[y * W + x] = gain * f[y * W + x] + bias;
  return { f: f.map((v) => v + sn * r.normal()), tpl, truth: [[16, 18], [54, 50]] };
}
export function match(f, tpl, m) {
  const out = new Float64Array(W * W).fill(NaN), tm = tpl.reduce((a, b) => a + b, 0) / tpl.length, tz = tpl.map((v) => v - tm), tn = Math.sqrt(tz.reduce((a, v) => a + v * v, 0));
  for (let y = h; y < W - h; y++) for (let x = h; x < W - h; x++) {
    let s = 0, ss = 0, fm = 0;
    if (m === 2) { for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) fm += f[(y - h + j) * W + x - h + i]; fm /= T * T; }
    for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) { const v = f[(y - h + j) * W + x - h + i], k = j * T + i; if (m === 0) s += tpl[k] * v; else if (m === 1) s -= (v - tpl[k]) ** 2; else { s += tz[k] * (v - fm); ss += (v - fm) ** 2; } }
    out[y * W + x] = m === 2 ? s / (tn * Math.sqrt(ss) + 1e-12) : s;
  }
  return out;
}
const METHODS = ["ham korelasyon Σ w·f", "−SSD (kare farkların toplamı)", "korelasyon katsayısı γ (NCC)"];
export default {
  params: [
    { id: "m", label: "Ölçüt", options: METHODS, value: 2 },
    { id: "gain", label: "Sağ yarı kazancı", min: 0.3, max: 2, step: 0.05, value: 0.6 },
    { id: "bias", label: "Sağ yarı ofseti", min: -0.3, max: 0.3, step: 0.01, value: 0.1 },
    { id: "dist", label: "Çeldirici kare parlaklığı", min: 0.2, max: 1, step: 0.05, value: 0.9 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.15, step: 0.005, value: 0.03 },
  ],
  seed: true,
  run(q, r) {
    const { f, tpl, truth } = scene(r, q.noise, q.gain, q.bias, q.dist), M = match(f, tpl, q.m);
    const vals = []; for (let y = h; y < W - h; y++) for (let x = h; x < W - h; x++) vals.push([M[y * W + x], x, y]);
    vals.sort((a, b) => b[0] - a[0]);
    const best = vals[0], second = vals.find((v) => Math.hypot(v[1] - best[1], v[2] - best[2]) > 6);
    const hit = truth.findIndex(([x, y]) => Math.hypot(best[1] - x, best[2] - y) <= 1);
    const at = (x, y) => M[y * W + x], mn = vals[vals.length - 1][0], mx = best[0];
    const P = [`En iyi eşleşme (${best[1]}, ${best[2]}): ${hit === 0 ? "sol üstteki özgün kopya" : hit === 1 ? "sağdaki (aydınlatması değişmiş) kopya" : "<b>yanlış yer</b>"}. İki kopyadaki değerler ${fmt(at(...truth[0]), 4)} ve ${fmt(at(...truth[1]), 4)}; en iyi ikinci tepe ${fmt(second[0], 4)}.`];
    P.push(q.m === 0 ? "Ham korelasyon genliğe duyarlıdır: parlak, düz bir bölge şablonla 'iyi' çarpım verir; çeldirici kare en yüksek değeri alabilir." : q.m === 1 ? "SSD tam eşleşmede 0'dır ama ofset ve kazanca duyarlıdır: aydınlatması değişmiş kopya cezalandırılır." : "γ hem şablondan hem pencereden ortalamayı çıkarıp normalize eder: kazanç ve ofset değişimine değişmezdir, düz bir bölgede pay sıfıra yakındır. Değeri [−1, 1] aralığındadır.");
    return {
      plots: [
        { title: "Sahne ve bulunan konum", img: { w: W, h: W, data: Array.from(f, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 300, h: 290 }, series: [{ name: "en iyi", x: [best[1] + 0.5], y: [W - best[2] - 0.5], kind: "scatter", c: 1, r: 6 }, { name: "gerçek kopyalar", x: truth.map((t) => t[0] + 0.5), y: truth.map((t) => W - t[1] - 0.5), kind: "scatter", c: 2, r: 3 }] },
        { title: `${METHODS[q.m]} haritası`, img: { w: W, h: W, data: Array.from(M, (v) => (Number.isNaN(v) ? 0 : (v - mn) / (mx - mn || 1))) }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "En iyi konum", value: `(${best[1]}, ${best[2]})` }, { label: "Özgün / değişmiş kopyada değer", value: `${fmt(at(...truth[0]), 4)} / ${fmt(at(...truth[1]), 4)}` }, { label: "İkinci tepe / birinci", value: second[0] / best[0] }],
      note: P.join(" "),
    };
  },
};
