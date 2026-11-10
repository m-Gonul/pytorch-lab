// M9.18 — ViBe'nin iç işleyişi: hayalet sönümlemesi ve hafızasız güncellemenin örnek ömrü. 48 × 48, 300 karelik yapay video: dokulu
// sabit arka plan ve gürültü; ilk 3 karede ortadaki 16 × 16 kutuda bir nesne vardır, sonra ayrılır. Model ilk kareden başlatıldığı için
// nesnenin değerleri kutunun modeline girer: nesne gittikten sonra orada görülen gerçek arka plan "ön plan" sanılır (hayalet). Uzamsal
// yayılım, kutunun kenarındaki arka plan piksellerinin değerlerini içeri taşıyarak hayaleti kenarlardan eritir. Ayrıca kutu dışındaki
// sabit bölgede başlangıç örneklerinin modelde kalma oranı ölçülür ve ((N − 1)/N)^{u·t/φ} ile karşılaştırılır (u = 1 yayılımsız, u = 2 yayılımla:
// her piksel kendi güncellemesine ek olarak ortalama bir komşu güncellemesi alır).
import { fmt, range } from "./plot.js";

const W = 48, TT = 300;
export default {
  params: [
    { id: "N", label: "Örnek sayısı N", min: 4, max: 40, value: 20, int: true },
    { id: "min", label: "Eşleşme sayısı #min", min: 1, max: 5, value: 2, int: true },
    { id: "R", label: "Yarıçap R", min: 0.02, max: 0.3, step: 0.01, value: 0.08 },
    { id: "phi", label: "Alt örnekleme φ", min: 1, max: 64, value: 16, int: true, log: true },
    { id: "diff", label: "Uzamsal yayılım", options: ["kapalı", "açık"], value: 1 },
    { id: "show", label: "Gösterilen kare", min: 3, max: 299, value: 60, int: true },
  ],
  seed: true,
  run(q, r) {
    const B = range(40).map(() => [48 * r(), 48 * r(), 1.5 + 3 * r(), (r() < 0.5 ? -1 : 1) * (0.06 + 0.12 * r())]);
    const bg = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let v = 0.45; for (const [cx, cy, s, a] of B) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)); bg[y * W + x] = v; }
    const inBox = (x, y) => x >= 16 && x < 32 && y >= 16 && y < 32;
    const frame = (t) => Float64Array.from(bg, (v, i) => (t < 3 && inBox(i % W, (i / W) | 0) ? 0.9 : v) + 0.02 * r.normal());
    const N = q.N, S = new Float64Array(W * W * N), tag = new Uint8Array(W * W * N).fill(1);
    const nb = (x, y) => { let nx, ny; do { nx = x + Math.floor(r() * 3) - 1; ny = y + Math.floor(r() * 3) - 1; } while ((nx === x && ny === y) || nx < 0 || ny < 0 || nx >= W || ny >= W); return [nx, ny]; };
    const F0 = frame(0);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) for (let k = 0; k < N; k++) { const [nx, ny] = nb(x, y); S[(y * W + x) * N + k] = F0[ny * W + nx]; }
    const ghost = [], surv = [], masks = []; let cntOut = 0; for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if (x < 8 || x >= 40 || y < 8 || y >= 40) cntOut++;
    for (let t = 1; t < TT; t++) {
      const I = frame(t), M = new Uint8Array(W * W);
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, v = I[i]; let c = 0;
        for (let k = 0; k < N && c < q.min; k++) if (Math.abs(v - S[i * N + k]) < q.R) c++;
        if (c >= q.min) {
          if (Math.floor(r() * q.phi) === 0) { const k = Math.floor(r() * N); S[i * N + k] = v; tag[i * N + k] = 0; }
          if (q.diff && Math.floor(r() * q.phi) === 0) { const [nx, ny] = nb(x, y), k = Math.floor(r() * N); S[(ny * W + nx) * N + k] = v; tag[(ny * W + nx) * N + k] = 0; }
        } else M[i] = 1;
      }
      let g = 0; for (let y = 16; y < 32; y++) for (let x = 16; x < 32; x++) g += M[y * W + x];
      ghost.push(t < 3 ? NaN : g / 256);
      let s = 0; for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if (x < 8 || x >= 40 || y < 8 || y >= 40) for (let k = 0; k < N; k++) s += tag[(y * W + x) * N + k];
      surv.push(s / (cntOut * N)); masks.push(M);
    }
    const ts = range(TT - 1).map((k) => k + 1), u = q.diff ? 2 : 1, th = ts.map((t) => ((N - 1) / N) ** ((u * t) / q.phi));
    const half = ghost.findIndex((g, k) => k >= 2 && g <= 0.5), gone = ghost.findIndex((g, k) => k >= 2 && g <= 0.05);
    const tHalfTh = (q.phi * Math.log(0.5)) / (u * Math.log((N - 1) / N));
    const P = [half < 0 ? "Hayalet (kutudaki yanlış ön plan oranı) 300 karede yarıya bile inmiyor." : `Hayalet (kutudaki yanlış ön plan oranı) ${half + 1}. karede yarıya iniyor${gone >= 0 ? `, ${gone + 1}. karede %5'in altında` : ", 300 karede %5'in altına inmiyor"}.`];
    P.push(`Kutu dışındaki başlangıç örneklerinin modelde kalma oranı kare 299'da ölçülen ${fmt(surv[TT - 2], 3)}, kuram ((N − 1)/N)^(${u}t/φ) = ${fmt(th[TT - 2], 3)}; yarı ömür kuramda ${fmt(tHalfTh, 3)} kare.`);
    if (!q.diff) P.push("Yayılım kapalı: hayaletin içindeki pikseller hiç arka plan sayılmadığı için kendi modellerini güncelleyemez; hayalet kalıcı.");
    else P.push("Yayılım hayaleti kenarlardan içe doğru eritiyor: kenardaki arka plan pikselleri değerlerini komşu modellere taşıyor.");
    return {
      plots: [
        { title: `Ön plan maskesi, kare ${q.show}`, img: { w: W, h: W, data: Array.from(masks[q.show - 1]) }, side: true, size: { w: 240, h: 240 }, series: [] },
        { title: "Hayalet alanı (kutuda ön plan oranı)", xlabel: "kare", ylabel: "oran", side: true, size: { w: 340, h: 240 }, ylim: [0, 1], series: [{ name: "hayalet", x: ts, y: ghost, c: 1 }], marks: [{ x: q.show, label: "gösterilen" }] },
        { title: "Başlangıç örneklerinin modelde kalma oranı", xlabel: "kare", ylabel: "oran", wide: true, size: { w: 620, h: 180 }, ylim: [0, 1], series: [{ name: "ölçülen", x: ts, y: surv, c: 0 }, { name: "kuram", x: ts, y: th, c: 2, dash: true }] },
      ],
      stats: [{ label: "Hayalet yarıya iniyor", value: half >= 0 ? `kare ${half + 1}` : "> 299" }, { label: "Hayalet < %5", value: gone >= 0 ? `kare ${gone + 1}` : "> 299" }, { label: "Örnek kalma oranı (kare 299): ölçülen / kuram", value: `${fmt(surv[TT - 2], 3)} / ${fmt(th[TT - 2], 3)}` }],
      note: P.join(" "),
    };
  },
};
