// M9.10 — Ters süzgeç ve Wiener. 64 × 64 sahne frekansta Gauss H(u, v) = exp(−D²/2D_b²) ile (dairesel) bulanıklaştırılır,
// beyaz Gauss gürültüsü eklenir. Geri kazanım: tam ters süzgeç G/H, yarıçapı D_c ile sınırlı ters süzgeç, sabit K'lı Wiener
// H*/(|H|² + K) ve gerçek güç spektrumlarıyla "kâhin" Wiener H*/(|H|² + S_η/S_f). Ölçüt: PSNR (tepe 1).
import { fmt, range } from "./plot.js";
import { fft2 } from "./circconv.js";
import { scene } from "./freqfilt.js";

const M = 64, c = M / 2;
const psnr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m += (a[i] - b[i]) ** 2; m /= a.length; return 10 * Math.log10(1 / m); };
const METHODS = ["tam ters süzgeç G/H", "D_c ile sınırlı ters süzgeç", "Wiener, sabit K", "Wiener, gerçek S_η/S_f (kâhin)"];
export default {
  params: [
    { id: "Db", label: "Bulanıklık D_b (frekansta Gauss genişliği)", min: 4, max: 30, step: 1, value: 10 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.05, step: 0.001, value: 0.01 },
    { id: "m", label: "Yöntem", options: METHODS, value: 2 },
    { id: "Dc", label: "Ters süzgeç yarıçapı D_c", min: 2, max: 32, step: 1, value: 14, on: { m: [1] } },
    { id: "K", label: "Wiener K", min: 1e-5, max: 0.1, value: 0.002, log: true, on: { m: [2] } },
  ],
  seed: true,
  run(q, r) {
    const f = scene(), sg = (a) => Float64Array.from(a, (v, i) => v * (((i % M) + ((i / M) | 0)) % 2 ? -1 : 1));
    const F = fft2(sg(f), new Float64Array(M * M), M), Hh = new Float64Array(M * M);
    for (let v = 0; v < M; v++) for (let u = 0; u < M; u++) Hh[v * M + u] = Math.exp(-((u - c) ** 2 + (v - c) ** 2) / (2 * q.Db * q.Db));
    const B = fft2(F.re.map((a, i) => a * Hh[i]), F.im.map((a, i) => a * Hh[i]), M, true), noiseImg = Float64Array.from(range(M * M), () => q.noise * r.normal());
    const g = sg(B.re).map((v, i) => v + noiseImg[i]), Gf = fft2(sg(g), new Float64Array(M * M), M), Nf = fft2(sg(noiseImg), new Float64Array(M * M), M);
    const gain = new Float64Array(M * M);
    for (let i = 0; i < M * M; i++) {
      const h = Hh[i], D = Math.hypot((i % M) - c, ((i / M) | 0) - c);
      gain[i] = q.m === 0 ? 1 / Math.max(h, 1e-12) : q.m === 1 ? (D <= q.Dc ? 1 / Math.max(h, 1e-12) : 0) : q.m === 2 ? h / (h * h + q.K) : (() => { const sf = F.re[i] ** 2 + F.im[i] ** 2, sn = Nf.re[i] ** 2 + Nf.im[i] ** 2; return h / (h * h + sn / Math.max(sf, 1e-12)); })();
    }
    const R = fft2(Gf.re.map((a, i) => a * gain[i]), Gf.im.map((a, i) => a * gain[i]), M, true), out = sg(R.re), outc = out.map((v) => Math.max(0, Math.min(1, v)));
    const p0 = psnr(g, f), p1 = psnr(outc, f);
    const rad = range(c + 1), prof = (fn) => rad.map((D) => fn(Math.exp(-(D * D) / (2 * q.Db * q.Db)), D));
    const P = [`Bulanık + gürültülü girdi PSNR ${fmt(p0, 3)} dB → ${METHODS[q.m]} ${fmt(p1, 3)} dB.`];
    P.push([
      `Tam ters süzgeç her frekansı 1/H ile büyütür; eksen üzerindeki en yüksek frekansta 1/H ≈ ${fmt(Math.exp((c * c) / (2 * q.Db * q.Db)), 3)} olur. ${q.noise > 0 ? "Bu frekanslardaki gürültü bu kazançla patlar ve görüntü kaybolur." : "Gürültü yokken neredeyse kusursuz geri kazanım (yalnız kayan nokta hatası)."}`,
      `D_c = ${q.Dc} yarıçapının dışı sıfırlanır: gürültünün patladığı yüksek frekanslar atılır ama orada taşınan ayrıntı da gider; D_c'yi büyütmek bir noktaya kadar keskinleştirir, sonra gürültüyü geri getirir.`,
      `Wiener kazancı H/(H² + K): H² ≫ K iken ≈ 1/H (ters süzgeç), H² ≪ K iken ≈ H/K → 0. K, gürültü/sinyal güç oranının sabit bir kestirimidir; çok küçük K ters süzgece, çok büyük K bulanık bir sonuca götürür.`,
      "Kâhin (oracle) Wiener: bu görüntünün ve bu gürültü örneğinin gerçek güç spektrumlarını kullanıp her frekansta ayrı bir NSR uygular. Pratikte S_f bilinmez; sabit K'nın ne kadar geride kaldığını gösteren bir başvuru düzeyidir.",
    ][q.m]);
    return {
      plots: [
        { title: "Bulanık ve gürültülü g", img: { w: M, h: M, data: Array.from(g, (v) => Math.max(0, Math.min(1, v))) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `Geri kazanım: ${METHODS[q.m]}`, img: { w: M, h: M, data: Array.from(outc) }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: "Radyal kazanç (log ölçek)", xlabel: "D (merkezden uzaklık)", ylabel: "kazanç", wide: true, size: { w: 620, h: 170 }, ylog: true, series: [{ name: "H", x: rad, y: prof((h) => h), c: 4, dash: true }, { name: "1/H", x: rad, y: prof((h) => 1 / h), c: 1, dash: true }, ...(q.m === 2 ? [{ name: `Wiener, K = ${fmt(q.K, 2)}`, x: rad, y: prof((h) => Math.max(h / (h * h + q.K), 1e-8)), c: 0 }] : q.m === 1 ? [{ name: "sınırlı ters", x: rad, y: prof((h, D) => (D <= q.Dc ? 1 / h : 1e-8)), c: 0 }] : [])] },
      ],
      stats: [{ label: "PSNR girdi → çıkış", value: `${fmt(p0, 3)} → ${fmt(p1, 3)} dB` }, { label: "En büyük 1/H (köşede değil, eksende D = 32)", value: Math.exp((c * c) / (2 * q.Db * q.Db)) }],
      note: P.join(" "),
    };
  },
};
