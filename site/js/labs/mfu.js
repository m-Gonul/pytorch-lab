// M8.1 — Model FLOP kullanımı (MFU). GPT benzeri model: L katman, genişlik d, bağlam T, sözlük V. Parametre N ≈ 12Ld² + Vd (+ Td konum).
// Token başına ileri FLOP ≈ 2N + 2·L·T·d (dikkat skorları ve değerlerle çarpım; nedensel maskeyle yarısı da sayılabilir);
// eğitim ≈ 3 × ileri (Kaplan; PaLM'ın MFU tanımı). MFU = (token/s × token başına eğitim FLOP'u) / tepe.
import { fmt, range } from "./plot.js";

export function cost(q) {
  const N = 12 * q.L * q.d * q.d + q.V * q.d + q.T * q.d, attn = 2 * q.L * q.T * q.d, fwd = 2 * N + attn;
  return { N, attn, fwd, train: 3 * fwd };
}
export default {
  params: [
    { id: "L", label: "Katman L", min: 1, max: 96, value: 2, int: true },
    { id: "d", label: "Genişlik d", min: 32, max: 12288, value: 128, int: true, log: true },
    { id: "T", label: "Bağlam T", min: 8, max: 32768, value: 9, int: true, log: true },
    { id: "V", label: "Sözlük V", min: 13, max: 256000, value: 13, int: true, log: true },
    { id: "B", label: "Batch (dizi)", min: 1, max: 4096, value: 128, int: true, log: true },
    { id: "ms", label: "Ölçülen adım süresi (ms)", min: 0.1, max: 100000, value: 15, log: true },
    { id: "peak", label: "Tepe hız (TFLOP/s)", min: 0.1, max: 2000, value: 1, log: true },
  ],
  run(q) {
    const c = cost(q), tok = q.B * q.T, flops = tok * c.train, ach = flops / (q.ms / 1000), mfu = ach / (q.peak * 1e12);
    const Ts = range(13).map((i) => 2 ** (i + 3)), frac = Ts.map((t) => { const cc = cost({ ...q, T: t }); return cc.attn / cc.fwd; });
    const P = [`N ≈ 12Ld² + Vd + Td = ${fmt(c.N, 4)} parametre. Token başına ileri FLOP ${fmt(c.fwd, 4)} (bunun ${fmt(c.attn / c.fwd, 3)}'i dikkat), eğitimde × 3 = ${fmt(c.train, 4)}. Adım başına ${tok} token, ${fmt(flops, 3)} FLOP; ${q.ms} ms'de ${fmt(ach / 1e12, 3)} TFLOP/s, MFU = ${fmt(mfu, 3)}.`];
    if (mfu > 1) P.push("<b>MFU > 1</b>: ya ölçülen süre çok kısa (eşzamansız çekirdekler senkronize edilmeden ölçülmüş olabilir) ya da tepe değer yanlış.");
    else if (mfu < 0.05) P.push("MFU çok düşük: küçük modellerde süre hesap değil Python ve çekirdek başlatma ek yüküyle belirlenir (M6.5). Profiler'da en uzun işlemler matmul değilse model ek yük bağımlıdır; daha büyük batch, torch.compile ya da işlemleri birleştirmek (aşama e) çözümdür.");
    else P.push(`Büyük modellerde iyi eğitim koşuları %40–60 MFU'ya ulaşır; kalan, bellek bağımlı işlemler, iletişim ve ek yüktür.`);
    P.push(`Bağlam uzadıkça dikkatin payı büyür (T = ${q.T} iken ${fmt(c.attn / c.fwd, 3)}); 6N kuralı bu terimi yok sayar.`);
    return {
      plots: [{ title: "İleri FLOP'ta dikkatin payı ve bağlam", xlabel: "T (log₂)", ylabel: "pay", ylim: [0, 1], series: [{ name: "2LTd / (2N + 2LTd)", x: Ts.map(Math.log2), y: frac, c: 0 }], marks: [{ x: Math.log2(q.T), label: "T" }], xticks: Ts.filter((_, i) => i % 2 === 0).map((t) => [Math.log2(t), String(t)]) }],
      stats: [
        { label: "Parametre N", value: c.N },
        { label: "Eğitim FLOP'u / token", value: c.train },
        { label: "Ulaşılan TFLOP/s", value: ach / 1e12 },
        { label: "MFU", value: mfu },
      ],
      note: P.join(" "),
    };
  },
};
