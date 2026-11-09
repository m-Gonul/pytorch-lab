// M4.1 — Konvolüsyon katmanının maliyeti: çıktı boyu, parametre, çarpma–toplama (MAC) ve aktivasyon belleği; aynı girdi/çıktı
// boyutlu tam bağlı katmanla karşılaştırma (ders slaytındaki 200×200×3 → 256 harita, 5×5 örneği).
import { fmt } from "./plot.js";

export function cost(q) {
  const Ho = Math.floor((q.H + 2 * q.P - q.K) / q.S) + 1, Wo = Math.floor((q.W + 2 * q.P - q.K) / q.S) + 1;
  const params = (q.K * q.K * q.C + 1) * q.F, outN = Ho * Wo * q.F, macs = outN * q.K * q.K * q.C, inN = q.H * q.W * q.C;
  const fcParams = inN * outN + outN, fcMacs = inN * outN, bytes = q.bits / 8;
  return { Ho, Wo, params, macs, outN, inN, fcParams, fcMacs, actBytes: outN * bytes, batchBytes: outN * bytes * q.B };
}
const human = (b) => (b >= 2 ** 30 ? `${fmt(b / 2 ** 30, 3)} GiB` : b >= 2 ** 20 ? `${fmt(b / 2 ** 20, 3)} MiB` : `${fmt(b / 2 ** 10, 3)} KiB`);
const big = (n) => (n >= 1e12 ? `${fmt(n / 1e12, 4)} trilyon` : n >= 1e9 ? `${fmt(n / 1e9, 4)} milyar` : n >= 1e6 ? `${fmt(n / 1e6, 4)} milyon` : n.toLocaleString("tr-TR"));

export default {
  params: [
    { id: "H", label: "Girdi yüksekliği H", min: 8, max: 512, value: 200, int: true, log: true },
    { id: "W", label: "Girdi genişliği W", min: 8, max: 512, value: 200, int: true, log: true },
    { id: "C", label: "Girdi kanalı C", min: 1, max: 512, value: 3, int: true, log: true },
    { id: "F", label: "Filtre (çıktı haritası) sayısı F", min: 1, max: 1024, value: 256, int: true, log: true },
    { id: "K", label: "Çekirdek boyu K (K × K)", min: 1, max: 11, value: 5, int: true },
    { id: "S", label: "Adım S", min: 1, max: 4, value: 1, int: true },
    { id: "P", label: "Dolgu P", min: 0, max: 5, value: 2, int: true },
    { id: "B", label: "Batch boyu", min: 1, max: 256, value: 32, int: true, log: true },
    { id: "bits", label: "Sayı biçimi", options: ["16 bit (fp16/bf16)", "32 bit (fp32)"], value: 1 },
  ],
  validate(q) { return q.H + 2 * q.P < q.K || q.W + 2 * q.P < q.K ? [{ level: "error", msg: "Çekirdek, dolgulu girdiden büyük: çıktı boşa düşer (H + 2P ≥ K olmalı)." }] : []; },
  run(q0) {
    const q = { ...q0, bits: q0.bits === 0 ? 16 : 32 }, c = cost(q);
    return {
      plots: [{ title: "Konvolüsyon vs aynı girdi/çıktı boyutlu tam bağlı katman (log)", xlabel: "", ylabel: "sayı", ylog: true, xlim: [-0.6, 3.6], ylim: [Math.max(1, Math.min(c.params, c.macs) / 10), Math.max(c.fcParams, c.fcMacs) * 10],
        xticks: [[0, "parametre (conv)"], [1, "parametre (FC)"], [2, "MAC (conv)"], [3, "MAC (FC)"]],
        series: [{ name: "konvolüsyon", x: [0, 2], y: [c.params, c.macs], kind: "bar", c: 0 }, { name: "tam bağlı", x: [1, 3], y: [c.fcParams, c.fcMacs], kind: "bar", c: 2 }] }],
      stats: [
        { label: "Çıktı (F × H_out × W_out)", value: `${q.F} × ${c.Ho} × ${c.Wo}` },
        { label: "Parametre (K·K·C + 1)·F", value: big(c.params) },
        { label: "Çarpma–toplama (MAC), tek örnek", value: big(c.macs) },
        { label: "Aynı boyutlu FC katmanının parametresi", value: big(c.fcParams) },
        { label: "Çıktı aktivasyonu, tek örnek", value: human(c.actBytes) },
        { label: `Çıktı aktivasyonu, batch ${q.B}`, value: human(c.batchBytes) },
      ],
      note: `Konvolüsyonun parametresi yalnız çekirdeğe bağlıdır (K·K·C + 1)·F — görüntü boyutundan <b>bağımsız</b>; ağırlık paylaşımı sayesinde. Ama hesap ve bellek her çıktı konumu için ödenir: ${c.Ho}·${c.Wo}·${q.F} çıktının her biri ${q.K}·${q.K}·${q.C} = ${q.K * q.K * q.C} çarpım ister. Eğitimde ileri geçişin tüm aktivasyonları geri geçiş için saklanır; bu katmanın çıktısı tek başına ${human(c.batchBytes)} tutar. Bellek yetmezse: batch'i küçült, adımla boyutu düşür, katman çıkar, 16 bit kullan, modeli cihazlara dağıt.`,
    };
  },
};
