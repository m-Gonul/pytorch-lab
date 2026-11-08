// M3.6 — Batch normalizasyonunun batch'e bağlı gürültüsü: aynı örnek, farklı batch'ler
import { fmt, linspace, range } from "./plot.js";
export default {
  params: [
    { id: "B", label: "Batch boyutu B", min: 2, max: 256, log: true, value: 16, int: true, hardMin: 2, why: "B = 1'de varyans 0: BN tanımsız (her şey 0'a gider)." },
    { id: "x", label: "İzlenen örneğin değeri x", min: -3, max: 3, step: 0.1, value: 1.5 },
    { id: "mix", label: "Batch'in ikinci kaynaktan gelen oranı", min: 0, max: 1, step: 0.05, value: 0, help: "Veri iki farklı dağılımdan (μ = 0 ve μ = 2) geliyorsa batch bileşimi istatistikleri kaydırır." },
  ],
  seed: true,
  run(p, rnd) {
    const R = 1500, ys = [];
    for (let r = 0; r < R; r++) {
      const b = [p.x];
      for (let i = 1; i < p.B; i++) b.push(rnd() < p.mix ? 2 + rnd.normal() : rnd.normal());
      const m = b.reduce((a, v) => a + v, 0) / p.B, v = b.reduce((a, q) => a + (q - m) ** 2, 0) / p.B;
      ys.push((p.x - m) / Math.sqrt(v + 1e-5));
    }
    const mean = ys.reduce((a, v) => a + v, 0) / R, sd = Math.sqrt(ys.reduce((a, v) => a + (v - mean) ** 2, 0) / R);
    const edges = linspace(-5, 5, 41), h = new Array(40).fill(0);
    for (const y of ys) { const k = Math.floor((y + 5) / 0.25); if (k >= 0 && k < 40) h[k] += 1 / R; }
    const pm = p.mix * 2, pv = 1 + p.mix * (1 - p.mix) * 4, pop = (p.x - pm) / Math.sqrt(pv);
    const Bs = [2, 4, 8, 16, 32, 64, 128, 256];
    return {
      plots: [{ title: `BN(x) dağılımı, ${R} farklı batch`, xlabel: "BN çıktısı", ylabel: "oran", yzero: true, series: [{ name: "eğitimde (batch istatistiği)", x: edges.slice(0, 40).map((e) => e + 0.125), y: h, kind: "bar" }], marks: [{ x: pop, label: "çıkarımda (koşu istatistiği)" }] }],
      stats: [{ label: "Eğitimdeki BN(x): ortalama", value: mean }, { label: "Eğitimdeki BN(x): std (batch gürültüsü)", value: sd }, { label: "Çıkarımdaki BN(x) (popülasyon istatistiği)", value: pop }, { label: "Gürültü kabaca ∝ 1/√B; B = 4 ile 64 oranı", value: Math.sqrt(64 / 4) }],
      note: p.B <= 4 ? "Küçük batch'te BN çıktısı aynı örnek için batch'ten batch'e çok değişir: eğitim–çıkarım farkı büyür, BN kötüleşir. GroupNorm/LayerNorm batch'ten bağımsızdır." : `Eğitimde aynı örneğin normalize değeri ≈ ${fmt(sd)} std'lik gürültü taşır (düzenlileştirici etki); çıkarımda sabit popülasyon istatistikleri kullanılır.${p.mix > 0 ? " Karışık kaynaklı batch'lerde ortalama da kayar." : ""}`,
    };
  },
};
