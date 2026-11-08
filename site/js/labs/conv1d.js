// M4.1 — 1B konvolüsyon (çapraz korelasyon): çekirdek, adım, dolgu, genişleme ve çıktı boyutu
import { fmt, range } from "./plot.js";
const KER = ["kutu (ortalama)", "Gauss", "türev [−1, 0, 1]", "Laplace [1, −2, 1]", "kenar algılayıcı [−1,…,−1, 1,…,1]"];
export default {
  params: [
    { id: "ker", label: "Çekirdek", options: KER, value: 1 },
    { id: "k", label: "Çekirdek boyu k", min: 1, max: 21, step: 2, value: 7, int: true, help: "Türev ve Laplace sabit boylu (3).", on: { ker: [0, 1, 4] } },
    { id: "s", label: "Adım (stride) s", min: 1, max: 8, step: 1, value: 1, int: true },
    { id: "pad", label: "Dolgu", options: ["yok (valid)", "aynı (same, sıfır)", "aynı (yansıtma)"], value: 1 },
    { id: "dil", label: "Genişleme (dilation) d", min: 1, max: 6, step: 1, value: 1, int: true },
    { id: "noise", label: "Gürültü σ", min: 0, max: 0.5, step: 0.01, value: 0.1 },
  ],
  seed: true,
  validate: (p) => { const k = p.ker === 2 || p.ker === 3 ? 3 : p.k, eff = (k - 1) * p.dil + 1; return eff > 120 ? [{ level: "error", msg: `Etkin çekirdek (k−1)d+1 = ${eff} girdiden (120) uzun: valid çıktı boş.` }] : []; },
  run(p, rnd) {
    const n = 120, x = range(n).map((i) => (i > 30 && i < 70 ? 1 : 0) + 0.4 * Math.sin(i / 4) * (i > 80) + p.noise * rnd.normal());
    let w;
    const k = p.ker === 2 || p.ker === 3 ? 3 : p.k;
    if (p.ker === 0) w = new Array(k).fill(1 / k);
    else if (p.ker === 1) { const s = k / 6 || 1; w = range(k).map((i) => Math.exp(-((i - (k - 1) / 2) ** 2) / (2 * s * s))); const z = w.reduce((a, b) => a + b, 0); w = w.map((v) => v / z); }
    else if (p.ker === 2) w = [-1, 0, 1];
    else if (p.ker === 3) w = [1, -2, 1];
    else w = range(k).map((i) => (i < (k - 1) / 2 ? -1 : i > (k - 1) / 2 ? 1 : 0).valueOf() / Math.max(1, Math.floor(k / 2)));
    const eff = (k - 1) * p.dil + 1, P = p.pad === 0 ? 0 : Math.floor((eff - 1) / 2);
    const at = (i) => (i >= 0 && i < n ? x[i] : p.pad === 2 ? x[i < 0 ? -i : 2 * n - 2 - i] ?? 0 : 0);
    const out = [], pos = [];
    for (let o = 0; ; o++) { const start = o * p.s - P; if (start + eff - 1 > n - 1 + P) break; let s = 0; for (let j = 0; j < k; j++) s += w[j] * at(start + j * p.dil); out.push(s); pos.push(start + (eff - 1) / 2); }
    const formula = Math.floor((n + 2 * P - eff) / p.s) + 1;
    return {
      plots: [
        { title: "Girdi ve çıktı (çıktı, alıcı alanının merkezine çizildi)", xlabel: "konum", ylabel: "değer", series: [{ name: "girdi", x: range(n), y: x, c: 4 }, { name: "çıktı", x: pos, y: out, dots: p.s > 1, c: 0 }] },
        { title: "Çekirdek w", xlabel: "j", ylabel: "w_j", series: [{ name: "w", x: range(k).map((j) => j * p.dil), y: w, kind: "bar" }] },
      ],
      stats: [{ label: "Çıktı uzunluğu (sayılan)", value: out.length }, { label: "⌊(n + 2P − ((k−1)d+1))/s⌋ + 1", value: formula }, { label: "Etkin çekirdek (k−1)d + 1", value: eff }, { label: "Dolgu P", value: P }, { label: "Parametre sayısı", value: k }],
      note: `Çıktı boyu formülü n = 120, P = ${P}, etkin çekirdek ${eff}, s = ${p.s} ile ${formula}. Genişleme parametre sayısını artırmadan alıcı alanını büyütür (WaveNet). PyTorch Conv1d aslında çapraz korelasyondur (çekirdek ters çevrilmez); öğrenilen ağırlıklarda fark etmez. Türev çekirdeği gürültüyü büyütür — önce düzleştirip sonra türev almak (Gauss türevi) bu yüzden.`,
    };
  },
};
