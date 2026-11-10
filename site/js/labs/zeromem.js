// M7.4 — GPU başına eğitim belleği: DDP ve ZeRO aşamaları (Rajbhandari ve ark. 2020). Parametre başına K bayt: ağırlık w_b, gradyan g_b,
// optimizasyon durumu o_b (fp32 Adam: 4 + 4 + 8 = 16; karışık hassasiyet: 2 + 2 + 12 = 16, 12 = fp32 ana kopya + iki moment).
// DDP: hepsi her GPU'da. ZeRO-1: durum / N. ZeRO-2: durum ve gradyan / N. ZeRO-3 (FSDP): hepsi / N. Aktivasyonlar ayrı (bölünmez).
// İletişim (adım başına, parametre sayısı Ψ cinsinden): DDP ve ZeRO-1/2: 2Ψ (all-reduce = reduce-scatter + all-gather); ZeRO-3: 3Ψ.
import { fmt, range } from "./plot.js";

const MODES = ["DDP", "ZeRO-1 (durum)", "ZeRO-2 (+ gradyan)", "ZeRO-3 / FSDP (+ ağırlık)"];
export function perGPU(P, N, b, mode) { const [w, g, o] = b; return P * ((mode >= 3 ? w / N : w) + (mode >= 2 ? g / N : g) + (mode >= 1 ? o / N : o)); }
export default {
  params: [
    { id: "P", label: "Parametre sayısı (milyar)", min: 0.1, max: 200, value: 1, log: true },
    { id: "N", label: "GPU sayısı N", min: 1, max: 1024, value: 8, int: true, log: true },
    { id: "prec", label: "Hassasiyet", options: ["fp32 Adam (4 + 4 + 8)", "karışık: bf16 ağırlık/gradyan + fp32 kopya ve momentler (2 + 2 + 12)"], value: 0 },
    { id: "mode", label: "Strateji", options: MODES, value: 3 },
    { id: "act", label: "Aktivasyonlar (GB / GPU)", min: 0, max: 80, step: 0.5, value: 4 },
    { id: "cap", label: "GPU belleği (GB)", min: 8, max: 192, value: 80, int: true },
  ],
  run(q) {
    const P = q.P * 1e9, b = q.prec ? [2, 2, 12] : [4, 4, 8], mem = (m, n) => perGPU(P, n, b, m) / 1e9 + q.act, cur = mem(q.mode, q.N);
    const all = MODES.map((_, m) => mem(m, q.N)), Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    const comm = q.mode === 3 ? 3 : 2, fits = cur <= q.cap;
    const P_ = [`${q.P} milyar parametre, ${q.N} GPU, ${MODES[q.mode]}: GPU başına ${fmt(cur - q.act, 4)} GB model durumu + ${q.act} GB aktivasyon = ${fmt(cur, 4)} GB (${fits ? "sığıyor" : "<b>sığmıyor</b>"}, sınır ${q.cap} GB).`];
    P_.push(`Dört strateji: ${MODES.map((m, i) => `${m} ${fmt(all[i], 4)} GB`).join(", ")}. Bölünemeyen kısım ZeRO-1'de ağırlık + gradyan (${b[0] + b[1]} bayt/parametre), ZeRO-2'de ağırlık (${b[0]}).`);
    P_.push(`İletişim: adım başına ≈ ${comm}Ψ eleman${q.mode === 3 ? " (ileri ve geri geçişte ağırlıklar all-gather ile toplanır, gradyanlar reduce-scatter ile dağıtılır)" : " (gradyanların all-reduce'u; ZeRO-1/2'de reduce-scatter + güncellenmiş parçaların all-gather'ı, toplam aynı)"}. ZeRO-3 belleği N'ye böler ama iletişimi 1.5 kat artırır.`);
    return {
      plots: [{ title: "GPU başına bellek ve GPU sayısı", xlabel: "N (log₂)", ylabel: "GB", ylog: true, series: MODES.map((m, i) => ({ name: m, x: Ns.map(Math.log2), y: Ns.map((n) => mem(i, n)), c: i })).concat([{ name: "GPU belleği", x: Ns.map(Math.log2), y: Ns.map(() => q.cap), c: 4, dash: true }]),
        marks: [{ x: Math.log2(q.N), label: "N" }], xticks: Ns.map((n) => [Math.log2(n), String(n)]) }],
      stats: [
        { label: "GPU başına (GB)", value: cur },
        { label: "DDP / Z1 / Z2 / Z3 (GB)", value: all.map((v) => fmt(v, 3)).join(" / ") },
        { label: "İletişim (Ψ cinsinden)", value: `${comm}Ψ` },
      ],
      note: P_.join(" "),
    };
  },
};
