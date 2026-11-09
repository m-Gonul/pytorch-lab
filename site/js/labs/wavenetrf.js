// M4.11 — Nedensel (genişletilmiş) 1-B konvolüsyon yığınının alıcı alanı: son çıktının hangi girdi adımlarına bağlı olduğu,
// bağımlılık grafiği izlenerek sayılır ve 1 + Σ (k − 1)·d formülüyle karşılaştırılır. Genişletme şemaları: yok, ikiye katlanan
// (WaveNet), sabit. Aynı alıcı alanı tek bir konvolüsyonla elde etmenin parametre bedeli.
import { fmt, range } from "./plot.js";

export function dilations(scheme, L, B, c) { const one = range(L).map((i) => (scheme === 0 ? 1 : scheme === 1 ? 2 ** i : c)); return range(B).flatMap(() => one); }
// son çıktının bağlı olduğu girdi gecikmeleri (0 = şimdiki adım)
export function support(k, ds) { let S = new Set([0]); for (let l = ds.length - 1; l >= 0; l--) { const N = new Set(); for (const s of S) for (let j = 0; j < k; j++) N.add(s + j * ds[l]); S = N; } return S; }

export default {
  params: [
    { id: "k", label: "Çekirdek boyu k", min: 2, max: 5, value: 2, int: true },
    { id: "L", label: "Blok başına katman sayısı", min: 1, max: 10, value: 4, int: true },
    { id: "B", label: "Blok sayısı (aynı genişletme dizisi tekrarlanır)", min: 1, max: 3, value: 2, int: true },
    { id: "scheme", label: "Genişletme şeması", options: ["yok (hepsi 1)", "ikiye katlanan 1, 2, 4, … (WaveNet)", "sabit c"], value: 1 },
    { id: "c", label: "Sabit genişletme c", min: 1, max: 8, value: 2, int: true, on: { scheme: [2] } },
    { id: "C", label: "Kanal sayısı (parametre hesabı için)", min: 1, max: 256, value: 32, int: true, log: true },
  ],
  run(q) {
    const ds = dilations(q.scheme, q.L, q.B, q.c), S = support(q.k, ds), rf = 1 + ds.reduce((s, d) => s + (q.k - 1) * d, 0), maxLag = Math.max(...S);
    const holes = maxLag + 1 - S.size, lags = range(maxLag + 1), conn = lags.map((l) => (S.has(l) ? 1 : 0));
    const pStack = ds.length * (q.k * q.C * q.C + q.C), pSingle = rf * q.C * q.C + q.C;
    return {
      plots: [
        { title: "Son çıktının bağlı olduğu girdi adımları (0 = şimdi, sağda geçmiş)", xlabel: "geçmişe doğru gecikme", ylabel: "bağlı mı", ylim: [0, 1.2], series: [{ name: "bağlı", x: lags, y: conn, kind: "bar", c: 0, bw: maxLag > 200 ? 1 : 0.8 }] },
        { title: "Katmanlara göre alıcı alan (log)", xlabel: "katman", ylabel: "alıcı alan", ylog: true, series: [{ name: "bu yığın", x: range(ds.length).map((i) => i + 1), y: ds.map((_, i) => 1 + ds.slice(0, i + 1).reduce((s, d) => s + (q.k - 1) * d, 0)), c: 0, dots: true }, { name: "genişletmesiz", x: range(ds.length).map((i) => i + 1), y: ds.map((_, i) => 1 + (i + 1) * (q.k - 1)), c: 4, dash: true }] },
      ],
      stats: [
        { label: "Genişletmeler", value: ds.join(", ") },
        { label: "Alıcı alan: formül 1 + Σ(k−1)d / sayılan en uzak gecikme + 1", value: `${rf} / ${maxLag + 1}` },
        { label: "Alıcı alan içinde görülmeyen adım ('delik')", value: holes },
        { label: `Parametre (${q.C} kanal): yığın / aynı alıcı alanlı tek konv.`, value: `${pStack.toLocaleString("tr-TR")} / ${pSingle.toLocaleString("tr-TR")}` },
      ],
      note: `Nedensel konvolüsyon yalnız şimdiki ve geçmiş adımlara bakar ("causal" dolgu: girdinin soluna (k − 1)·d sıfır). Genişletme d, çekirdek elemanları arasına d − 1 adım boşluk koyar. ${q.scheme === 1 ? `İkiye katlanan genişletmelerle alıcı alan katman sayısıyla <b>üstel</b> büyür ve hiç delik kalmaz: ${q.L} katmanlık bir blok ${q.k === 2 ? 2 ** q.L : "k^L"} adımlık bir pencereyi görür. WaveNet 1, 2, …, 512'lik on katmanı üç kez yığar (blok başına 1024 adım).` : q.scheme === 0 ? "Genişletme olmadan alıcı alan katman sayısıyla yalnız <b>doğrusal</b> büyür: uzun ses dizileri için yüzlerce katman gerekirdi." : holes > 0 ? `<b>Izgaralanma:</b> sabit genişletme c = ${q.c} ve k = ${q.k} ile bazı geçmiş adımlar hiç görülmüyor (${holes} delik).` : "Sabit genişletmede alıcı alan doğrusal büyür."} Tek bir konvolüsyonla aynı alıcı alan ${fmt(pSingle / pStack, 3)} kat parametre isterdi.`,
    };
  },
};
