// M10.1 — Amdahl ve Gustafson–Barsis: seri kesir, işlemci sayısı, verim
import { fmt } from "./plot.js";
export default {
  params: [
    { id: "f", label: "Seri kesir f", min: 0, max: 0.5, step: 0.005, value: 0.05 },
    { id: "N", label: "İşlemci sayısı N (gösterilen)", min: 1, max: 1024, log: true, value: 64, int: true },
    { id: "ov", label: "İşlemci başına ek yük (iletişim) o", min: 0, max: 0.01, step: 0.0005, value: 0, help: "T(N) = f + (1−f)/N + o·log₂N  (basit indirgeme modeli)." },
  ],
  run(p) {
    const Ns = []; for (let k = 0; k <= 10; k += 0.25) Ns.push(2 ** k);
    const am = (N) => 1 / (p.f + (1 - p.f) / N + p.ov * Math.log2(N));
    const gu = (N) => p.f + (1 - p.f) * N;
    const best = Ns.reduce((b, N) => (am(N) > am(b) ? N : b), 1);
    return {
      plots: [
        { title: "Hızlanma S(N)", xlabel: "log₂ N", ylabel: "S", ylog: true, series: [{ name: "ideal N", x: Ns.map(Math.log2), y: Ns, c: 4, dash: true }, { name: "Amdahl (sabit problem)", x: Ns.map(Math.log2), y: Ns.map(am) }, { name: "Gustafson (problem N ile büyür)", x: Ns.map(Math.log2), y: Ns.map(gu), c: 2 }], marks: [{ x: Math.log2(p.N), label: "N" }, ...(p.f > 0 ? [{ y: 1 / p.f, label: "1/f" }] : [])] },
        { title: "Verim E = S/N", xlabel: "log₂ N", ylabel: "E", ylim: [0, 1.02], series: [{ name: "Amdahl", x: Ns.map(Math.log2), y: Ns.map((N) => am(N) / N) }, { name: "Gustafson", x: Ns.map(Math.log2), y: Ns.map((N) => gu(N) / N), c: 2 }] },
      ],
      stats: [{ label: "Amdahl S(N)", value: am(p.N) }, { label: "Üst sınır 1/f", value: p.f > 0 ? 1 / p.f : "∞" }, { label: "Gustafson ölçekli S(N)", value: gu(p.N) }, { label: "Verim (Amdahl)", value: am(p.N) / p.N }, ...(p.ov > 0 ? [{ label: "Ek yükle en iyi N", value: best }] : [])],
      note: `Amdahl: problem sabitken seri kısım hızlanmayı 1/f = ${p.f > 0 ? fmt(1 / p.f) : "∞"} ile tavanlar. Gustafson: paralel kısım N ile büyütülürse (aynı sürede daha büyük problem) ölçekli hızlanma doğrusal kalır — iki yasa çelişmez, farklı soruları cevaplar.${p.ov > 0 ? " İletişim ek yükü varsa hızlanma bir N'den sonra düşer: daha çok işlemci zarar verir." : ""}`,
    };
  },
};
