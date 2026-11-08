// M1.5 / M4.16 — Softmax sıcaklığı: olasılık dağılımı, entropi, örnekleme çeşitliliği
import { fmt } from "./plot.js";
export default {
  params: [
    { id: "T", label: "Sıcaklık T", min: 0.05, max: 10, log: true, value: 1, hardMin: 1e-4, why: "T → 0 argmax'a, T = 0 tanımsız (sıfıra bölme)." },
    { id: "spread", label: "Logit aralığı (en büyük − en küçük)", min: 0, max: 12, step: 0.1, value: 4 },
    { id: "k", label: "Sınıf sayısı", min: 2, max: 20, step: 1, value: 8, int: true },
    { id: "shift", label: "Tüm logitlere sabit ekle", min: -50, max: 50, step: 1, value: 0, help: "Softmax'ın kayma değişmezliğini sına: sonuç değişmemeli." },
  ],
  run(p) {
    const z = Array.from({ length: p.k }, (_, i) => p.spread * (1 - i / (p.k - 1)) ** 1.5 + p.shift);
    const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / p.T)), S = e.reduce((a, b) => a + b, 0), P = e.map((v) => v / S);
    const H = -P.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0);
    return {
      plots: [{ title: "Olasılıklar p_i = softmax(z/T)", xlabel: "sınıf (logit büyükten küçüğe)", ylabel: "p", yzero: true, ylim: [0, 1], series: [{ name: "p", x: P.map((_, i) => i + 1), y: P, kind: "bar" }, { name: "düzgün 1/K", x: [0.5, p.k + 0.5], y: [1 / p.k, 1 / p.k], c: 2, dash: true }] }],
      stats: [{ label: "En büyük olasılık", value: P[0] }, { label: "Entropi H (nat)", value: H }, { label: "Azami entropi ln K", value: Math.log(p.k) }, { label: "Etkin sınıf sayısı e^H", value: Math.exp(H) }],
      note: `T küçüldükçe dağılım en büyük logite yığılır (T → 0: argmax), büyüdükçe düzgüne yaklaşır (T → ∞: entropi ln K). Sabit kaydırma (${fmt(p.shift)}) hiçbir şeyi değiştirmez: softmax(z + c) = softmax(z); kod bu yüzden en büyük logiti çıkarıp taşmayı önler. Bilgi damıtmada T > 1 "karanlık bilgiyi" (yanlış sınıflar arası benzerlik) görünür kılar.`,
    };
  },
};
