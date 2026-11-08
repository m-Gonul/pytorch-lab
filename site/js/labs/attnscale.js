// M4.6 — Ölçekli nokta çarpımı: √d_k bölmesi olmadan softmax doyar
import { fmt, range } from "./plot.js";
function ent(P) { return -P.reduce((a, q) => a + (q > 1e-300 ? q * Math.log(q) : 0), 0); }
function soft(z) { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }
export default {
  params: [
    { id: "d", label: "Anahtar boyutu d_k", min: 1, max: 1024, log: true, value: 64, int: true },
    { id: "scale", label: "Ölçekleme", options: ["1/√d_k (Transformer)", "yok"], value: 0 },
    { id: "n", label: "Anahtar sayısı (dizi uzunluğu)", min: 2, max: 128, step: 1, value: 32, int: true },
  ],
  seed: true,
  run(p, rnd) {
    const gen = (d) => {
      const q = Array.from({ length: d }, () => rnd.normal());
      const s = range(p.n).map(() => { let a = 0; for (let i = 0; i < d; i++) a += q[i] * rnd.normal(); return p.scale === 0 ? a / Math.sqrt(d) : a; });
      return soft(s);
    };
    const P = gen(p.d).sort((a, b) => b - a);
    const ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    const H = (sc) => ds.map((d) => { let h = 0; for (let r = 0; r < 12; r++) { const q = Array.from({ length: d }, () => rnd.normal()); const s = range(p.n).map(() => { let a = 0; for (let i = 0; i < d; i++) a += q[i] * rnd.normal(); return sc ? a / Math.sqrt(d) : a; }); h += ent(soft(s)); } return h / 12; });
    const Hs = H(true), Hn = H(false);
    const top = P[0];
    return {
      plots: [
        { title: "Bir sorgunun attention ağırlıkları (büyükten küçüğe)", xlabel: "anahtar sırası", ylabel: "ağırlık", yzero: true, series: [{ name: "α", x: range(p.n).map((i) => i + 1), y: P, kind: "bar" }] },
        { title: "Ortalama entropi – d_k (azami ln n)", xlabel: "log₂ d_k", ylabel: "H (nat)", ylim: [0, Math.log(p.n) * 1.05], series: [{ name: "1/√d_k ile", x: ds.map(Math.log2), y: Hs, dots: true }, { name: "ölçeksiz", x: ds.map(Math.log2), y: Hn, dots: true, c: 3 }], marks: [{ x: Math.log2(p.d), label: "seçili" }, { y: Math.log(p.n), label: "ln n" }] },
      ],
      stats: [{ label: "Skor std (q·k ~ N(0, d))", value: p.scale === 0 ? 1 : Math.sqrt(p.d) }, { label: "En büyük ağırlık", value: top }, { label: "Entropi / ln n", value: ent(P) / Math.log(p.n) }],
      note: p.scale === 1 && p.d >= 16 ? `Ölçeksiz skorların std'si √d_k = ${fmt(Math.sqrt(p.d))}: softmax neredeyse one-hot (en büyük ağırlık ${fmt(top)}). Doymuş softmax'ın Jacobian'ı diag(α) − ααᵀ ≈ 0 olur → gradyan kaybolur.` : "q, k bileşenleri bağımsız N(0,1) iken q·k'nın varyansı d_k'dır; √d_k'ya bölmek varyansı 1'e çeker ve entropi d_k'dan bağımsız kalır.",
    };
  },
};
