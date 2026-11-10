// M9.3 — Kenar dolgusu: k × 1 kutu süzgeci bir satırın sol kenarına uygulanıyor. Satır: parlak (0.8) bir bölge, sağ uca doğru karanlık (0.2)
// ve hafif eğimli. Dolgu modları (OpenCV adlarıyla): CONSTANT (0), REPLICATE (aaa|abc), REFLECT (cba|abc), REFLECT_101 (dcb|abcd; varsayılan),
// WRAP (xyz|abc, dairesel). Ölçüt: ilk r çıkış pikselinin, "kenar olmasaydı" beklenen değerden ortalama sapması.
import { fmt, range } from "./plot.js";

const MODES = ["CONSTANT (0)", "REPLICATE", "REFLECT", "REFLECT_101 (varsayılan)", "WRAP (dairesel)"];
export function padIdx(i, n, m) {
  if (i >= 0 && i < n) return i;
  if (m === 1) return i < 0 ? 0 : n - 1;
  if (m === 2) return i < 0 ? -i - 1 : 2 * n - i - 1;
  if (m === 3) return i < 0 ? -i : 2 * n - i - 2;
  if (m === 4) return ((i % n) + n) % n;
  return -1;
}
export default {
  params: [
    { id: "m", label: "Dolgu modu", options: MODES, value: 0 },
    { id: "k", label: "Süzgeç boyu k (tek)", min: 3, max: 31, step: 2, value: 9, int: true },
    { id: "slope", label: "Kenardaki eğim (piksel başına)", min: -0.02, max: 0.02, step: 0.001, value: 0.005 },
  ],
  run(q) {
    const n = 64, row = range(n).map((i) => (i < 48 ? 0.8 - q.slope * i : 0.2)), r = Math.floor(q.k / 2);
    const val = (i) => { const j = padIdx(i, n, q.m); return j < 0 ? 0 : row[j]; };
    const out = range(n).map((i) => range(q.k).reduce((s, t) => s + val(i - r + t), 0) / q.k);
    const ideal = range(n).map((i) => range(q.k).reduce((s, t) => s + (0.8 - q.slope * (i - r + t)), 0) / q.k);
    const bias = range(r).reduce((s, i) => s + (out[i] - ideal[i]), 0) / Math.max(1, r);
    const allB = MODES.map((_, m) => { const v = (i) => { const j = padIdx(i, n, m); return j < 0 ? 0 : row[j]; }; return range(r).reduce((s, i) => s + (range(q.k).reduce((a, t) => a + v(i - r + t), 0) / q.k - ideal[i]), 0) / Math.max(1, r); });
    const P = [`${MODES[q.m]}, k = ${q.k}: ilk ${r} pikselin "kenar olmasaydı" beklenen değerden ortalama sapması ${fmt(bias, 3)}. Bütün modlar: ${MODES.map((mm, i) => `${mm.split(" ")[0]} ${fmt(allB[i], 3)}`).join(", ")}.`];
    if (q.m === 0) P.push("Sıfır dolgu görüntünün dışını siyah sayar: kenar pikselleri karanlığa çekilir (çerçeve gibi koyu bir şerit). Bu, CNN'lerde de görülen bir kenar etkisidir.");
    if (q.m === 4) P.push("Dairesel dolgu sol kenarın dışına sağ kenarın (karanlık) değerlerini koyar: görüntü periyodikmiş gibi davranır. Bu, DFT ile süzgeçlemenin varsayımıdır (M9.10); karşıt kenarlar farklıysa kenarda yapay bir sızıntı oluşur.");
    if (q.m === 1 || q.m === 3 || q.m === 2) P.push(q.slope !== 0 ? "Kopyalama ve yansıtma kenarda düzlük ya da simetri varsayar; eğimli bir bölgede küçük bir sapma kalır ama sıfır dolgunun koyu şeridi oluşmaz. REFLECT_101 kenar pikselini tekrar etmez (dcb|abcd), REFLECT eder (cba|abc)." : "Kenar düz olduğunda kopyalama ve yansıtma sapmasız sonuç verir.");
    const ext = range(n + 2 * 16).map((i) => i - 16);
    return {
      plots: [{ title: "Dolgulu satır ve süzgeç çıkışı (sol kenar)", xlabel: "konum (negatifler dolgu)", ylabel: "değer", wide: true, size: { w: 620, h: 230 }, xlim: [-16, 40], series: [{ name: "dolgulu girdi", x: ext.filter((i) => i < 40), y: ext.filter((i) => i < 40).map(val), kind: "scatter", c: 4, r: 2.5 }, { name: "çıkış", x: range(40), y: out.slice(0, 40), c: 0 }, { name: "kenarsız beklenen", x: range(40), y: ideal.slice(0, 40), c: 1, dash: true }], marks: [{ x: -0.5, label: "kenar" }] }],
      stats: [{ label: "Kenar sapması (ilk r piksel)", value: bias }, { label: "Bütün modlarda sapma", value: allB.map((v) => fmt(v, 2)).join(" / ") }],
      note: P.join(" "),
    };
  },
};
