// M3.8 — Evrensel yaklaşım, yapıcı: (a) genişlik — K ReLU menteşesiyle parçalı doğrusal ara değerleme; (b) derinlik — çadır haritasının
// L kez bileşkesi 2L nöronla 2^L doğrusal parça üretir, tek gizli katman için ≥ 2^L − 1 nöron gerekir (Telgarsky 2016).
import { fmt, linspace, range } from "./plot.js";

export const TARGETS = [
  { name: "sin(2πx)", f: (x) => Math.sin(2 * Math.PI * x), d2: 4 * Math.PI ** 2 },
  { name: "Gauss tepesi exp(−50(x − ½)²)", f: (x) => Math.exp(-50 * (x - 0.5) ** 2), d2: 100 },
  { name: "x²", f: (x) => x * x, d2: 2 },
  { name: "|x − 0.3| (kırık)", f: (x) => Math.abs(x - 0.3), d2: null },
  { name: "basamak (x > 0.5) — süreksiz", f: (x) => (x > 0.5 ? 1 : 0), d2: null },
];
const relu = (z) => (z > 0 ? z : 0);
// K eşit aralıklı düğümde f'yi ara değerleyen tek gizli katmanlı ReLU ağı: f̂(x) = f(0) + Σ_k c_k ReLU(x − t_k)
export function hinge(f, K) {
  const t = range(K + 1).map((k) => k / K), s = range(K).map((k) => (f(t[k + 1]) - f(t[k])) * K), c = s.map((v, k) => v - (k ? s[k - 1] : 0));
  return { t, c, b: f(0), at: (x) => c.reduce((a, ck, k) => a + ck * relu(x - t[k]), f(0)) };
}
export const tent = (x) => 2 * relu(x) - 4 * relu(x - 0.5);
export const tentL = (x, L) => { for (let i = 0; i < L; i++) x = tent(x); return x; };

export default {
  params: [
    { id: "mode", label: "Gösterim", options: ["genişlik: K ReLU ile parçalı doğrusal yaklaşım", "derinlik: çadır haritasının L katlı bileşkesi"], value: 0 },
    { id: "tg", label: "Hedef fonksiyon ([0, 1] üzerinde)", options: TARGETS.map((T) => T.name), value: 0, on: { mode: [0] } },
    { id: "K", label: "Gizli ReLU nöron sayısı K", min: 1, max: 128, value: 6, int: true, log: true, on: { mode: [0] } },
    { id: "L", label: "Derinlik L (her katmanda 2 ReLU)", min: 1, max: 10, value: 3, int: true, on: { mode: [1] } },
  ],
  run(q) {
    const xs = linspace(0, 1, 801);
    if (q.mode === 0) {
      const T = TARGETS[q.tg], K = Math.round(q.K), H = hinge(T.f, K), err = (h) => Math.max(...xs.map((x) => Math.abs(h.at(x) - T.f(x))));
      const e = err(H), Ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], Es = Ks.map((k) => err(hinge(T.f, k)));
      const bound = T.d2 != null ? T.d2 / (8 * K * K) : null;
      return {
        plots: [
          { title: `${T.name} ve ${K} ReLU'lu yaklaşım`, xlabel: "x", ylabel: "y", series: [{ name: "hedef f", x: xs, y: xs.map(T.f), c: 5, dash: true }, { name: `f̂ (K = ${K})`, x: xs, y: xs.map(H.at), c: 0 }, { name: "menteşe noktaları tₖ", x: H.t.slice(0, K), y: H.t.slice(0, K).map(H.at), kind: "scatter", c: 2, r: 3.5 }] },
          { title: "En büyük hata, K'ya göre (log–log)", xlabel: "K", ylabel: "max |f − f̂|", ylog: true, series: [{ name: "ölçülen", x: Ks, y: Es.map((v) => Math.max(v, 1e-12)), c: 0, dots: true }, ...(T.d2 != null ? [{ name: "sınır max|f″| / (8K²)", x: Ks, y: Ks.map((k) => T.d2 / (8 * k * k)), c: 4, dash: true }] : []), { name: "seçili K", x: [K], y: [Math.max(e, 1e-12)], kind: "scatter", c: 2, r: 6 }] },
        ],
        stats: [
          { label: "En büyük hata max|f − f̂|", value: e },
          { label: "Teorik sınır h²·max|f″|/8 (h = 1/K)", value: bound != null ? bound : "— (f″ yok: f iki kez türevli değil)" },
          { label: "Parametre sayısı (K gizli + çıktı)", value: 3 * K + 1 },
        ],
        note: `Her ReLU nöronu ReLU(x − tₖ) bir <b>menteşe</b>dir: tₖ'de eğimi cₖ kadar değiştirir. K menteşe, K parçalı bir doğrusal fonksiyon kurar; düğümlerde f'ye tam eşittir. ${T.d2 != null ? `f iki kez türevli olduğu için hata h²·max|f″|/8 ile sınırlı: K'yı ikiye katlamak hatayı ≈ 4'e böler (log–log'da eğim −2).` : q.tg === 3 ? "Kırık nokta (0.3) bir düğüme denk gelmezse hata yalnız o aralıkta kalır ve ∝ 1/K azalır; denk gelirse sıfır olur." : "Süreksiz bir fonksiyonda en büyük hata K'yla azalmaz: bu inşada sıçramanın hemen sağında ≈ 1; hiçbir sürekli fonksiyon (dolayısıyla hiçbir ReLU ağı) en büyük hatayı 0.5'in altına indiremez: evrensel yaklaşım <b>sürekli</b> fonksiyonlar içindir (süreksizlikte yaklaşım yalnız ortalama anlamda iyileşir)."} Teorem yalnız <b>varlık</b> söyler: kaç nöron gerektiği, onların gradyan inişiyle bulunup bulunamayacağı ve genellemenin ne olacağı ayrı sorulardır.`,
      };
    }
    const L = Math.round(q.L), Ls = range(10).map((i) => i + 1);
    let br = 0; const xd = linspace(0, 1, 1025), ys = xd.map((x) => tentL(x, L)); for (let i = 1; i < xd.length - 1; i++) if (Math.abs(ys[i + 1] - 2 * ys[i] + ys[i - 1]) > 1e-9) br++;

    return {
      plots: [
        { title: `Çadır haritası g(x) = 2·ReLU(x) − 4·ReLU(x − ½), ${L} kez bileşke`, xlabel: "x", ylabel: `g${"⁰¹²³⁴⁵⁶⁷⁸⁹"[L] || ""}(x)`, ylim: [-0.05, 1.05], series: [{ name: `${L} katman, ${2 * L} ReLU`, x: xd, y: ys, c: 0 }] },
        { title: "Aynı testere fonksiyonu için gereken ReLU sayısı", xlabel: "L", ylabel: "nöron", ylog: true, ylim: [1, 2000], series: [{ name: "derin: 2L", x: Ls, y: Ls.map((l) => 2 * l), c: 1, dots: true }, { name: "tek gizli katman: ≥ 2ᴸ − 1", x: Ls, y: Ls.map((l) => 2 ** l - 1), c: 2, dots: true }, { name: "seçili L", x: [L], y: [2 ** L - 1], kind: "scatter", c: 5, r: 6 }] },
      ],
      stats: [
        { label: "Doğrusal parça sayısı", value: 2 ** L },
        { label: "İç kırılma noktası (ölçülen, 2¹⁰ aralıklı ızgarada)", value: br },
        { label: "Derin ağ: nöron / katman", value: `${2 * L} / ${L}` },
        { label: "Tek gizli katman için en az nöron", value: 2 ** L - 1 },
      ],
      note: `Çadır haritası [0, 1]'i ikiye katlayıp katlar; her bileşke diş sayısını ikiye katlar: L katman → 2ᴸ doğrusal parça, yalnız 2L nöronla. Tek gizli katmanlı bir ReLU ağında (1 boyutlu girdi) her nöron en fazla <b>bir</b> kırılma noktası ekler; 2ᴸ − 1 kırılma için en az 2ᴸ − 1 nöron gerekir. Derinlik, bazı fonksiyonları <b>üstel olarak</b> daha az nöronla ifade eder (Telgarsky 2016). Bu bir ifade gücü sonucudur; gradyan inişinin bu çözümü bulacağını söylemez.`,
    };
  },
};
