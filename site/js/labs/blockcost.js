// M4.8 — Aynı girdi/çıktı boyutlu beş CNN bloğunun maliyeti: standart K×K, grup konvolüsyonu, derinlemesine ayrılabilir,
// ResNet darboğazı (1×1 → K×K → 1×1) ve MobileNetV2 ters artığı (1×1 genişlet → derinlemesine → 1×1 daralt).
// Parametre, çarpma–toplama (MAC) ve kaba bir bellek trafiği modeliyle aritmetik yoğunluk (FLOP/bayt).
import { fmt } from "./plot.js";

// katman: [ad, Cin, Cout, K, gruplar]; "same" dolgu, adım 1, bias yok
export function layers(q) {
  const { Cin, Cout, K, g, t, r } = q, m = Math.max(1, Math.round(Cout / r)), e = Math.round(t * Cin);
  return {
    "standart": [[Cin, Cout, K, 1]],
    "grup (g)": [[Cin, Cout, K, g]],
    "derinlemesine ayrılabilir": [[Cin, Cin, K, Cin], [Cin, Cout, 1, 1]],
    "darboğaz (ResNet)": [[Cin, m, 1, 1], [m, m, K, 1], [m, Cout, 1, 1]],
    "ters artık (MobileNetV2)": [[Cin, e, 1, 1], [e, e, K, e], [e, Cout, 1, 1]],
  };
}
export function cost(L, HW, bytes = 4) {
  let P = 0, M = 0, B = 0;
  for (const [ci, co, k, g] of L) { const p = (k * k * ci * co) / g; P += p; M += p * HW; B += bytes * (ci * HW + p + co * HW); }
  return { P, M, B, I: (2 * M) / B };
}
const big = (n) => (n >= 1e9 ? `${fmt(n / 1e9, 4)} G` : n >= 1e6 ? `${fmt(n / 1e6, 4)} M` : n >= 1e3 ? `${fmt(n / 1e3, 4)} K` : fmt(n, 4));

export default {
  params: [
    { id: "Cin", label: "Girdi kanalı C_in", min: 1, max: 1024, value: 128, int: true, log: true },
    { id: "Cout", label: "Çıktı kanalı C_out", min: 1, max: 1024, value: 256, int: true, log: true },
    { id: "K", label: "Çekirdek boyu K", min: 1, max: 9, value: 3, int: true },
    { id: "H", label: "Öznitelik haritası boyu H = W", min: 1, max: 224, value: 28, int: true, log: true },
    { id: "g", label: "Grup sayısı g (grup konvolüsyonu)", min: 1, max: 64, value: 32, int: true, log: true },
    { id: "r", label: "Darboğaz daraltma oranı (C_out / ara kanal)", min: 1, max: 16, value: 4, int: true },
    { id: "t", label: "Ters artık genişleme oranı t", min: 1, max: 12, value: 6, int: true },
  ],
  validate(q) {
    const e = [];
    if (q.Cin % q.g !== 0 || q.Cout % q.g !== 0) e.push({ level: "error", msg: `Grup sayısı g = ${q.g}, hem C_in = ${q.Cin} hem C_out = ${q.Cout}'u tam bölmeli (PyTorch: "in_channels must be divisible by groups").` });
    if (q.Cin < 8) e.push({ level: "warn", msg: "Çok az girdi kanalı: derinlemesine konvolüsyon kanal başına tek bir uzamsal filtre öğrenir; girdi katmanından hemen sonra kullanılmaz (Xception ilk iki katmanda standart konvolüsyon kullanır)." });
    return e;
  },
  run(q) {
    const HW = q.H * q.H, Ls = layers(q), names = Object.keys(Ls), C = names.map((n) => cost(Ls[n], HW)), xs = names.map((_, i) => i);
    const xt = names.map((n, i) => [i, ["standart", "grup", "derinlemesine", "darboğaz", "ters artık"][i]]);
    const ratio = C[2].M / C[0].M, formula = 1 / q.Cout + 1 / (q.K * q.K);
    return {
      plots: [
        { title: "Parametre ve MAC (log)", xlabel: "", ylabel: "sayı", ylog: true, xlim: [-0.6, 4.6], xticks: xt, ylim: [Math.min(...C.map((c) => c.P)) / 5, Math.max(...C.map((c) => c.M)) * 5],
          series: [{ name: "parametre", x: xs, y: C.map((c) => c.P), kind: "bar", c: 0, bw: 0.35, off: -0.5 }, { name: `MAC (${q.H}×${q.H} harita)`, x: xs, y: C.map((c) => c.M), kind: "bar", c: 2, bw: 0.35, off: 0.5 }] },
        { title: "Aritmetik yoğunluk: FLOP / bellek baytı (log)", xlabel: "", ylabel: "FLOP / bayt", ylog: true, xlim: [-0.6, 4.6], xticks: xt, ylim: [Math.min(...C.map((c) => c.I), 1) / 3, Math.max(...C.map((c) => c.I), 150) * 3],
          series: [{ name: "yoğunluk (fp32, batch 1)", x: xs, y: C.map((c) => c.I), kind: "bar", c: 3 }], marks: [{ y: 150, label: "≈ GPU sırt noktası (ör. 300 TFLOP/s ÷ 2 TB/s)" }] },
      ],
      stats: [
        ...names.map((n, i) => ({ label: n, value: `${big(C[i].P)} parametre · ${big(C[i].M)} MAC · ${fmt(C[i].I, 3)} FLOP/B` })),
        { label: "Derinlemesine ayrılabilir / standart (MAC)", value: `${fmt(ratio, 4)} (formül 1/C_out + 1/K² = ${fmt(formula, 4)})` },
      ],
      note: `Standart konvolüsyonun her çıktı pikseli için K²·C_in·C_out çarpma gerekir. Derinlemesine ayrılabilir blok bunu K²·C_in (uzamsal, kanal başına) + C_in·C_out (kanal karışımı) ile değiştirir: oran 1/C_out + 1/K². Grup konvolüsyonu her çıktı kanalını yalnız C_in/g girdi kanalına bağlar: maliyet g'ye bölünür. ${C[2].I < 150 ? `<b>Ama hesap ≠ süre:</b> derinlemesine katmanın yoğunluğu (${fmt(Cost1(Ls, HW), 3)} FLOP/B) çok düşük — her ağırlık ve aktivasyon az sayıda çarpmada kullanılıyor, GPU'da süreyi bellek trafiği belirler. MobileNet'ler FLOP'ta küçük ama GPU'da FLOP'larının düşündüreceği kadar hızlı değildir; mobil işlemcilerde ise kazanç büyüktür.` : ""} Darboğaz bloğu pahalı K×K katmanını C_out/r kanala daraltarak ucuzlatır (burada standardın %${fmt((100 * C[3].M) / C[0].M, 3)}'i). Ters artık bunu tersine çevirir: girdi ve çıktı dar, ara katman t kat geniş ama derinlemesine; ${C[4].M < C[0].M ? `burada standardın %${fmt((100 * C[4].M) / C[0].M, 3)}'i` : `<b>bu ayarda standarttan bile pahalı</b> (%${fmt((100 * C[4].M) / C[0].M, 3)}) — MobileNetV2'de girdi/çıktı kanalları küçük (16–320) tutulur; genişleme, t·C_in kanallı bir temsili ucuza işlemenin yoludur, her C_in'de kendiliğinden tasarruf değildir`}.`,
    };
  },
};
function Cost1(Ls, HW) { return cost([Ls["derinlemesine ayrılabilir"][0]], HW).I; }
