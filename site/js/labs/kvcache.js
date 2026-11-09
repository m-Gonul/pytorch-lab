// M4.7 — Otoregresif üretimde KV önbelleği: her adımda tüm öneki yeniden hesaplamak vs yalnız yeni token'ı işleyip geçmiş K/V'yi
// okumak. Toplam FLOP (hesap) ve önbellek belleği; çoklu sorgu / grup sorgulu dikkat (MQA/GQA) ile K/V başlarını paylaşmak.
import { fmt, range } from "./plot.js";

export function costs(q) {
  const { d, L, P, n } = q, N = 12 * L * d * d + 50257 * d; // gömme hariç bloklar + çıkış izdüşümü (V = 50 257)
  const fwd = (t) => 2 * N + 4 * L * t * d; // t. konumdaki bir token'ın ileri FLOP'u (QKᵀ ve αV, t anahtara karşı)
  const full = (t) => 2 * N * t + 2 * L * d * t * (t + 1); // t token'lık önekin tamamı: Σ_{i=1..t} fwd(i)
  const prefill = full(P), noCache = [], withCache = []; let a = 0, b = 0;
  // k. adım (k = 0, 1, …) (k+1). token'ı üretir; bağlamda P + k token var. Önbellekli: ilk adım istemi işler (prefill), sonrakiler tek token.
  for (let k = 0; k < n; k++) { a += full(P + k); b += k === 0 ? prefill : fwd(P + k); noCache.push(a); withCache.push(b); }
  const hd = d / q.h, kvBytes = (T) => 2 * L * T * q.B * q.kv * hd * (q.bits === 0 ? 2 : 4);
  return { N, noCache, withCache, kvBytes, hd, prefill };
}
const human = (b) => (b >= 2 ** 30 ? `${fmt(b / 2 ** 30, 3)} GiB` : b >= 2 ** 20 ? `${fmt(b / 2 ** 20, 3)} MiB` : `${fmt(b / 2 ** 10, 3)} KiB`);

export default {
  params: [
    { id: "d", label: "Model boyu d", min: 128, max: 8192, value: 768, int: true, log: true },
    { id: "L", label: "Blok sayısı L", min: 1, max: 96, value: 12, int: true },
    { id: "h", label: "Sorgu başı sayısı h", min: 1, max: 128, value: 12, int: true },
    { id: "kv", label: "K/V başı sayısı (h = çok başlı, 1 = MQA, arası = GQA)", min: 1, max: 128, value: 12, int: true },
    { id: "P", label: "İstem (prompt) uzunluğu", min: 1, max: 2048, value: 64, int: true, log: true },
    { id: "n", label: "Üretilecek token sayısı", min: 1, max: 1024, value: 256, int: true, log: true },
    { id: "B", label: "Batch (eşzamanlı istek) sayısı", min: 1, max: 256, value: 1, int: true, log: true },
    { id: "bits", label: "K/V sayı biçimi", options: ["16 bit", "32 bit"], value: 0 },
  ],
  validate(q) {
    const e = [];
    if (q.d % q.h !== 0) e.push({ level: "error", msg: `d = ${q.d}, h = ${q.h}'ye tam bölünmüyor: baş boyu d/h tam sayı olmalı.` });
    if (q.kv > q.h || q.h % q.kv !== 0) e.push({ level: "error", msg: `K/V başı sayısı (${q.kv}) sorgu başı sayısını (${q.h}) tam bölmeli ve ondan büyük olmamalı: GQA'da her K/V başını h/kv sorgu başı paylaşır.` });
    return e;
  },
  run(q) {
    const c = costs(q), steps = range(q.n).map((k) => k + 1), Tmax = q.P + q.n, wBytes = (c.N + 1024 * q.d) * (q.bits === 0 ? 2 : 4);
    const Ts = range(41).map((i) => Math.round(2 ** (6 + (i * 11) / 40)));
    return {
      plots: [
        { title: "Toplam ileri FLOP: üretilen token sayısına göre (log)", xlabel: "üretilen token", ylabel: "birikimli FLOP", ylog: true, series: [{ name: "önbelleksiz (her adımda tüm önek)", x: steps, y: c.noCache, c: 5 }, { name: "KV önbellekli (yalnız yeni token)", x: steps, y: c.withCache, c: 0 }] },
        { title: `Bellek: K/V önbelleği vs ağırlıklar (batch ${q.B}, log–log)`, xlabel: "bağlam uzunluğu (log₂)", ylabel: "bayt", ylog: true, xticks: [[6, "64"], [8, "256"], [10, "1K"], [12, "4K"], [14, "16K"], [16, "64K"]],
          series: [{ name: "K/V önbelleği", x: Ts.map(Math.log2), y: Ts.map(c.kvBytes), c: 2 }, { name: "model ağırlıkları", x: Ts.map(Math.log2), y: Ts.map(() => wBytes), c: 4 }], marks: [{ x: Math.log2(Tmax), label: `P + n = ${Tmax}` }] },
      ],
      stats: [
        { label: "Önbelleksiz / önbellekli toplam FLOP", value: `× ${fmt(c.noCache[q.n - 1] / c.withCache[q.n - 1], 3)}` },
        { label: `K/V önbelleği (P + n = ${Tmax} token)`, value: human(c.kvBytes(Tmax)) },
        { label: "Token başına önbellek", value: human(c.kvBytes(1) / q.B) },
        { label: "Ağırlıklar (yaklaşık)", value: human(wBytes) },
        { label: "Çok başlıya göre önbellek küçülmesi", value: `× ${fmt(q.h / q.kv, 3)}` },
      ],
      note: `Nedensel maske yüzünden eski konumların K, V ve çıktıları yeni token geldiğinde <b>değişmez</b>; önbelleksiz üretim bunları her adımda boşuna yeniden hesaplar (toplam iş ≈ kareseldir). Önbellekte her adım yalnız yeni token'ın ileri geçişidir: ≈ 2N + 4·L·t·d FLOP. Bedeli bellek: katman × token × 2 (K ve V) × K/V başı × baş boyu × bayt. ${c.kvBytes(Tmax) > wBytes ? "<b>Bu ayarda önbellek ağırlıklardan büyük</b> — uzun bağlam ve büyük batch'te çıkarımın darboğazı budur." : "Bu ayarda önbellek ağırlıklardan küçük."} ${q.kv < q.h ? `${q.kv === 1 ? "MQA" : "GQA"}: ${q.h / q.kv} sorgu başı bir K/V başını paylaşıyor; önbellek ${q.h / q.kv} kat küçük (kalite kaybı genelde küçüktür; Ainslie ve ark. 2023).` : "K/V başı sayısını azaltmayı (MQA/GQA) dene."} Üretim adımı ağırlıkları ve önbelleği bellekten okumakla sınırlıdır (bellek bant genişliği), istem işleme (prefill) ise hesapla.`,
    };
  },
};
