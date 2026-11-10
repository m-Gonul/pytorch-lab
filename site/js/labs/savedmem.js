// M6.1 — Geri geçiş için saklanan tensörlerin belleği: Pre-LN MLP bloğu x → LN → Linear(d, 4d) → aktivasyon → [Dropout] → Linear(4d, d) → + x.
// Her düğümün sakladıkları (PyTorch 2.x'te _saved_* öznitelikleriyle doğrulandı): LayerNorm girdi + satır başına ortalama/rstd; Linear
// (addmm) girdisi ve ağırlığı (parametre, ek bellek değil); ReLU/tanh çıktısı; GELU girdisi; Dropout (mul) maskesi. Aynı depoyu paylaşan
// tensörler bir kez sayılır (ReLU çıktısı = ikinci Linear'ın girdisi). CPU ölçümüyle (saved_tensors_hooks) katman başına 6/10/14 birim doğrulandı.
import { fmt, range } from "./plot.js";

const ACT = ["ReLU", "tanh", "GELU"];
export function layerSaved(q) {
  // birimler: eleman sayısı (N ile çarpılacak) ve bayt/eleman
  const fb = q.dt === 1 ? 2 : 4, items = [];
  items.push({ name: "LN girdisi (artık akış)", el: q.d, b: fb });
  items.push({ name: "LN ortalama/rstd", el: 2, b: fb });
  items.push({ name: "Linear₁ girdisi (LN çıktısı)", el: q.d, b: fb });
  if (q.act === 2) items.push({ name: "GELU girdisi", el: 4 * q.d, b: fb });
  else items.push({ name: `${ACT[q.act]} çıktısı`, el: 4 * q.d, b: fb });
  if (q.drop) {
    items.push({ name: "Dropout maskesi", el: 4 * q.d, b: q.mask === 1 ? 1 : fb });
    items.push({ name: "Linear₂ girdisi (Dropout çıktısı)", el: 4 * q.d, b: fb });
  } else if (q.act === 2) items.push({ name: "Linear₂ girdisi (GELU çıktısı)", el: 4 * q.d, b: fb });
  // ReLU/tanh, dropout yok: Linear₂ girdisi aktivasyon çıktısıyla aynı depo, ek bellek yok
  return items;
}
const GB = 1024 ** 3;
export default {
  params: [
    { id: "N", label: "Satır sayısı N = batch × dizi uzunluğu", min: 256, max: 131072, value: 8192, int: true, log: true },
    { id: "d", label: "Model genişliği d", min: 128, max: 8192, value: 1024, int: true, log: true },
    { id: "L", label: "Katman sayısı", min: 1, max: 96, value: 24, int: true },
    { id: "act", label: "Aktivasyon", options: ACT, value: 0 },
    { id: "drop", label: "Dropout", options: ["yok", "var (p = 0.1)"], value: 0 },
    { id: "mask", label: "Dropout maskesinin tipi", options: ["float32 (CPU'da ölçülen)", "bool, 1 bayt (CUDA'daki birleşik çekirdek)"], value: 0, on: { drop: [1] } },
    { id: "dt", label: "Saklanan tensörlerin tipi", options: ["float32", "bf16 (saved_tensors_hooks ile sıkıştırılmış)"], value: 0 },
  ],
  run(q) {
    const items = layerSaved(q), perRow = items.reduce((s, it) => s + it.el * it.b, 0), layer = perRow * q.N, act = layer * q.L;
    const params = q.L * (8 * q.d * q.d + 7 * q.d), state = params * 16;
    const units = items.filter((it) => it.el > 2).reduce((s, it) => s + (it.el * it.b) / (q.dt === 1 ? 2 : 4), 0) / q.d;
    const Ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], curve = Ns.map((n) => (perRow * n * q.L) / GB);
    const cross = state / (perRow * q.L);
    const P = [`Bir katman satır başına ${fmt(units, 3)} × d elemanlık tensör saklıyor; ${q.L} katman ve N = ${q.N} için aktivasyon belleği ${fmt(act / GB, 3)} GiB. Parametreler ${fmt(params / 1e6, 4)} M; parametre + gradyan + Adam durumları (fp32, 16 bayt/parametre) ${fmt(state / GB, 3)} GiB.`];
    P.push(act > state ? `Aktivasyonlar baskın: N ≈ ${fmt(cross, 3)} satırın üstünde eğitim belleğinin çoğu saklanan tensörlerdir. Bu yüzden batch ve dizi uzunluğu belleği parametre sayısından daha çok belirler.` : `Bu N'de parametre ve optimizasyon durumu baskın; aktivasyonlar N ≈ ${fmt(cross, 3)} satırda onları geçer.`);
    if (q.act === 2 && !q.drop) P.push("GELU girdisini saklar; ikinci Linear da GELU çıktısını saklar: aynı boyutta iki ayrı tensör (10 birim). ReLU'da ikisi aynı depodur (6 birim).");
    if (q.act !== 2 && !q.drop) P.push(`${ACT[q.act]} kendi çıktısını saklar ve ikinci Linear'ın girdisi de aynı tensördür; motor depoyu bir kez tutar.`);
    if (q.drop) P.push(`Dropout bir çarpma düğümüdür ve yalnız maskeyi saklar; ikinci Linear Dropout'un yeni çıktısını saklar. Aktivasyon türünden bağımsız 14 birim${q.mask === 1 ? "; bool maske ile maskenin payı 4 kat küçülür" : ""}.`);
    if (q.dt === 1) P.push("saved_tensors_hooks ile paketlenen her kayan noktalı tensör bf16'ya çevrilir: ileri geçiş aynıdır, geri geçiş yuvarlanmış değerlerle yapılır ve bellek yarıya iner.");
    return {
      plots: [
        { title: "Bir katmanın sakladıkları (MiB)", xlabel: "", ylabel: "MiB", yzero: true, xlim: [-0.6, items.length - 0.4], series: [{ name: "saklanan", x: range(items.length), y: items.map((it) => (it.el * it.b * q.N) / 1024 ** 2), kind: "bar", c: 0 }], xticks: items.map((it, i) => [i, it.name.split(" (")[0].replace("Linear₁ girdisi", "Lin₁ girdi").replace("Linear₂ girdisi", "Lin₂ girdi").replace("LN ortalama/rstd", "LN ist.").replace("Dropout maskesi", "maske").replace("LN girdisi", "LN girdi")]) },
        { title: "Toplam aktivasyon belleği ve N", xlabel: "N (log)", ylabel: "GiB", series: [
          { name: "aktivasyonlar", x: Ns.map(Math.log2), y: curve, c: 0 },
          { name: "parametre + grad + Adam", x: Ns.map(Math.log2), y: Ns.map(() => state / GB), c: 1, dash: true }],
          xticks: Ns.filter((_, i) => i % 2 === 0).map((n) => [Math.log2(n), n >= 1024 ? `${n / 1024}K` : String(n)]), marks: [{ x: Math.log2(q.N), label: "N" }] },
      ],
      stats: [
        { label: "Katman başına (birim × N·d)", value: units },
        { label: "Aktivasyon belleği (GiB)", value: act / GB },
        { label: "Parametre + grad + Adam (GiB)", value: state / GB },
      ],
      note: P.join(" "),
    };
  },
};
