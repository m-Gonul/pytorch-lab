// M10.7 — GPU'da karolu 2-B evrişim: çıktı karosu T × T, filtre K × K, yarıçap r = (K − 1)/2. Stratejiler:
// (0) doğrudan: her çıktı K² girdiyi global bellekten okur; (1) girdi boyutlu blok: blok (T + K − 1)² thread, her thread bir girdi öğesi
// yükler, yalnız iç T² thread çıktı hesaplar; (2) çıktı boyutlu blok: blok T² thread, girdi karosu (T + K − 1)² öğe birkaç turda
// yüklenir; (3) haleyi önbelleğe bırak: blok T², paylaşımlı bellekte yalnız iç T² öğe, hale öğeleri global bellekten (L1/L2) okunur.
// Ölçütler: çıktı başına global okuma (önbellek yokken), paylaşımlı bellek, boşta thread oranı, en büyük blok sınırı (1024 thread).
import { fmt, range } from "./plot.js";

export function stats(T, K, s) {
  const I = T + K - 1, r = (K - 1) / 2;
  if (s === 0) return { reads: K * K, smem: 0, idle: 0, threads: T * T, loads: 1 };
  if (s === 1) return { reads: (I * I) / (T * T), smem: I * I * 4, idle: 1 - (T * T) / (I * I), threads: I * I, loads: 1 };
  if (s === 2) return { reads: (I * I) / (T * T), smem: I * I * 4, idle: 0, threads: T * T, loads: Math.ceil((I * I) / (T * T)) };
  // (3) iç karo paylaşımlıda; hale okumaları: her çıktı için penceresinin karonun dışına düşen kısmı (önbellekten, ama global erişim)
  let halo = 0; for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const yy = y + dy, xx = x + dx; if (yy < 0 || yy >= T || xx < 0 || xx >= T) halo++; }
  return { reads: 1 + halo / (T * T), smem: T * T * 4, idle: 0, threads: T * T, loads: 1, halo: halo / (T * T) };
}
const NAMES = ["doğrudan (paylaşımlı bellek yok)", "girdi boyutlu blok", "çıktı boyutlu blok, çok turlu yükleme", "iç karo paylaşımlıda, hale önbellekten"];
export default {
  params: [
    { id: "s", label: "Strateji", options: NAMES, value: 2 },
    { id: "T", label: "Çıktı karosu T", min: 4, max: 32, step: 4, value: 16, int: true },
    { id: "K", label: "Filtre K × K", options: ["3 × 3", "5 × 5", "7 × 7", "11 × 11", "21 × 21"], value: 1 },
  ],
  validate(q) { const K = [3, 5, 7, 11, 21][q.K], st = stats(q.T, K, q.s); return st.threads > 1024 ? [{ level: "error", msg: `Strateji: blok ${st.threads} thread ister, en çok 1024 olabilir (çıktı karosunu küçült).` }] : []; },
  run(q) {
    const K = [3, 5, 7, 11, 21][q.K], st = stats(q.T, K, q.s), I = q.T + K - 1;
    const Ts = [4, 8, 12, 16, 20, 24, 28, 32], ratio = (t) => (K * K) / stats(t, K, 2).reads, ceil = K * K;
    const P = [`${NAMES[q.s]}, T = ${q.T}, K = ${K}: çıktı başına ${fmt(st.reads, 3)} global okuma (doğrudan ${K * K}; azalma ${fmt((K * K) / st.reads, 3)} kat). Girdi karosu ${I} × ${I} = ${I * I} öğe; paylaşımlı bellek ${fmt(st.smem / 1024, 3)} KB; blok ${st.threads} thread.`];
    if (q.s === 1) P.push(`Thread'lerin ${fmt(100 * st.idle, 3)}%'i yalnız yükleme yapıp çıktı hesaplamıyor.`);
    if (q.s === 2) P.push(`Her thread girdi karosundan ${st.loads} öğe yüklüyor (sınır denetimli döngü).`);
    if (q.s === 3) P.push(`Paylaşımlı bellek iç karoyla sınırlı; çıktı başına ${fmt(st.halo, 3)} hale okuması L1/L2'ye gidiyor: komşu bloklar bunları zaten getirmişse önbellekten gelir.`);
    P.push(`T → ∞ sınırında azalma K² = ${ceil} kata yaklaşır; küçük karolarda hale payı büyüktür: azalma oranı K²T² / (T + K − 1)².`);
    return {
      plots: [{ title: `Global okuma azalması ve karo boyu (K = ${K})`, xlabel: "çıktı karosu T", ylabel: "azalma (kat)", wide: true, size: { w: 620, h: 220 }, ylim: [0, ceil * 1.2], series: [{ name: "çıktı/girdi boyutlu blok", x: Ts, y: Ts.map(ratio), c: 0, dots: true }, { name: "üst sınır K²", x: [4, 32], y: [ceil, ceil], c: 4, dash: true }], marks: [{ x: q.T, label: "seçili T" }] }],
      stats: [{ label: "Çıktı başına global okuma", value: fmt(st.reads, 3) }, { label: "Azalma (kat)", value: fmt((K * K) / st.reads, 3) }, { label: "Paylaşımlı bellek", value: `${fmt(st.smem / 1024, 3)} KB` }, { label: "Boşta thread", value: `${fmt(100 * st.idle, 3)}%` }],
      note: P.join(" "),
    };
  },
};
