// M10.3 — Çok thread'li histogram (Ex. 1.1): p thread, N piksellik 8 bitlik görüntü, 256 kutu. Stratejiler: (0) paylaşılan histogram +
// atomik artırma, (1) paylaşılan histogram + tek küresel mutex, (2) thread'e özel histogramlar + sonda birleştirme. Maliyet modeli
// (çevrim): piksel okuma + kutu hesabı 1; atomik artırma, kutunun önbellek satırı başka bir çekirdekteyse satır transferi ct (MESI), aynı
// çekirdekteyse kilitli artırma ≈ 20 çevrim; mutex alma/bırakma cm ve kritik bölge seri; özel histogramda artırma 1, birleştirme 256·p. Thread'ler pikselleri
// döngüsel sırayla işler (eşzamanlı ilerleme); bir kutunun satırı (64 B = 16 kutu) son kim yazdıysa onda. Görüntü dağılımı: düzgün,
// gerçekçi (doğal görüntü benzeri iki tepeli) ya da dar tepeli (kutuların %90'ı birkaç değerde).
import { fmt, range } from "./plot.js";

function pixels(r, N, dist) {
  const v = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (dist === 0) v[i] = Math.floor(r() * 256);
    else if (dist === 1) { const g = r() < 0.6 ? 90 + 25 * r.normal() : 180 + 20 * r.normal(); v[i] = Math.max(0, Math.min(255, Math.round(g))); }
    else v[i] = r() < 0.9 ? 128 + Math.floor(r() * 4) : Math.floor(r() * 256);
  }
  return v;
}
export function simulate(v, p, strat, ct, cm) {
  const N = v.length, per = Math.ceil(N / p), owner = new Int32Array(16).fill(-1), clock = new Float64Array(p);
  let transfers = 0;
  if (strat === 2) { for (let t = 0; t < p; t++) clock[t] = Math.min(per, N - t * per) * 2; const T = Math.max(...clock) + 256 * p; return { T, transfers: 0 }; }
  if (strat === 1) { // küresel mutex: kritik bölgeler seri; toplam süre ≈ max(paralel kısım, seri kısım)
    const serial = N * (cm + 1), par = per * 1; return { T: Math.max(par + serial / 1, serial), transfers: 0 };
  }
  // atomik: thread'ler eşzamanlı ilerler; i. adımda her thread kendi bloğunun i. pikselini işler. Bir önbellek satırı aynı anda tek
  // çekirdekte olabilir: satır başka çekirdekteyse transfer, satırın serbest kalmasını bekler (sıcak satırda transferler sıraya girer).
  const lineFree = new Float64Array(16);
  for (let i = 0; i < per; i++) for (let t = 0; t < p; t++) {
    const k = t * per + i; if (k >= N) continue;
    const L = v[k] >> 4;
    if (owner[L] !== t && owner[L] !== -1) { const start = Math.max(clock[t], lineFree[L]); clock[t] = start + ct + 21; transfers++; }
    else clock[t] += 21; // okuma + kutu (1) + kilitli artırma (lock xadd ≈ 20)
    owner[L] = t; lineFree[L] = Math.max(lineFree[L], clock[t]);
  }
  return { T: Math.max(...clock), transfers };
}
export default {
  params: [
    { id: "strat", label: "Strateji", options: ["paylaşılan histogram + atomik", "paylaşılan histogram + küresel mutex", "thread'e özel histogram + birleştirme"], value: 0 },
    { id: "p", label: "Thread sayısı p", min: 1, max: 32, value: 8, int: true },
    { id: "logN", label: "Piksel sayısı N = 2^k", min: 10, max: 20, value: 16, int: true },
    { id: "dist", label: "Görüntü histogramı", options: ["düzgün", "doğal (iki tepeli)", "dar tepeli (%90 dört değerde)"], value: 1 },
    { id: "ct", label: "Önbellek satırı transfer maliyeti (çevrim)", min: 0, max: 200, value: 60, int: true },
    { id: "cm", label: "Mutex al/bırak maliyeti (çevrim)", min: 5, max: 200, value: 40, int: true, on: { strat: [1] } },
  ],
  seed: true,
  run(q, r) {
    const N = 2 ** q.logN, v = pixels(r, N, q.dist), T1 = 2 * N;
    const cur = simulate(v, q.p, q.strat, q.ct, q.cm), sp = T1 / cur.T;
    const ps = [1, 2, 4, 8, 16, 32], names = ["atomik", "küresel mutex", "özel + birleştirme"];
    const curves = [0, 1, 2].map((s) => ps.map((p) => T1 / simulate(v, p, s, q.ct, q.cm).T));
    const P = [`${names[q.strat]}, p = ${q.p}: hızlanma ${fmt(sp, 3)} (tek thread'li histogram ${T1} çevrim, bu strateji ${fmt(cur.T, 4)}).`];
    if (q.strat === 0) P.push(`Önbellek satırı transferi: ${cur.transfers} (${fmt(cur.transfers / N, 3)} piksel başına). Kutular 16'şarlık satırlarda; aynı satıra yazan çekirdekler satırı birbirinden çekiyor.${q.dist === 2 ? " Dar tepeli histogramda neredeyse her artırma aynı satıra düşüyor." : ""}`);
    if (q.strat === 1) P.push("Her piksel için kilit alınıyor: kritik bölge bütün işi sıraya sokuyor ve kilit maliyeti ekleniyor — tek thread'den yavaş.");
    if (q.strat === 2) P.push(`Paylaşım yok: yalnız sonda 256 × ${q.p} = ${256 * q.p} toplama. Küçük N'de birleştirme payı büyür.`);
    return {
      plots: [{ title: "Hızlanma ve thread sayısı", xlabel: "p", ylabel: "hızlanma", wide: true, size: { w: 620, h: 230 }, series: [{ name: "ideal", x: ps, y: ps, c: 4, dash: true }, ...[0, 1, 2].map((s) => ({ name: names[s], x: ps, y: curves[s], c: s, dots: true }))], marks: [{ x: q.p, label: "seçili p" }] }],
      stats: [{ label: "Hızlanma", value: fmt(sp, 3) }, { label: "Süre (çevrim)", value: fmt(cur.T, 4) }, ...(q.strat === 0 ? [{ label: "Satır transferi / piksel", value: fmt(cur.transfers / N, 3) }] : [])],
      note: P.join(" "),
    };
  },
};
