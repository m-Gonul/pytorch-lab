// M10.3 — Çok thread'li integral dizi (önek toplamı, Ex. 1.2). Algoritmalar: (0) tara-sonra-düzelt: her thread bloğunu yerel tarar
// (oku + yaz), blok toplamları taranır, her thread bloğuna ofset ekler (oku + yaz) → bellek trafiği 4N eleman; (1) topla-sonra-tara: her
// thread bloğunun yalnız toplamını alır (oku), blok toplamları taranır, her thread ofsetle başlayıp bloğunu tarar (oku + yaz) → 3N.
// Maliyet modeli (çevrim): eleman başına hesap 1; bellek: dizi son düzey önbelleğe (8 MB) sığmıyorsa eleman başına aktarım bir çekirdekte
// 1 çevrim, toplam bant genişliği en çok m çekirdek eşdeğeri; thread yaratma cs (sıralı), her bariyer 2000 çevrim.
import { fmt, range } from "./plot.js";

export function model(N, p, alg, cs, m) {
  const big = N * 4 > 8e6, seq = N * (1 + (big ? 2 : 0)); // sıralı: oku + yaz, hesap 1
  if (p === 1) return { seq, par: seq };
  const traffic = alg === 0 ? 4 * N : 3 * N, bwTime = big ? traffic / Math.min(p, m) : 0, comp = (2 * N) / p;
  const par = cs * p + Math.max(comp, bwTime) + p + 2 * 2000;
  return { seq, par };
}
export default {
  params: [
    { id: "alg", label: "Algoritma", options: ["tara, sonra ofset ekle (4N trafik)", "topla, sonra ofsetle tara (3N trafik)"], value: 0 },
    { id: "p", label: "Thread sayısı p", min: 1, max: 32, value: 8, int: true },
    { id: "logN", label: "Dizi boyu N = 2^k", min: 8, max: 26, value: 12, int: true },
    { id: "cs", label: "Thread yaratma maliyeti (çevrim)", min: 100, max: 100000, value: 20000, log: true, int: true },
    { id: "m", label: "Bellek bant genişliği tavanı (çekirdek eşdeğeri)", min: 1, max: 16, value: 4, int: true },
  ],
  run(q) {
    const N = 2 ** q.logN, cur = model(N, q.p, q.alg, q.cs, q.m), sp = cur.seq / cur.par;
    const ks = range(19).map((k) => k + 8), sx = ks.map((k) => model(2 ** k, q.p, q.alg, q.cs, q.m)), spCurve = sx.map((o) => o.seq / o.par);
    const be = ks.find((k, i) => spCurve[i] >= 1);
    // küçük örnek: [1..8], 4 blok
    const a = range(8).map((i) => i + 1), B = 4, bs = range(B).map((b) => a.slice(2 * b, 2 * b + 2).reduce((x, y) => x + y, 0)), off = bs.map((_, b) => bs.slice(0, b).reduce((x, y) => x + y, 0));
    const P = [`N = ${N}, p = ${q.p}: sıralı ${fmt(cur.seq, 4)}, paralel ${fmt(cur.par, 4)} çevrim; hızlanma ${fmt(sp, 3)}.`];
    if (q.p === 1) P.push("Tek thread: paralel sürüm sıralıyla aynı.");
    else P.push(be !== undefined ? `Paralel sürüm ancak N ≥ 2^${be} (${2 ** be}) için kazandırıyor${be > 12 ? "; 4096 elemanda thread yaratma maliyeti işin kendisinden büyük" : ""}.` : "Bu ayarlarla paralel sürüm hiçbir boyutta kazandırmıyor.");
    if (N * 4 > 8e6) P.push(`Dizi (${fmt((N * 4) / 1e6, 3)} MB) son düzey önbelleğe sığmıyor: süreyi bellek trafiği belirliyor ve hızlanma bant genişliği tavanında (${q.m}) doyuyor; topla-sonra-tara trafiği 4N'den 3N'e indirir.`);
    P.push(`Algoritma örneği [1…8], 4 blok: blok toplamları (${bs.join(", ")}), ofsetler (${off.join(", ")}); 3. bloğa 10 eklenir.`);
    return {
      plots: [{ title: `Hızlanma ve dizi boyu (p = ${q.p})`, xlabel: "log₂ N", ylabel: "hızlanma", wide: true, size: { w: 620, h: 230 }, ylog: true, series: [0, 1].map((al) => ({ name: al === 0 ? "tara + ofset (4N)" : "topla + tara (3N)", x: ks, y: ks.map((k) => { const o = model(2 ** k, q.p, al, q.cs, q.m); return o.seq / o.par; }), c: al, dash: al !== q.alg, dots: al === q.alg })).concat([{ name: "1 (başa baş)", x: [8, 26], y: [1, 1], c: 4, dash: true }]), marks: [{ x: q.logN, label: "seçili N" }] }],
      stats: [{ label: "Hızlanma", value: fmt(sp, 3) }, { label: "Başa baş boyu", value: be !== undefined ? `2^${be}` : "yok" }, { label: "Sıralı / paralel (çevrim)", value: `${fmt(cur.seq, 4)} / ${fmt(cur.par, 4)}` }],
      note: P.join(" "),
    };
  },
};
