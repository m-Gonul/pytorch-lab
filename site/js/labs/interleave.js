// M10.2 — Veri yarışı ve yarış koşulu: T thread paylaşılan bir sayacı n kez artırır. Korumasız artırma üç adımdır: oku (yazmaca),
// artır (yerel), yaz. "Atomik oku + atomik yaz" (slayt 53'teki iki ayrı atomic bloğu) veri yarışını giderir ama oku–yaz arası kesilebilir:
// yarış koşulu sürer. "Atomik fetch_add / mutex" üç adımı bölünmez yapar. Zamanlayıcı: her adımda çalışan thread 1 − q olasılıkla devam
// eder, q olasılıkla rastgele başka bir thread'e geçilir (önalım). Ayrıca 2 thread × 1 artırım için 20 iç içe geçmenin tamamı sayılır.
import { fmt, range } from "./plot.js";

function runOnce(T, n, q, mode, r) {
  let shared = 0; const pc = new Int32Array(T), reg = new Float64Array(T), left = Int32Array.from({ length: T }, () => n), alive = range(T);
  let cur = Math.floor(r() * T);
  while (alive.length) {
    if (!alive.includes(cur) || r() < q) cur = alive[Math.floor(r() * alive.length)];
    const t = cur;
    if (mode === 2) { shared += 1; left[t]--; } // bölünmez artırma
    else if (pc[t] === 0) { reg[t] = shared; pc[t] = 1; }
    else if (pc[t] === 1) { reg[t] += 1; pc[t] = 2; }
    else { shared = reg[t]; pc[t] = 0; left[t]--; }
    if (left[t] === 0 && (mode === 2 || pc[t] === 0)) alive.splice(alive.indexOf(t), 1);
  }
  return shared;
}
// 2 thread × 1 artırım (3 adım): bütün C(6, 3) = 20 sırayı say
function enumerate(mode) {
  if (mode === 2) return { total: 2, ok: 2 }; // bölünmez: yalnız 2 sıra (A sonra B, B sonra A)
  let total = 0, ok = 0;
  const rec = (a, b, sh, ra, rb, seq) => {
    if (a === 3 && b === 3) { total++; if (sh === 2) ok++; return; }
    if (a < 3) { let s = sh, x = ra; if (a === 0) x = s; else if (a === 1) x += 1; else s = x; rec(a + 1, b, s, x, rb); }
    if (b < 3) { let s = sh, y = rb; if (b === 0) y = s; else if (b === 1) y += 1; else s = y; rec(a, b + 1, s, ra, y); }
  };
  rec(0, 0, 0, 0, 0);
  return { total, ok };
}
export default {
  params: [
    { id: "mode", label: "Artırma", options: ["korumasız oku–artır–yaz (veri yarışı)", "atomik oku + atomik yaz, arada yerel artırma", "atomik fetch_add (ya da mutex)"], value: 0 },
    { id: "T", label: "Thread sayısı T", min: 2, max: 8, value: 2, int: true },
    { id: "n", label: "Thread başına artırım n", min: 1, max: 200, value: 50, int: true, log: true },
    { id: "q", label: "Önalım olasılığı q (adım başına)", min: 0.001, max: 0.9, value: 0.05, log: true },
  ],
  seed: true,
  run(q, r) {
    const R = 1000, want = q.T * q.n, res = range(R).map(() => runOnce(q.T, q.n, q.q, q.mode, r));
    const okFrac = res.filter((v) => v === want).length / R, lost = res.reduce((a, v) => a + (want - v), 0) / R;
    const lo = Math.min(...res), hi = Math.max(...res), nb = Math.min(40, hi - lo + 1), bw = (hi - lo + 1) / nb;
    const hist = range(nb).map((b) => res.filter((v) => Math.min(nb - 1, Math.floor((v - lo) / bw)) === b).length);
    const qs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.9], curve = qs.map((qq) => range(200).map(() => runOnce(q.T, q.n, qq, q.mode, r)).reduce((a, v) => a + (want - v), 0) / 200);
    const E = enumerate(q.mode);
    const P = [`${R} rastgele çalıştırmada doğru sonuç (${want}) oranı ${fmt(okFrac, 3)}; ortalama kayıp güncelleme ${fmt(lost, 3)}, en küçük sonuç ${lo}.`];
    P.push(`Tam sayım (2 thread × 1 artırım): ${E.total} iç içe geçmenin ${E.ok}'si doğru sonucu (2) veriyor.`);
    if (q.mode === 0) P.push("C++'ta bu bir veri yarışıdır: tanımsız davranış; derleyici döngüyü yazmaçta tutup sonucu tek seferde yazabilir, o zaman kayıplar burada görülenden de büyük olur.");
    if (q.mode === 1) P.push("Veri yarışı yok (her erişim atomik), ama oku ile yaz arasında başka bir thread yazabildiği için yarış koşulu aynen sürüyor: kayıplar korumasız sürümle aynı dağılımda.");
    if (q.mode === 2) P.push("Oku–artır–yaz bölünmez: her sıralama aynı sonucu veriyor.");
    return {
      plots: [
        { title: "Sonuç dağılımı (1000 çalıştırma)", xlabel: "son sayaç değeri", ylabel: "sayı", side: true, size: { w: 300, h: 230 }, series: [{ name: "çalıştırma", x: range(nb).map((b) => lo + (b + 0.5) * bw - 0.5), y: hist, kind: "bar", c: 1 }], marks: [{ x: want, label: "doğru" }] },
        { title: "Kayıp güncelleme ve önalım olasılığı", xlabel: "q", ylabel: "ortalama kayıp", side: true, size: { w: 300, h: 230 }, series: [{ name: "kayıp", x: qs.map(Math.log10), y: curve, c: 0, dots: true }], xticks: qs.map((v) => [Math.log10(v), String(v)]) },
      ],
      stats: [{ label: "Doğru sonuç oranı", value: fmt(okFrac, 3) }, { label: "Ortalama kayıp güncelleme", value: fmt(lost, 3) }, { label: "Tam sayım: doğru / toplam", value: `${E.ok} / ${E.total}` }],
      note: P.join(" "),
    };
  },
};
