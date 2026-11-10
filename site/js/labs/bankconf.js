// M10.4 — Paylaşımlı bellekte bank çakışmaları: 32 bank, bank genişliği 4 bayt (Maxwell ve sonrası). Bir warp'ın 32 thread'inin
// eriştiği adresler bank'lara eşlenir; aynı bankta FARKLI 4 baytlık sözcüklere erişen thread'ler sıraya girer (n yollu çakışma = en
// kalabalık bankta farklı sözcük sayısı). Aynı sözcüğe erişen thread'ler yayın (broadcast) ile tek seferde hizmet alır. Desenler:
// adımlı 1-B erişim A[s·tid], 2-B karo sütun erişimi tile[tid][c] (satır uzunluğu W, dolgu P), yayın, 8/16 baytlık öğeler.
import { fmt, range } from "./plot.js";

export function analyze(addrs) { // addrs: bayt adresleri (32 thread)
  const banks = range(32).map(() => new Set());
  for (const a of addrs) { const word = Math.floor(a / 4); banks[word % 32].add(word); }
  const per = banks.map((s) => s.size), way = Math.max(...per);
  return { per, way };
}
const PAT = ["1-B adımlı: A[s · tid] (4 bayt)", "2-B karo, satır okuma: tile[r][tid]", "2-B karo, sütun okuma: tile[tid][c]", "yayın: hepsi A[0]", "8 baytlık öğe: A[s · tid] (double)", "16 baytlık öğe: A[tid] (float4)"];
export default {
  params: [
    { id: "pat", label: "Erişim deseni", options: PAT, value: 2 },
    { id: "s", label: "Adım s (öğe)", min: 1, max: 33, value: 2, int: true, on: { pat: [0, 4] } },
    { id: "W", label: "Karo satır uzunluğu W (float)", min: 8, max: 64, value: 32, int: true, on: { pat: [1, 2] } },
    { id: "P", label: "Dolgu (satır başına ek float)", min: 0, max: 4, value: 0, int: true, on: { pat: [1, 2] } },
  ],
  run(q) {
    const tid = range(32), row = q.W + q.P;
    const addrs = q.pat === 0 ? tid.map((t) => 4 * q.s * t) : q.pat === 1 ? tid.map((t) => 4 * (3 * row + t)) : q.pat === 2 ? tid.map((t) => 4 * (t * row + 5)) : q.pat === 3 ? tid.map(() => 0) : q.pat === 4 ? tid.map((t) => 8 * q.s * t) : tid.map((t) => 16 * t);
    // 8 ve 16 baytlık erişimlerde her thread 2 ya da 4 sözcük okur; donanım isteği yarım/çeyrek warp'lara böler (Maxwell+): sözcük bazında say
    let res;
    if (q.pat === 4 || q.pat === 5) {
      const wpt = q.pat === 4 ? 2 : 4, group = 32 / wpt; let total = 0; const per = range(32).map(() => 0);
      for (let g = 0; g < wpt; g++) { const words = []; for (let t = g * group; t < (g + 1) * group; t++) for (let k = 0; k < wpt; k++) words.push(addrs[t] + 4 * k); const A = analyze(words); total += A.way; A.per.forEach((v, b) => (per[b] = Math.max(per[b], v))); }
      res = { per, way: total / wpt, passes: total };
    } else { const A = analyze(addrs); res = { ...A, passes: A.way }; }
    const ideal = q.pat === 4 ? 2 : q.pat === 5 ? 4 : 1;
    const P = [`${res.passes} geçiş (çakışmasız ideal ${ideal}); en kalabalık bankta ${Math.max(...res.per)} farklı sözcük${res.passes > ideal ? ` → ${res.passes / ideal} yollu çakışma, bant genişliği ${res.passes / ideal} kat azalır` : ": çakışma yok"}.`];
    if (q.pat === 0) { const g = (function gcd(a, b) { return b ? gcd(b, a % b) : a; })(q.s, 32); P.push(`Adım s ile çakışma derecesi gcd(s, 32) = ${g} (tek adımlarda her zaman 1).`); }
    if (q.pat === 2) P.push(q.P === 0 && row % 32 === 0 ? `Satır uzunluğu ${row} = 32'nin katı: bir sütundaki 32 öğe aynı bankta. [${q.W}][${q.W + 1}] dolgusu bankları kaydırır.` : `Satır uzunluğu ${row}: sütun öğeleri bank'lara ${32 / (function gcd(a, b) { return b ? gcd(b, a % b) : a; })(row, 32)} farklı bank üzerinden dağılıyor.`);
    if (q.pat === 3) P.push("Bütün thread'ler aynı sözcüğü okuyor: yayın, tek geçiş.");
    return {
      plots: [{ title: "Bank başına farklı sözcük (warp isteği)", xlabel: "bank", ylabel: "sözcük", wide: true, size: { w: 620, h: 200 }, ylim: [0, Math.max(2, Math.max(...res.per)) * 1.1], series: [{ name: "farklı sözcük", x: range(32), y: res.per, kind: "bar", c: res.passes > ideal ? 1 : 0 }] }],
      stats: [{ label: "Geçiş sayısı / ideal", value: `${res.passes} / ${ideal}` }, { label: "Bant genişliği verimi", value: `${fmt((100 * ideal) / res.passes, 3)}%` }],
      note: P.join(" "),
    };
  },
};
