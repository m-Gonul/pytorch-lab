// M9.5 — Gürültü olasılık yoğunlukları ve parametre kestirimi. Üç gri seviyeli (40, 128, 200) bir test deseni
// g = f + η ile bozulur (tuz-biberde piksel 0 ya da 255 ile değiştirilir), 8 bite kırpılır. Düz bir şeritte (seviye 128,
// kırpılmadan) η'nin örnek ortalaması ve varyansından moment yöntemiyle parametreler kestirilir ve gerçek PDF ile karşılaştırılır.
import { fmt, range } from "./plot.js";

const TYPES = ["Gauss", "Rayleigh", "Erlang (Gamma)", "Üstel", "Düzgün", "Tuz-biber"];
const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
export function pdf(t, P, z) {
  if (t === 0) return Math.exp(-0.5 * ((z - P.mu) / P.sig) ** 2) / (Math.sqrt(2 * Math.PI) * P.sig);
  if (t === 1) return z < P.a ? 0 : (2 / P.b) * (z - P.a) * Math.exp(-((z - P.a) ** 2) / P.b);
  if (t === 2) return z < 0 ? 0 : (P.a ** P.b * z ** (P.b - 1) / fact(P.b - 1)) * Math.exp(-P.a * z);
  if (t === 3) return z < 0 ? 0 : P.a * Math.exp(-P.a * z);
  if (t === 4) return z >= P.a && z <= P.b ? 1 / (P.b - P.a) : 0;
  return 0;
}
function sample(t, P, r) {
  if (t === 0) return P.mu + P.sig * r.normal();
  if (t === 1) return P.a + Math.sqrt(-P.b * Math.log(1 - r()));
  if (t === 2) { let s = 0; for (let i = 0; i < P.b; i++) s -= Math.log(1 - r()) / P.a; return s; }
  if (t === 3) return -Math.log(1 - r()) / P.a;
  return P.a + (P.b - P.a) * r();
}
export function estimate(t, m, v) {
  const s = Math.sqrt(v);
  if (t === 0) return { mu: m, sig: s };
  if (t === 1) { const b = (4 * v) / (4 - Math.PI); return { a: m - Math.sqrt((Math.PI * b) / 4), b }; }
  if (t === 2) return { a: m / v, b: Math.max(1, Math.round((m * m) / v)) };
  if (t === 3) return { a: 1 / m };
  return { a: m - Math.sqrt(3) * s, b: m + Math.sqrt(3) * s };
}
const show = (t, P) => t === 0 ? `μ = ${fmt(P.mu, 3)}, σ = ${fmt(P.sig, 3)}` : t === 3 ? `a = ${fmt(P.a, 3)}` : `a = ${fmt(P.a, 3)}, b = ${fmt(P.b, 3)}`;

export default {
  params: [
    { id: "t", label: "Gürültü türü", options: TYPES, value: 1 },
    { id: "mu", label: "Gauss ortalaması z̄", min: -30, max: 30, step: 1, value: 0, on: { t: [0] } },
    { id: "sig", label: "Gauss std σ", min: 2, max: 50, step: 1, value: 15, on: { t: [0] } },
    { id: "ra", label: "Rayleigh a (başlangıç)", min: 0, max: 40, step: 1, value: 0, on: { t: [1] } },
    { id: "rb", label: "Rayleigh b", min: 20, max: 4000, value: 800, log: true, on: { t: [1] } },
    { id: "ea", label: "Erlang a (oran)", min: 0.05, max: 1, step: 0.01, value: 0.2, on: { t: [2] } },
    { id: "eb", label: "Erlang b (şekil, tamsayı)", min: 1, max: 12, value: 4, int: true, on: { t: [2] } },
    { id: "xa", label: "Üstel a", min: 0.02, max: 0.5, step: 0.01, value: 0.07, on: { t: [3] } },
    { id: "ua", label: "Düzgün a (alt)", min: -60, max: 0, step: 1, value: -25, on: { t: [4] } },
    { id: "ub", label: "Düzgün b (üst)", min: 0, max: 60, step: 1, value: 25, on: { t: [4] } },
    { id: "ps", label: "Tuz olasılığı P_s", min: 0, max: 0.3, step: 0.01, value: 0.05, on: { t: [5] } },
    { id: "pp", label: "Biber olasılığı P_p", min: 0, max: 0.3, step: 0.01, value: 0.05, on: { t: [5] } },
    { id: "n", label: "Düz şeritteki örnek sayısı", min: 30, max: 20000, value: 2000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const t = q.t, P = [{ mu: q.mu, sig: q.sig }, { a: q.ra, b: q.rb }, { a: q.ea, b: q.eb }, { a: q.xa }, { a: q.ua, b: q.ub }, null][t];
    const levels = [40, 128, 200], NP = 12000, H = new Array(256).fill(0);
    let clip = 0;
    for (let i = 0; i < NP; i++) {
      const f = levels[i % 3];
      let g;
      if (t === 5) { const u = r(); g = u < q.ps ? 255 : u < q.ps + q.pp ? 0 : f; }
      else { g = Math.round(f + sample(t, P, r)); if (g < 0 || g > 255) clip++; g = Math.max(0, Math.min(255, g)); }
      H[g]++;
    }
    const plots = [{ title: "Gürültülü test deseninin histogramı (seviyeler 40, 128, 200)", xlabel: "yoğunluk", ylabel: "piksel sayısı", wide: true, size: { w: 620, h: 170 }, series: [{ name: "histogram", x: range(256), y: H, kind: "bar", c: 0 }], xlim: [0, 255], marks: levels.map((l) => ({ x: l, label: String(l) })) }];
    const stats = [], P2 = [];
    if (t === 5) {
      let s = 0, p = 0; for (let i = 0; i < q.n; i++) { const u = r(); if (u < q.ps) s++; else if (u < q.ps + q.pp) p++; }
      const V = 128, m = 255 * q.ps + V * (1 - q.ps - q.pp), v = (0 - m) ** 2 * q.pp + (V - m) ** 2 * (1 - q.ps - q.pp) + (255 - m) ** 2 * q.ps;
      stats.push({ label: "Gerçek P_s, P_p", value: `${q.ps}, ${q.pp}` }, { label: "Şeritten kestirim", value: `${fmt(s / q.n, 3)}, ${fmt(p / q.n, 3)}` }, { label: "Gürültü yoğunluğu P = P_s + P_p", value: fmt(q.ps + q.pp, 3) }, { label: "Şerit (V = 128) ortalaması, varyansı", value: `${fmt(m, 4)}, ${fmt(v, 4)}` });
      P2.push(`Tuz-biber gürültüsü toplamsal değildir: pikseller ${q.ps > 0 && q.pp > 0 ? "0 ve 255 ile" : q.ps > 0 ? "yalnız 255 ile (tek kutuplu)" : q.pp > 0 ? "yalnız 0 ile (tek kutuplu)" : "hiç"} değiştirilir; histogramda uçlarda iki çubuk belirir. Parametreleri ortalama ve varyans değil, uçlardaki piksellerin oranı verir: ${q.n} örnekte P_s kestirimi ${fmt(s / q.n, 3)}. Ortalama ve varyans formülü V'ye bağlıdır (burada V = 128).`);
      plots.push({ title: "Şeridin değer dağılımı", xlabel: "yoğunluk", ylabel: "oran", series: [{ name: "gerçek", x: [0, 128, 255], y: [q.pp, 1 - q.ps - q.pp, q.ps], kind: "bar", c: 4 }, { name: "kestirim", x: [8, 136, 247], y: [p / q.n, 1 - (s + p) / q.n, s / q.n], kind: "bar", c: 0 }], xlim: [-10, 265] });
    } else {
      const z = range(q.n).map(() => sample(t, P, r)), m = z.reduce((a, b) => a + b, 0) / q.n, v = z.reduce((a, b) => a + (b - m) ** 2, 0) / (q.n - 1), E = estimate(t, m, v);
      const lo = Math.min(...z), hi = Math.max(...z), nb = 40, w = (hi - lo) / nb || 1, cnt = new Array(nb).fill(0);
      z.forEach((x) => cnt[Math.min(nb - 1, Math.floor((x - lo) / w))]++);
      const xs = range(201).map((i) => lo - 0.1 * (hi - lo) + (1.2 * (hi - lo) * i) / 200);
      plots.push({ title: "Düz şeritteki η: histogram, gerçek ve kestirilen PDF", xlabel: "η", ylabel: "yoğunluk", series: [{ name: "histogram", x: range(nb).map((i) => lo + (i + 0.5) * w), y: cnt.map((c) => c / (q.n * w)), kind: "bar", c: 4 }, { name: "gerçek PDF", x: xs, y: xs.map((x) => pdf(t, P, x)), c: 0 }, { name: "kestirilen PDF", x: xs, y: xs.map((x) => pdf(t, E, x)), c: 1, dash: true }] });
      const th = [[q.mu, q.sig ** 2], [q.ra + Math.sqrt((Math.PI * q.rb) / 4), (q.rb * (4 - Math.PI)) / 4], [q.eb / q.ea, q.eb / q.ea ** 2], [1 / q.xa, 1 / q.xa ** 2], [(q.ua + q.ub) / 2, (q.ub - q.ua) ** 2 / 12]][t];
      stats.push({ label: "Gerçek parametreler", value: show(t, P) }, { label: "Moment kestirimi", value: show(t, E) }, { label: "Örnek z̄, σ² (kuram)", value: `${fmt(m, 3)}, ${fmt(v, 4)} (${fmt(th[0], 3)}, ${fmt(th[1], 4)})` }, { label: "Kırpılan piksel oranı (desen)", value: fmt(clip / NP, 3) });
      P2.push(`${q.n} örnekten moment kestirimi: ${show(t, E)} (gerçek: ${show(t, P)}). Örnek sayısı küçüldükçe kestirim tohumdan tohuma daha çok dağılır (Gauss için örnek varyansının standart hatası ≈ σ²√(2/n)).`);
      P2.push(["Gauss: elektronik devre ve sensör gürültüsü (kötü aydınlatma, yüksek sıcaklık); ortalama ve varyans iki parametreyi doğrudan verir.", "Rayleigh: sağa çarpık; a'dan başlar. Uzaklık (range) görüntülemede görülür.", "Erlang: oranı a olan b tane üstel değişkenin toplamı; b = 1 üsteldir, b büyüdükçe Gauss'a yaklaşır. Lazer görüntülemede.", "Üstel: Erlang'ın b = 1 hâli; tek parametre ortalamanın tersidir. Lazer görüntülemede.", "Düzgün: pratikte en az tanımlayıcı olan, ama rastgele sayı üreteçlerinin temeli.", ""][t]);
      if (clip / NP > 0.01) P2.push(`<b>Dikkat:</b> desen piksellerinin ${fmt((100 * clip) / NP, 3)}%'i 8 bit aralığın dışına taştı ve kırpıldı; histogramın uçlarında yapay yığılmalar bunlardır (şerit kırpılmadan ölçülüyor).`);
    }
    return { plots, stats, note: P2.join(" ") };
  },
  validate(q) { return q.t === 4 && q.ua >= q.ub ? [{ level: "error", msg: "Düzgün dağılımda a < b olmalı." }] : q.t === 5 && q.ps + q.pp > 1 ? [{ level: "error", msg: "P_s + P_p ≤ 1 olmalı." }] : []; },
};
