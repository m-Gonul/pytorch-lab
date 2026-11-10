// M10.2 — Üretici–tüketici: P üretici, C tüketici, B yuvalı dairesel tampon; toplam K = 200 öğe. Üretim ve tüketim süreleri üstel
// dağılımlı (ortalamalar tp, tc). Bekleme: bloklayan semafor (slotsAvail, resAvail) ya da meşgul bekleme (boş while döngüsü, CPU harcar).
// Sonlandırma: zehirli hap (üreticiler bitince C adet özel öğe tampona konur) ya da paylaşılan bayrak (tüketiciler döngü başında bakar;
// semaforda bloklu bir tüketici bayrağı hiç görmez). Zaman adımlı simülasyon (dt = 0.02).
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "P", label: "Üretici sayısı P", min: 1, max: 8, value: 2, int: true },
    { id: "C", label: "Tüketici sayısı C", min: 1, max: 8, value: 3, int: true },
    { id: "B", label: "Tampon boyu B", min: 1, max: 64, value: 8, int: true, log: true },
    { id: "tp", label: "Ortalama üretim süresi", min: 0.1, max: 4, step: 0.1, value: 1 },
    { id: "tc", label: "Ortalama tüketim süresi", min: 0.1, max: 4, step: 0.1, value: 0.5 },
    { id: "wait", label: "Bekleme", options: ["bloklayan semafor", "meşgul bekleme (spin)"], value: 0 },
    { id: "term", label: "Sonlandırma", options: ["zehirli hap (C adet)", "paylaşılan bayrak"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const K = 200, dt = 0.02, ex = (m) => -m * Math.log(1 - r());
    let items = 0, pills = 0, produced = 0, consumed = 0, t = 0, flag = false, pillsToPut = 0, spin = 0, blockedT = 0;
    const prod = range(q.P).map(() => ({ st: "make", rem: ex(q.tp), done: false })), cons = range(q.C).map(() => ({ st: "get", rem: 0, done: false }));
    const ts = [], occ = [], pb = [], cb = [];
    let k = 0;
    while (t < 2000) {
      // üreticiler
      for (const p of prod) {
        if (p.done) continue;
        if (p.st === "make") { p.rem -= dt; if (p.rem <= 0) p.st = "put"; }
        if (p.st === "put") {
          if (items + pills < q.B) { items++; produced++; if (produced >= K) { for (const x of prod) x.done = true; if (q.term === 0) pillsToPut = q.C; } else { p.st = "make"; p.rem = ex(q.tp); } }
          else if (q.wait === 1) spin += dt; else blockedT += dt;
        }
      }
      if (pillsToPut > 0 && items + pills < q.B) { pills++; pillsToPut--; }
      if (q.term === 1 && produced >= K && items === 0) flag = true;
      // tüketiciler
      for (const c of cons) {
        if (c.done) continue;
        if (c.st === "use") { c.rem -= dt; if (c.rem <= 0) { c.st = "get"; consumed++; } continue; }
        // döngü başı: bayrağı yalnız bloklu değilken görebilir
        if (q.term === 1 && flag && (q.wait === 1 || !c.blocked)) { c.done = true; continue; }
        if (items > 0) { items--; c.blocked = false; c.st = "use"; c.rem = ex(q.tc); }
        else if (pills > 0 && items === 0) { pills--; c.done = true; }
        else { if (q.wait === 1) spin += dt; else { c.blocked = true; blockedT += dt; } }
      }
      t += dt;
      if (k++ % 10 === 0) { ts.push(t); occ.push(items + pills); pb.push(prod.filter((p) => !p.done && p.st === "put").length); cb.push(cons.filter((c) => !c.done && c.st === "get").length); }
      const allProdDone = prod.every((p) => p.done) && pillsToPut === 0, consLive = cons.filter((c) => !c.done);
      if (allProdDone && consLive.every((c) => c.st === "get") && consumed >= K && (consLive.length === 0 || (q.term === 1 && q.wait === 0 && consLive.every((c) => c.blocked)))) break;
    }
    const hung = cons.filter((c) => !c.done).length, thr = consumed / t;
    const bound = Math.min(q.P / q.tp, q.C / q.tc);
    const P = [`${consumed}/${K} öğe tüketildi, süre ${fmt(t, 3)}; verim ${fmt(thr, 3)} öğe/birim (beklenen sınır min(P/tp, C/tc) = ${fmt(bound, 3)}; süreler rastgele olduğu için tek bir çalıştırma bunu biraz aşabilir).`];
    P.push(q.wait === 0 ? `Bloklu bekleme toplamı ${fmt(blockedT, 3)} thread·birim: bu sürede CPU başka işe verilebilir.` : `Meşgul beklemede boşa harcanan CPU ${fmt(spin, 3)} thread·birim (toplam sürenin ${fmt(spin / t, 3)} katı çekirdek).`);
    if (hung) P.push(`${hung} tüketici hiç sonlanmadı: semaforda bloklu beklerken bayrağı göremiyor (zehirli hap ya da bayrak + bekleyenleri uyandırma gerekir).`);
    else P.push(q.term === 0 ? `Her tüketici bir zehirli hap alıp çıktı (${q.C} hap).` : "Bütün tüketiciler bayrağı gördü.");
    return {
      plots: [
        { title: "Tampon doluluğu", xlabel: "zaman", ylabel: "öğe", wide: true, size: { w: 620, h: 180 }, ylim: [0, q.B], series: [{ name: "doluluk", x: ts, y: occ, c: 0 }] },
        { title: "Bekleyen thread'ler", xlabel: "zaman", ylabel: "thread", wide: true, size: { w: 620, h: 160 }, series: [{ name: "yer bekleyen üretici", x: ts, y: pb, c: 1 }, { name: "öğe bekleyen tüketici", x: ts, y: cb, c: 2 }] },
      ],
      stats: [{ label: "Verim / beklenen sınır", value: `${fmt(thr, 3)} / ${fmt(bound, 3)}` }, { label: "Süre", value: fmt(t, 3) }, { label: q.wait === 0 ? "Bloklu bekleme (thread·birim)" : "Boşa harcanan CPU (thread·birim)", value: fmt(q.wait === 0 ? blockedT : spin, 3) }, { label: "Sonlanmayan tüketici", value: hung }],
      note: P.join(" "),
    };
  },
};
