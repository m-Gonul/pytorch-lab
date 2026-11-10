// M10.2 — Kilitlenme, livelock ve açlık: N thread (yemek yiyen filozoflar) halka üzerinde; thread i, i ve (i + 1) mod N kilitlerini
// birlikte tutarak "yemek" yer. Stratejiler: (0) önce sol, sonra sağ — döngüsel bekleme mümkün; (1) küresel sıra (küçük indeksli kilit
// önce) — döngüsel bekleme kırılır; (2) önce sol, sağı try_lock; olmazsa solu bırak, sabit süre bekle, yeniden dene; (3) aynısı, rastgele
// geri çekilme süresiyle. Düşünme ve yeme süreleri ortalama etrafında ±sapma ile düzgün dağılımlı. Zaman adımlı simülasyon, 2000 adım.
import { fmt, range } from "./plot.js";

export default {
  params: [
    { id: "strat", label: "Kilit stratejisi", options: ["önce sol, sonra sağ (bekle)", "küresel kilit sırası", "try_lock + sabit geri çekilme", "try_lock + rastgele geri çekilme"], value: 0 },
    { id: "N", label: "Thread (filozof) sayısı N", min: 2, max: 10, value: 5, int: true },
    { id: "jit", label: "Süre sapması (0: tam eşzamanlı)", min: 0, max: 1, step: 0.05, value: 0.5 },
    { id: "think", label: "Ortalama düşünme süresi (adım)", min: 1, max: 40, value: 10, int: true },
    { id: "eat", label: "Ortalama yeme süresi (adım)", min: 1, max: 40, value: 10, int: true },
  ],
  seed: true,
  run(q, r) {
    const N = q.N, TT = 2000, owner = new Int32Array(N).fill(-1), dur = (m) => Math.max(1, Math.round(m * (1 + q.jit * (2 * r() - 1))));
    const th = range(N).map(() => ({ st: "think", rem: dur(q.think), meals: 0, tries: 0, holdL: false })); let deadAt = -1, wasted = 0;
    const first = (i) => (q.strat === 1 ? Math.min(i, (i + 1) % N) : i), second = (i) => (q.strat === 1 ? Math.max(i, (i + 1) % N) : (i + 1) % N);
    const mealsT = range(N).map(() => []);
    for (let t = 0; t < TT; t++) {
      let progressed = false;
      // faz A: zamanlayıcılar; yemeği biten kilitlerini bırakır
      for (let i = 0; i < N; i++) {
        const s = th[i];
        if (s.st === "think") { if (--s.rem <= 0) s.st = "want1"; progressed = true; }
        else if (s.st === "eat") { if (--s.rem <= 0) { owner[first(i)] = -1; owner[second(i)] = -1; s.meals++; mealsT[i].push(t); s.st = "think"; s.rem = dur(q.think); } progressed = true; }
        else if (s.st === "backoff") { wasted++; if (--s.rem <= 0) s.st = "want1"; }
      }
      // faz B: bütün istekler adım başındaki kilit durumuna göre aynı anda; çakışmada rastgele kazanan
      const req = new Map();
      for (let i = 0; i < N; i++) { const s = th[i]; if (s.st === "want1" || s.st === "want2") { const L = s.st === "want1" ? first(i) : second(i); if (!req.has(L)) req.set(L, []); req.get(L).push(i); } }
      const granted = new Set();
      for (const [L, rs] of req) if (owner[L] === -1) { const w = rs[Math.floor(r() * rs.length)]; owner[L] = w; granted.add(w); }
      for (let i = 0; i < N; i++) {
        const s = th[i];
        if (s.st === "want1" && granted.has(i)) { s.st = "want2"; progressed = true; }
        else if (s.st === "want2") {
          if (granted.has(i)) { s.st = "eat"; s.rem = dur(q.eat); progressed = true; }
          else if (q.strat >= 2) { owner[first(i)] = -1; s.tries++; wasted++; s.st = "backoff"; s.rem = q.strat === 2 ? 3 : 1 + Math.floor(r() * 8); }
        }
      }
      if (!progressed && q.strat < 2 && th.every((s) => s.st === "want2")) { deadAt = t; break; }
    }
    const meals = th.map((s) => s.meals), tot = meals.reduce((a, b) => a + b, 0), minM = Math.min(...meals), maxM = Math.max(...meals);
    const ideal = Math.floor(N / 2) * (TT / (q.eat + 0.0001)) ; // en fazla ⌊N/2⌋ thread aynı anda yiyebilir
    const names = ["önce sol, sonra sağ", "küresel kilit sırası", "try_lock + sabit geri çekilme", "try_lock + rastgele geri çekilme"];
    const P = [`${names[q.strat]}: ${deadAt >= 0 ? `${deadAt}. adımda KİLİTLENME — her thread sol kilidini tutup sağı bekliyor (döngüsel bekleme).` : `${TT} adımda kilitlenme yok.`} Toplam ${tot} yemek; thread başına en az ${minM}, en çok ${maxM}.`];
    if (q.strat >= 2) P.push(`Geri çekilmede ve başarısız denemelerde harcanan adım: ${wasted} (${fmt(wasted / (N * TT), 3)} oranında thread zamanı).`);
    if (q.strat >= 2 && tot < 0.1 * N * (TT / (q.think + q.eat))) P.push("LIVELOCK: thread'ler kilitleri alıp bırakmayı eşzamanlı tekrarlıyor; kimse bloklu değil, CPU harcanıyor ama neredeyse hiç yemek yenmiyor. Rastgele geri çekilme simetriyi kırar.");
    if (deadAt < 0 && tot > 0 && minM < 0.5 * (tot / N)) P.push("Bir thread diğerlerinin yarısından az yiyebildi: açlık eğilimi (adil olmayan kilit sırası).");
    return {
      plots: [{ title: "Thread başına toplam yemek", xlabel: "thread", ylabel: "yemek", side: true, size: { w: 300, h: 220 }, ylim: [0, Math.max(1, maxM) * 1.1], xticks: range(N).map((i) => [i, String(i)]), series: [{ name: "yemek", x: range(N), y: meals, kind: "bar", c: 0 }] },
        { title: "Yemek zamanları", xlabel: "adım", ylabel: "thread", side: true, size: { w: 300, h: 220 }, xlim: [0, TT], ylim: [-0.5, N - 0.5], series: [{ name: "yemek bitti", x: mealsT.flat(), y: mealsT.flatMap((m, i) => m.map(() => i)), kind: "scatter", c: 2, r: 1.6 }] }],
      stats: [{ label: "Kilitlenme", value: deadAt >= 0 ? `${deadAt}. adımda` : "yok" }, { label: "Toplam yemek", value: tot }, { label: "En az / en çok (thread)", value: `${minM} / ${maxM}` }, ...(q.strat >= 2 ? [{ label: "Boşa harcanan adım", value: wasted }] : [])],
      note: P.join(" "),
    };
  },
};
