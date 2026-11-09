// M4.11 — Ders alıştırmasının (Géron) sentetik serileri: iki rastgele sinüs + gürültü. Naif tahmin, doğrusal otoregresif model
// (son k değer, en küçük kareler) ve çok adımlı tahminin iki yolu: yinelemeli (tek adımlık modeli kendi tahminiyle besle) ve
// doğrudan (her ufuk için ayrı doğrusal model). Ufka göre MSE.
import { fmt, range } from "./plot.js";
import { ridge } from "./ddescent.js";

export function series(r, n, noise) {
  const f1 = r(), f2 = r(), o1 = r(), o2 = r();
  return range(n).map((i) => { const t = i / (n - 1); /* Géron: np.linspace(0, 1, n) */ return 0.5 * Math.sin((t - o1) * (f1 * 10 + 10)) + 0.2 * Math.sin((t - o2) * (f2 * 20 + 20)) + noise * (r() - 0.5); });
}
export default {
  params: [
    { id: "k", label: "Doğrusal modelin gördüğü son değer sayısı k", min: 1, max: 50, value: 50, int: true },
    { id: "H", label: "Tahmin ufku H (adım)", min: 1, max: 20, value: 10, int: true },
    { id: "noise", label: "Gürültü genliği (Géron: 0.1)", min: 0, max: 1, step: 0.05, value: 0.1 },
    { id: "N", label: "Eğitim serisi sayısı", min: 20, max: 2000, value: 500, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const T = 50, H = Math.round(q.H), k = Math.round(q.k), mk = () => series(r, T + H, q.noise);
    const tr = range(Math.round(q.N)).map(mk), te = range(400).map(mk), feat = (s, end) => [...s.slice(end - k, end), 1];
    const w1 = ridge(tr.map((s) => feat(s, T)), tr.map((s) => s[T]), 1e-6);
    const wd = range(H).map((h) => ridge(tr.map((s) => feat(s, T)), tr.map((s) => s[T + h]), 1e-6));
    const dot = (w, x) => w.reduce((a, v, i) => a + v * x[i], 0);
    const mse = { naive: range(H).map(() => 0), iter: range(H).map(() => 0), direct: range(H).map(() => 0) };
    const ex = te[0], exIter = [], exDir = [];
    te.forEach((s, j) => {
      const buf = s.slice(0, T);
      for (let h = 0; h < H; h++) {
        const it = dot(w1, [...buf.slice(buf.length - k), 1]); buf.push(it);
        const di = dot(wd[h], feat(s, T)), y = s[T + h];
        mse.naive[h] += (s[T - 1] - y) ** 2 / te.length; mse.iter[h] += (it - y) ** 2 / te.length; mse.direct[h] += (di - y) ** 2 / te.length;
        if (j === 0) { exIter.push(it); exDir.push(di); }
      }
    });
    const hs = range(H).map((h) => h + 1), avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    return {
      plots: [
        { title: "Bir test serisi ve tahminler", xlabel: "zaman adımı", ylabel: "değer", series: [{ name: "gözlenen (girdi)", x: range(T), y: ex.slice(0, T), c: 4 }, { name: "gerçek gelecek", x: range(T - 1, T + H), y: ex.slice(T - 1), c: 1 }, { name: "naif", x: [T - 1, T + H - 1], y: [ex[T - 1], ex[T - 1]], c: 5, dash: true }, { name: "yinelemeli", x: [T - 1, ...range(H).map((h) => T + h)], y: [ex[T - 1], ...exIter], c: 2 }, { name: "doğrudan", x: [T - 1, ...range(H).map((h) => T + h)], y: [ex[T - 1], ...exDir], c: 0 }], marks: [{ x: T - 0.5, label: "şimdi" }] },
        { title: "Test MSE'si, ufka göre (400 seri, log)", xlabel: "ufuk h", ylabel: "MSE", ylog: true, series: [{ name: "naif (son değer)", x: hs, y: mse.naive.map((v) => Math.max(v, 1e-6)), c: 5, dots: true }, { name: "doğrusal, yinelemeli", x: hs, y: mse.iter.map((v) => Math.max(v, 1e-6)), c: 2, dots: true }, { name: "doğrusal, doğrudan", x: hs, y: mse.direct.map((v) => Math.max(v, 1e-6)), c: 0, dots: true }], marks: q.noise > 0 ? [{ y: (q.noise * q.noise) / 12, label: "gürültü tabanı a²/12" }] : [] },
      ],
      stats: [
        { label: "1 adım MSE: naif / doğrusal", value: `${fmt(mse.naive[0], 3)} / ${fmt(mse.iter[0], 3)}` },
        { label: `${H} adım ortalama MSE: naif`, value: avg(mse.naive) },
        { label: `${H} adım ortalama MSE: yinelemeli`, value: avg(mse.iter) },
        { label: `${H} adım ortalama MSE: doğrudan`, value: avg(mse.direct) },
        { label: "Doğrusal model parametresi", value: `${k + 1} (k ağırlık + bias)` },
      ],
      note: `Naif tahmin ("yarın bugün gibi") her zaman ilk kontrol edilecek temel çizgidir; bu seriler yavaş değiştiği için bir adımda zor geçilir, ufuk uzadıkça çok kötüleşir. Doğrusal model ${k} değerin ağırlıklı toplamı: bir AR(${k}) modeli. <b>Yinelemeli</b> tahminde model kendi çıktısını girdi olarak kullanır; hatalar birikir. <b>Doğrudan</b> tahminde her ufuk ayrı eğitilir; hata birikmez ama ${H} ayrı model gerekir (dizi→vektör RNN, H çıktıyı tek ağla birlikte öğrenerek bunun ortasında durur). ${q.noise > 0 ? `Gürültü düzgün dağılımlı (genlik ${fmt(q.noise, 2)}): öngörülemez kısmının varyansı a²/12 = ${fmt((q.noise * q.noise) / 12, 3)}; hiçbir model bunun altına inemez.` : "Gürültü yok: tek bir serideki iki sinüsün toplamı 4 gecikmeli doğrusal bir yinelemeyle (AR(4)) kesin üretilir; ama katsayıları frekanslara bağlıdır ve frekanslar seriden seriye değişir. Tüm serilere tek bir doğrusal model uydurulduğu için hata sıfır olmaz; k büyüdükçe azalır."}`,
    };
  },
};
