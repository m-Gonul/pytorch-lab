// M11.6 — Deneyim tekrarının neden işe yaradığını gözetimli bir benzerle gösterir. Durum x ∈ [0, 1] yavaşça ileri geri kayar
// (x_t = ½ + ½ sin(2π t / P)); her adımda f(x) = sin(2πx) + 0.5 cos(6πx) hedefi gürültüyle (σ = 0.1) gözlenir. Model: 20 Gauss RBF
// özelliğinin doğrusal birleşimi. Çevrimiçi SGD her adımda yalnız son örnekle güncellenir (ardışık örnekler güçlü ilişkili); deneyim
// tekrarı son N örneği bir tamponda tutar ve her adımda tampondan düzgün çekilen B örneklik bir mini yığınla güncellenir. Hata,
// bütün [0, 1] aralığında (yalnız o an ziyaret edilen bölgede değil) ölçülür. Özellikler yerel (RBF) ya da küresel (Fourier) seçilebilir:
// küresel özelliklerde bir bölgeye uymak bütün ağırlıkları değiştirir ve uzak bölgeler unutulur.
import { fmt, range } from "./plot.js";

const K = 20, CS = range(K).map((i) => i / (K - 1)), SW = 0.06;
const rbf = (x) => CS.map((c) => Math.exp(-((x - c) ** 2) / (2 * SW * SW)));
const four = (x) => range(K).map((k) => (k === 0 ? 1 : k % 2 ? Math.sin(Math.PI * (k + 1) * x) : Math.cos(Math.PI * k * x)) / Math.sqrt(K / 4));
let phi = rbf;
const f = (x) => Math.sin(2 * Math.PI * x) + 0.5 * Math.cos(6 * Math.PI * x);
const GRID = range(101).map((i) => i / 100);
let PG = GRID.map(phi);
const err = (w) => Math.sqrt(GRID.reduce((a, x, i) => a + (PG[i].reduce((s, v, k) => s + v * w[k], 0) - f(x)) ** 2, 0) / GRID.length);
export default {
  params: [
    { id: "feat", label: "Özellikler", options: ["20 Gauss RBF (yerel)", "20 Fourier terimi (küresel)"], value: 1 },
    { id: "P", label: "Bir gidiş-geliş süresi P (adım)", min: 100, max: 4000, step: 100, value: 2000, int: true },
    { id: "lr", label: "Öğrenme oranı", min: 0.01, max: 0.5, step: 0.01, value: 0.1 },
    { id: "N", label: "Tampon boyu N", min: 10, max: 5000, step: 10, value: 2000, int: true },
    { id: "B", label: "Mini yığın B", min: 1, max: 64, value: 16, int: true },
    { id: "T", label: "Adım sayısı", min: 1000, max: 10000, step: 500, value: 6000, int: true },
  ],
  seed: true,
  run(q, r) {
    phi = q.feat === 0 ? rbf : four; PG = GRID.map(phi);
    let w1 = Array(K).fill(0), w2 = Array(K).fill(0); const buf = [], e1 = [], e2 = [], xs = [], pos = [];
    const sgd = (w, batch) => { const g = Array(K).fill(0); for (const [p, y] of batch) { const d = p.reduce((s, v, k) => s + v * w[k], 0) - y; for (let k = 0; k < K; k++) g[k] += (d * p[k]) / batch.length; } return w.map((v, k) => v - q.lr * g[k]); };
    for (let t = 0; t < q.T; t++) {
      const x = 0.5 + 0.5 * Math.sin((2 * Math.PI * t) / q.P), y = f(x) + 0.1 * r.normal(), p = phi(x);
      w1 = sgd(w1, [[p, y]]);
      buf.push([p, y]); if (buf.length > q.N) buf.shift();
      w2 = sgd(w2, range(q.B).map(() => buf[Math.floor(r() * buf.length)]));
      if (t % 25 === 0) { xs.push(t); e1.push(err(w1)); e2.push(err(w2)); pos.push(x); }
    }
    const tail = (a) => a.slice(-Math.floor(a.length / 3)), mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const P = [`Son üçte birdeki ortalama hata (bütün aralıkta, RMS): çevrimiçi SGD ${fmt(mean(tail(e1)), 3)}, deneyim tekrarı ${fmt(mean(tail(e2)), 3)}; çevrimiçinin en büyüğü ${fmt(Math.max(...tail(e1)), 3)}.`];
    const m1 = mean(tail(e1)), m2 = mean(tail(e2));
    if (q.P > 300 && m1 > 3 * m2 && q.feat === 1) P.push("Çevrimiçi öğrenici o an bulunduğu bölgeye büyük adımlarla uyar; küresel özelliklerde bu, uzak bölgelerin değerlerini de bozar ve hata durum gidip geldikçe dalgalanır. Tampondan düzgün örneklemek ardışık örneklerin ilişkisini kırar: her güncelleme yakın geçmişin bütün dağılımını görür.");
    if (q.P > 300 && m1 > 3 * m2 && q.feat === 0) P.push("Yerel özelliklerde bir bölgeye uymak uzak ağırlıkları neredeyse etkilemez; çevrimiçinin sorunu unutmaktan çok, aynı noktada yığılan ilişkili güncellemelerle yavaş ilerlemesidir. Tekrar aynı veriyi karışık sırayla kullanarak çok daha hızlı yakınsar.");
    if (q.P <= 300 || m1 <= 3 * m2) P.push(q.P <= 300 ? "Durum hızlı değişince ardışık örnekler zaten aralığın her yerinden gelir; çevrimiçi SGD de iyi çalışır." : "Bu öğrenme oranında çevrimiçi SGD de kararlı: ilişkili güncellemeler küçük adımlarla ortalanıyor. Tekrarın getirisi, daha büyük adımlarla aynı kararlılık ve her örneğin birçok kez kullanılmasıdır (veri verimliliği).");
    if (q.N < q.P / 4) P.push(`Tampon (${q.N} örnek) bir gidiş-gelişin küçük bir kısmını tutuyor; bu ayarda tekrarın hatası ${fmt(m2, 3)}.`);
    return {
      plots: [
        { title: "Bütün aralıkta hata (RMS)", xlabel: "adım", ylabel: "RMS", wide: true, size: { w: 620, h: 220 }, ylim: [0, 1.2], series: [{ name: "çevrimiçi SGD", x: xs, y: e1, c: 2 }, { name: "deneyim tekrarı", x: xs, y: e2, c: 0 }] },
        { title: "Durumun konumu x_t (ilişkili ziyaret)", xlabel: "adım", ylabel: "x", wide: true, size: { w: 620, h: 120 }, ylim: [0, 1], series: [{ name: "x_t", x: xs, y: pos, c: 4 }] },
      ],
      stats: [{ label: "Son üçte bir RMS (çevrimiçi / tekrar)", value: `${fmt(mean(tail(e1)), 3)} / ${fmt(mean(tail(e2)), 3)}` }, { label: "Çevrimiçinin en büyük hatası", value: fmt(Math.max(...tail(e1)), 3) }],
      note: P.join(" "),
    };
  },
};
