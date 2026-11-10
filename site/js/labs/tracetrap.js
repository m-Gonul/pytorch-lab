// M7.6 — İzleme (tracing) kontrol akışını dondurur. f(x) = 2x eğer koşul, değilse −x. Kayıt tabanlı izleyici örnek girdi x₀ ile
// çalışır ve yalnız alınan dalın işlemlerini kaydeder; kaydedilen graf sonra her girdiye aynı dalı uygular. Koşul türleri: veriye bağlı
// (Σx > 0), şekle bağlı (n > 4), sabit (Python bayrağı). Test girdileri: x ~ N(μ, 1)ⁿ, n ∈ [1, 8]. Yanlış çıktı oranı ölçülür.
import { fmt, range } from "./plot.js";

const KIND = ["veriye bağlı: Σx > 0", "şekle bağlı: n > 4", "sabit bayrak (her zaman doğru)"];
const cond = (k, x) => (k === 0 ? x.reduce((s, v) => s + v, 0) > 0 : k === 1 ? x.length > 4 : true);
export default {
  params: [
    { id: "kind", label: "Dal koşulu", options: KIND, value: 0 },
    { id: "x0", label: "İzlemedeki örnek girdinin ortalaması", min: -2, max: 2, step: 0.1, value: 1 },
    { id: "n0", label: "İzlemedeki örnek girdinin uzunluğu n₀", min: 1, max: 8, value: 2, int: true },
    { id: "mu", label: "Test girdilerinin ortalaması μ", min: -2, max: 2, step: 0.1, value: 0 },
    { id: "N", label: "Test girdisi sayısı", min: 100, max: 5000, value: 1000, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const x0 = range(q.n0).map(() => q.x0), branch = cond(q.kind, x0);  // izlemede alınan dal
    let wrong = 0, errSum = 0; const byN = range(8).map(() => [0, 0]), sums = [], ok = [];
    for (let t = 0; t < q.N; t++) {
      const n = 1 + Math.floor(r() * 8), x = range(n).map(() => q.mu + r.normal()), c = cond(q.kind, x), bad = c !== branch;
      if (bad) { wrong++; errSum += x.reduce((s, v) => s + Math.abs(3 * v), 0) / n; }
      byN[n - 1][0]++; if (bad) byN[n - 1][1]++;
      if (t < 400) { sums.push(x.reduce((s, v) => s + v, 0)); ok.push(bad ? 0 : 1); }
    }
    const P = [`İzleme, örnek girdide (ortalama ${q.x0}, n₀ = ${q.n0}) "${branch ? "2x" : "−x"}" dalını kaydetti. ${q.N} test girdisinin ${wrong} tanesinde (oran ${fmt(wrong / q.N, 3)}) kaydedilen graf yanlış dalı uyguladı; yanlışlarda ortalama eleman hatası ${wrong ? fmt(errSum / wrong, 3) : 0}.`];
    if (q.kind === 0) P.push("Veriye bağlı dal: izleyici Σx'in değerini somut bir sayı olarak gördü ve dalı seçti; grafa koşul hiç girmedi. torch.export bu durumda hata verir (veriye bağlı kontrol akışı); çözüm torch.cond ya da torch.where ile iki dalı da grafa koymaktır.");
    if (q.kind === 1) P.push("Şekle bağlı dal: export, şekli bir guard'a çevirir. Boyut dinamik işaretlenmişse (Dim) koşul sembolik kalır ve dışa aktarma ya iki dalı ayırır ya da hangi aralıkta geçerli olduğunu bir kısıta yazar; işaretlenmemişse örnek şekle özelleşir ve başka uzunluklarda reddedilir. Burada kayıt tabanlı izleyici yalnız n₀'ın tarafını öğrendi.");
    if (q.kind === 2) P.push("Sabit bayrak: dal her girdide aynı; izleme güvenlidir. Python düzeyindeki yapılandırma bayrakları (ör. eğitim/çıkarım modu) böyle dondurulabilir.");
    return {
      plots: [
        { title: "Test girdileri: Σx ve doğruluk", xlabel: "Σx", ylabel: "doğru (1) / yanlış (0)", ylim: [-0.1, 1.1], series: [{ name: "test girdisi", x: sums, y: ok.map((v, i) => v + 0.04 * ((i % 7) - 3) / 3), kind: "scatter", c: 0, r: 2 }], marks: [{ x: 0, label: "Σx = 0" }] },
        { title: "Uzunluğa göre yanlış oranı", xlabel: "n", ylabel: "yanlış oranı", ylim: [0, 1.05], yzero: true, series: [{ name: "yanlış", x: range(8).map((i) => i + 1), y: byN.map(([a, b]) => (a ? b / a : 0)), kind: "bar", c: 1 }], xlim: [0.4, 8.6] },
      ],
      stats: [{ label: "İzlenen dal", value: branch ? "2x" : "−x" }, { label: "Yanlış oranı", value: wrong / q.N }],
      note: P.join(" "),
    };
  },
};
