// M3.9 — Teşhis: öğrenme eğrileri (hata vs veri) ve kapasite eğrisi (hata vs model genişliği).
// Model: p rastgele ReLU gizli nöronlu, yalnız çıktı katmanı (hafif ridge ile) eğitilen tek gizli katmanlı ağ.
import { fmt, linspace, range } from "./plot.js";
import { ridge } from "./ddescent.js";

const truth = (x) => Math.sin(3 * x) + 0.4 * x;
export function experiment(q, r) {
  const sig = q.noise, lam = q.lam, D = 4, Pm = 120, NV = 200;
  const draws = range(D).map(() => ({ F: range(Pm).map(() => ({ a: 3 * r.normal(), b: 2 * r.normal() })), X: range(400).map(() => 2 * r() - 1), E: range(400).map(() => r.normal()) }));
  const Xv = range(NV).map(() => 2 * r() - 1), Yv = Xv.map((x) => truth(x) + sig * r.normal());
  const feat = (F, p, x) => { const v = new Array(p + 1); v[0] = 1; for (let j = 0; j < p; j++) { const z = F[j].a * x + F[j].b; v[j + 1] = z > 0 ? z : 0; } return v; };
  const fitEval = (d, p, n) => {
    const X = d.X.slice(0, n), Y = X.map((x, i) => truth(x) + sig * d.E[i]), w = ridge(X.map((x) => feat(d.F, p, x)), Y, lam);
    const pr = (x) => feat(d.F, p, x).reduce((s, v, j) => s + v * w[j], 0);
    return { tr: X.reduce((s, x, i) => s + (pr(x) - Y[i]) ** 2, 0) / n, va: Xv.reduce((s, x, i) => s + (pr(x) - Yv[i]) ** 2, 0) / NV, pr, X, Y };
  };
  const avg = (p, n) => { let a = 0, b = 0; for (const d of draws) { const o = fitEval(d, p, n); a += o.tr / D; b += o.va / D; } return [a, b]; };
  return { avg, fitEval, draws };
}

export default {
  params: [
    { id: "p", label: "Model genişliği p (gizli ReLU nöronu)", min: 1, max: 120, value: 4, int: true, log: true },
    { id: "n", label: "Eğitim örneği sayısı n", min: 5, max: 400, value: 40, int: true, log: true },
    { id: "noise", label: "Etiket gürültüsü σ", min: 0.02, max: 0.6, step: 0.02, value: 0.3 },
    { id: "lam", label: "Ridge λ (son katman)", min: 1e-6, max: 10, value: 1e-3, log: true },
  ],
  seed: true,
  run(q, r) {
    const p = Math.round(q.p), n = Math.round(q.n), E = experiment(q, r), s2 = q.noise ** 2;
    const Ns = [...new Set([5, 8, 12, 20, 30, 45, 70, 100, 150, 220, 300, 400, n])].sort((a, b) => a - b), Ps = [...new Set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 90, 120, p])].sort((a, b) => a - b);
    const LC = Ns.map((k) => E.avg(p, k)), CC = Ps.map((k) => E.avg(k, n)), [tr, va] = E.avg(p, n), F = E.fitEval(E.draws[0], p, n), xs = linspace(-1, 1, 240);
    const gap = va - tr, ex = tr - s2, hiB = ex > 0.5 * s2, hiV = gap > 0.5 * s2;
    const diag = hiB && hiV ? "yanlılık ve varyans birlikte" : hiB ? "yüksek yanlılık (eksik uyum)" : hiV ? "yüksek varyans (aşırı uyum)" : "dengeli";
    const lg = (v) => v.map((x) => Math.max(x, 1e-5)), xtN = [5, 10, 30, 100, 300].map((v) => [Math.log10(v), String(v)]), xtP = [1, 3, 10, 30, 100].map((v) => [Math.log10(v), String(v)]);
    return {
      plots: [
        { title: `Öğrenme eğrisi: p = ${p} sabit, n değişiyor`, xlabel: "eğitim örneği n (log)", ylabel: "MSE", ylog: true, ylim: [Math.max(1e-4, s2 / 20), 10], xticks: xtN, series: [{ name: "doğrulama", x: Ns.map(Math.log10), y: lg(LC.map((v) => v[1])), c: 2, dots: true }, { name: "eğitim", x: Ns.map(Math.log10), y: lg(LC.map((v) => v[0])), c: 0, dots: true }], marks: [{ x: Math.log10(n), label: "seçili n" }, { y: s2, label: "σ²" }] },
        { title: `Kapasite eğrisi: n = ${n} sabit, p değişiyor`, xlabel: "genişlik p (log)", ylabel: "MSE", ylog: true, ylim: [Math.max(1e-4, s2 / 20), 10], xticks: xtP, series: [{ name: "doğrulama", x: Ps.map(Math.log10), y: lg(CC.map((v) => v[1])), c: 2, dots: true }, { name: "eğitim", x: Ps.map(Math.log10), y: lg(CC.map((v) => v[0])), c: 0, dots: true }], marks: [{ x: Math.log10(p), label: "seçili p" }, { y: s2, label: "σ²" }] },
        { title: `Uyum (n = ${n}, p = ${p}, ilk çekiliş)`, xlabel: "x", ylabel: "y", ylim: [-2.5, 2.5], series: [{ name: "gerçek f", x: xs, y: xs.map(truth), c: 5, dash: true }, { name: "model", x: xs, y: xs.map(F.pr), c: 0 }, { name: "eğitim örnekleri", x: F.X, y: F.Y, kind: "scatter", c: 1, r: 3 }] },
      ],
      stats: [
        { label: "Eğitim MSE", value: tr },
        { label: "Doğrulama MSE", value: va },
        { label: "Fark (doğrulama − eğitim)", value: gap },
        { label: "Gürültü tabanı σ²", value: s2 },
        { label: "Teşhis (eşik: σ²/2)", value: diag },
      ],
      note: `${diag.startsWith("yanlılık ve") ? "Eğitim hatası gürültü tabanının belirgin üstünde <b>ve</b> doğrulama hatası eğitim hatasından belirgin büyük: hem kapasite hem veri sınırlı. Önce yanlılığı çöz (p ↑), sonra varyansı (n ↑ ya da λ ↑)." : diag.startsWith("yüksek yanlılık") ? "Eğitim hatası bile gürültü tabanının belirgin üstünde ve iki eğri birbirine yakın: model veriyi temsil edemiyor. Öğrenme eğrisi sağa doğru düzleşmiş — <b>daha çok veri yardım etmez</b>; kapasiteyi artır (p ↑), düzenlileştirmeyi azalt (λ ↓)." : diag.startsWith("yüksek varyans") ? "Eğitim hatası düşük, doğrulama hatası çok daha yüksek: model gürültüyü ezberliyor. Öğrenme eğrisinde iki eğri n arttıkça birbirine yaklaşıyor — <b>daha çok veri yardım eder</b>; ya da kapasiteyi azalt, düzenlileştirmeyi artır (λ ↑)." : "Eğitim ve doğrulama hatası gürültü tabanına yakın ve birbirine yakın: bu (n, p) için iyi bir denge."} Kural (Andrew Ng'nin teşhis tablosu): eğitim hatası yüksekse yanlılık sorunu, eğitim–doğrulama farkı büyükse varyans sorunu vardır.`,
    };
  },
};
