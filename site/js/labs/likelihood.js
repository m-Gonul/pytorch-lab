// M0.4 — Yazı-tura için olabilirlik: L(p) = p^k (1−p)^(n−k). MLE k/n; Beta(a, a) önseliyle MAP (k+a−1)/(n+2a−2).
import { fmt, linspace } from "./plot.js";

export const logLik = (p, n, k) => (k ? k * Math.log(p) : 0) + (n - k ? (n - k) * Math.log(1 - p) : 0);
export const mapEst = (n, k, a) => (n + 2 * a - 2 > 0 ? (k + a - 1) / (n + 2 * a - 2) : 0.5);

export default {
  params: [
    { id: "n", label: "Atış sayısı n", min: 1, max: 200, step: 1, value: 10, int: true },
    { id: "k", label: "Tura sayısı k", min: 0, max: 200, step: 1, value: 7, int: true, help: "k ≤ n olmalı." },
    { id: "a", label: "Önsel gücü a (Beta(a, a))", min: 1, max: 30, step: 0.5, value: 1, help: "a = 1: düzgün önsel, MAP = MLE. a büyüdükçe tahmin 0.5'e çekilir: her iki yana (a − 1) sahte gözlem eklemek gibi." },
    { id: "probe", label: "İncelenen p", min: 0.01, max: 0.99, step: 0.01, value: 0.5, hardMin: 0.0001, hardMax: 0.9999, why: "p = 0 ya da 1'de log-olabilirlik −∞ olabilir." },
  ],
  validate(q) {
    if (q.k > q.n) return [{ level: "error", msg: `Tura sayısı k = ${q.k}, atış sayısı n = ${q.n}'den büyük olamaz.` }];
    if (q.k === 0 || q.k === q.n) return [{ level: "warn", msg: `k = ${q.k}: MLE p̂ = ${q.k / q.n} — hiç ${q.k ? "yazı" : "tura"} görülmedi diye bunun olasılığı 0 sayılıyor. Az veride MLE aşırı güvenlidir; önsel (a > 1) bunu yumuşatır.` }];
    return [];
  },
  run(q) {
    const { n, k, a, probe } = q, ps = linspace(0.001, 0.999, 400);
    const mle = k / n, map = mapEst(n, k, a);
    const ll = ps.map((p) => logLik(p, n, k)), llMax = logLik(Math.min(Math.max(mle, 1e-9), 1 - 1e-9), n, k);
    const post = ps.map((p) => (k + a - 1) * Math.log(p) + (n - k + a - 1) * Math.log(1 - p)), pMax = Math.max(...post);
    const llp = logLik(probe, n, k), slope = k / probe - (n - k) / (1 - probe);
    const se = Math.sqrt(Math.max(mle * (1 - mle), 1e-12) / n);
    return {
      plots: [
        { title: "Olabilirlik L(p) ve sonsal (en büyük değere bölünmüş)", xlabel: "p", ylabel: "göreli değer", ylim: [0, 1.05], series: [
          { name: "L(p) / max L", x: ps, y: ll.map((v) => Math.exp(v - llMax)), c: 0 },
          ...(a > 1 ? [{ name: `sonsal ∝ L(p)·Beta(p; ${fmt(a, 3)}, ${fmt(a, 3)})`, x: ps, y: post.map((v) => Math.exp(v - pMax)), c: 2, dash: true }] : []),
        ], marks: [{ x: mle, label: "MLE" }, ...(a > 1 ? [{ x: map, label: "MAP" }] : []), { x: probe, label: "p" }] },
        { title: "Log-olabilirlik ℓ(p) = k log p + (n−k) log(1−p)", xlabel: "p", ylabel: "ℓ(p)", ylim: [Math.max(llMax - 30, Math.min(...ll)), llMax + 1], series: [
          { name: "ℓ(p)", x: ps, y: ll, c: 0 },
          { name: `p'deki teğet: eğim ${fmt(slope, 3)}`, x: ps, y: ps.map((x) => llp + slope * (x - probe)), c: 1, dash: true },
          { name: "", x: [probe], y: [llp], kind: "scatter", c: 1, r: 4 },
        ], marks: [{ x: mle, label: "MLE: eğim 0" }] },
      ],
      stats: [
        { label: "MLE p̂ = k/n", value: mle },
        { label: "MAP (k+a−1)/(n+2a−2)", value: map },
        { label: `L(p = ${fmt(probe, 3)}) = pᵏ(1−p)ⁿ⁻ᵏ`, value: Math.exp(llp) },
        { label: "ℓ(p) = log L(p)", value: llp },
        { label: "dℓ/dp = k/p − (n−k)/(1−p)", value: slope },
        { label: "MLE'nin yaklaşık standart hatası √(p̂(1−p̂)/n)", value: se },
      ],
      note: `L(p) bir olasılık <b>dağılımı değildir</b>: p üzerinden integrali 1 değil; burada şekli görmek için tepe değerine bölündü. ℓ'nin eğimi MLE'de sıfırdır (dℓ/dp = 0 ⇒ p = k/n). Aynı oranı (k/n) koruyup n'yi artır: eğri daralır — daha çok veri, daha kesin tahmin (genişlik ~1/√n). ${a > 1 ? `Önsel Beta(${fmt(a, 3)}, ${fmt(a, 3)}), tahmini 0.5'e doğru çeker; n büyüdükçe verinin ağırlığı önseli bastırır ve MAP → MLE.` : "a'yı artırarak önselin etkisini gör: MAP, MLE'nin düzenlileştirilmiş hâlidir."}`,
    };
  },
};
