// M11.7 — UCB1'in iç işleyişi: K Bernoulli kol (ortalamalar sabit, en iyisi 0.7), n. adıma kadar UCB1 oynanır ve o andaki görgül
// ortalamalar R̃(a) ile güven terimleri c·√(2 ln n / n_a) çizilir. Seçilen kol en yüksek üst sınırlı koldur. Ayrıca her kolun çekilme
// sayısı, Auer ve ark. (2002) üst sınırı 8 ln n / Δ_a² ile karşılaştırılır; birikimli pişmanlık ve Lai–Robbins alt sınırı
// Σ Δ_a ln n / KL(μ_a, μ*) (Bernoulli KL) ile.
import { fmt, range } from "./plot.js";

const MU = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
const kl = (p, q) => p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
export default {
  params: [
    { id: "K", label: "Kol sayısı", min: 2, max: 6, value: 4, int: true },
    { id: "n", label: "Adım n", min: 10, max: 5000, step: 10, value: 1000, int: true },
    { id: "c", label: "Güven çarpanı c (UCB1: c = 1)", min: 0, max: 2, step: 0.05, value: 1 },
  ],
  seed: true,
  run(q, r) {
    const K = q.K, mu = MU.slice(0, K), S = Array(K).fill(0), N = Array(K).fill(0); let reg = 0; const regs = [], ts = [];
    for (let t = 1; t <= q.n; t++) {
      let a;
      if (t <= K) a = t - 1;
      else { let best = -Infinity; for (let i = 0; i < K; i++) { const u = S[i] / N[i] + q.c * Math.sqrt((2 * Math.log(t)) / N[i]); if (u > best) { best = u; a = i; } } }
      const rw = r() < mu[a] ? 1 : 0; S[a] += rw; N[a]++; reg += mu[0] - mu[a];
      if (t % Math.max(1, Math.floor(q.n / 200)) === 0) { ts.push(t); regs.push(reg); }
    }
    const mean = S.map((s, i) => s / N[i]), bonus = N.map((n) => q.c * Math.sqrt((2 * Math.log(q.n)) / n)), ub = mean.map((m, i) => m + bonus[i]);
    const auer = mu.map((m, i) => (i === 0 ? NaN : (8 * Math.log(q.n)) / (mu[0] - m) ** 2)), lr = mu.slice(1).reduce((acc, m) => acc + ((mu[0] - m) * Math.log(q.n)) / kl(m, mu[0]), 0);
    const P = [`n = ${q.n}'de kolların çekilme sayıları: ${N.join(", ")} (gerçek ortalamalar ${mu.join(", ")}). Birikimli pişmanlık ${fmt(reg, 3)}.`];
    P.push(`Auer sınırı 8 ln n / Δ² (alt-en iyi kollar): ${auer.slice(1).map((v) => fmt(v, 3)).join(", ")} — gerçek sayılar bunun çok altında; sınır kötümserdir. Lai–Robbins alt sınırının öncü terimi Σ Δ ln n / KL ≈ ${fmt(lr, 3)}: hiçbir tutarlı algoritma bundan sistematik olarak daha az pişmanlık yapamaz.`);
    if (reg < lr) P.push("Bu tek çalıştırmada pişmanlık alt sınırın altında: Lai–Robbins sınırı beklenen pişmanlık ve her ortamda tutarlı algoritmalar içindir. Az keşif bazı tohumlarda şanslıdır, bazılarında en iyi kolu kaçırıp doğrusal pişmanlık yapar.");
    if (q.c === 0) P.push("c = 0: güven terimi yok, saf açgözlü. İlk çekilişlerde şanssız kalan en iyi kol bir daha denenmeyebilir (tohumları dene).");
    else if (q.c < 0.5) P.push("Küçük c az keşif demektir; çoğu tohumda daha az pişmanlık verir ama bazı tohumlarda en iyi kolu uzun süre bulamaz.");
    return {
      plots: [
        { title: `n = ${q.n}'de görgül ortalama ve güven terimi`, xlabel: "kol", ylabel: "değer", side: true, size: { w: 300, h: 240 }, xlim: [-0.6, K - 0.4], ylim: [0, 1.6], xticks: range(K).map((i) => [i, `μ=${mu[i]}`]),
          series: [{ name: "görgül ortalama", x: range(K), y: mean, kind: "bar", c: 0, bw: 0.5 }, { name: "ortalama + güven terimi", x: range(K), y: ub, kind: "scatter", c: 2 }, { name: "gerçek μ", x: range(K), y: mu, kind: "scatter", c: 4 }] },
        { title: "Birikimli pişmanlık", xlabel: "adım", ylabel: "pişmanlık", side: true, size: { w: 300, h: 240 }, series: [{ name: "UCB", x: ts, y: regs, c: 0 }, { name: "Lai–Robbins öncü terim", x: ts, y: ts.map((t) => mu.slice(1).reduce((acc, m) => acc + ((mu[0] - m) * Math.log(t)) / kl(m, mu[0]), 0)), c: 4, dash: true }] },
      ],
      stats: [{ label: "Çekilme sayıları", value: N.join(" / ") }, { label: "Pişmanlık", value: fmt(reg, 3) }, { label: "En büyük üst sınır", value: `kol ${ub.indexOf(Math.max(...ub)) + 1}` }],
      note: P.join(" "),
    };
  },
};
