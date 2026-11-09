// M4.16 — Tercihlerden hizalama, tam izlenebilir bir oyuncakta: K olası yanıt, referans politika p_ref, gerçek ödül r.
// (1) KL cezalı amacın kapalı form optimumu π*_β ∝ p_ref·e^{r/β}; (2) best-of-n: p_ref'ten n örnek, en yüksek ödüllüyü seç
// (dağılımı F(y)^n − F(y⁻)^n); (3) DPO: p_ref'ten çekilen N çift, Bradley–Terry ile etiketlenir (P(a ≻ b) = σ(r_a − r_b));
// tablo politikası π_θ = softmax(θ) DPO kaybıyla eğitilir. Ödül–KL düzleminde karşılaştırma.
import { fmt, range } from "./plot.js";

const lse = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, x) => s + Math.exp(x - m), 0)); };
const softmax = (a) => { const z = lse(a); return a.map((x) => Math.exp(x - z)); };
export const kl = (q, p) => q.reduce((s, x, i) => (x > 0 ? s + x * Math.log(x / p[i]) : s), 0);
const Er = (q, r) => q.reduce((s, x, i) => s + x * r[i], 0);
export const optimal = (lp, r, beta) => softmax(lp.map((x, i) => x + r[i] / beta));
export function bon(p, r, n) {
  const idx = range(p.length).sort((a, b) => r[a] - r[b]), out = new Array(p.length).fill(0); let F = 0;
  for (const i of idx) { const F1 = F + p[i]; out[i] = F1 ** n - F ** n; F = F1; }
  return out;
}
export function dpo(pairs, lp, beta, steps, lr) {
  const th = lp.slice(), K = lp.length; let loss = 0, acc = 0;
  for (let s = 0; s < steps; s++) {
    const g = new Array(K).fill(0); loss = 0; acc = 0;
    for (const [w, l] of pairs) { const d = beta * ((th[w] - th[l]) - (lp[w] - lp[l])), sg = 1 / (1 + Math.exp(d)); loss += Math.log1p(Math.exp(-d)); acc += d > 0 ? 1 : 0; g[w] -= beta * sg; g[l] += beta * sg; }
    for (let k = 0; k < K; k++) th[k] -= (lr * g[k]) / pairs.length * K;
  }
  return { pi: softmax(th), loss: loss / pairs.length, acc: acc / pairs.length };
}

export default {
  params: [
    { id: "K", label: "Yanıt sayısı K", min: 4, max: 40, value: 12, int: true },
    { id: "sharp", label: "Referans politikanın keskinliği", min: 0, max: 3, step: 0.1, value: 1 },
    { id: "beta", label: "KL katsayısı β", min: 0.05, max: 3, value: 0.5, log: true },
    { id: "n", label: "Best-of-n: n", min: 1, max: 64, value: 4, int: true, log: true },
    { id: "N", label: "DPO tercih çifti sayısı", min: 10, max: 3000, value: 300, int: true, log: true },
    { id: "steps", label: "DPO gradyan adımı", min: 10, max: 3000, value: 800, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const lp0 = range(q.K).map(() => q.sharp * r.normal()), Z = lse(lp0), lp = lp0.map((x) => x - Z), p = lp.map(Math.exp), rew = range(q.K).map(() => r.normal());
    const cdf = []; p.reduce((a, x, i) => (cdf[i] = a + x), 0); const draw = () => { const u = r(); const i = cdf.findIndex((c) => u <= c); return i < 0 ? q.K - 1 : i; };
    const pairs = []; for (let k = 0; k < q.N; k++) { const a = draw(), b = draw(); if (a === b) continue; pairs.push(r() < 1 / (1 + Math.exp(-(rew[a] - rew[b]))) ? [a, b] : [b, a]); }
    const star = optimal(lp, rew, q.beta), B = bon(p, rew, q.n), D = dpo(pairs, lp, q.beta, q.steps, 0.5);
    const betas = range(25).map((i) => 0.03 * 1.35 ** i), front = betas.map((b) => { const s = optimal(lp, rew, b); return [kl(s, p), Er(s, rew)]; });
    const ns = [1, 2, 4, 8, 16, 32, 64, 128], bf = ns.map((n) => { const s = bon(p, rew, n); return [kl(s, p), Er(s, rew)]; });
    const order = range(q.K).sort((a, b) => rew[a] - rew[b]), seen = new Set(pairs.flat()), unseen = q.K - seen.size;
    const P = [`K = ${q.K} yanıt; referans altında beklenen ödül ${fmt(Er(p, rew), 3)}, en iyi yanıtın ödülü ${fmt(Math.max(...rew), 3)}.`];
    P.push(`KL cezalı optimum (β = ${q.beta}): E[r] = ${fmt(Er(star, rew), 3)}, KL(π*‖p_ref) = ${fmt(kl(star, p), 3)} nat. Best-of-${q.n}: E[r] = ${fmt(Er(B, rew), 3)}, KL = ${fmt(kl(B, p), 3)} (üst sınır log n − (n−1)/n = ${fmt(Math.log(q.n) - (q.n - 1) / q.n, 3)}).`);
    P.push(`DPO, ${pairs.length} çift (aynı yanıtlı çiftler atıldı), ${q.steps} adım: E[r] = ${fmt(Er(D.pi, rew), 3)}, KL(π_DPO‖p_ref) = ${fmt(kl(D.pi, p), 3)}, hedef optimuma uzaklık KL(π_DPO‖π*) = ${fmt(kl(D.pi, star), 3)}. Çiftlerde örtük ödülün sıralama doğruluğu ${fmt(D.acc, 3)}, kayıp ${fmt(D.loss, 3)}.${unseen ? ` ${unseen} yanıt hiçbir çiftte yer almadı: DPO onların olasılığını doğrudan değiştiremez (yalnız normalizasyonla).` : ""}`);
    P.push(kl(D.pi, star) > 0.1 ? "DPO optimuma tam ulaşmadı: sonlu veride az görülen yanıtların ödül farkı gürültülü kestirilir; tercihler deterministik görünürse (hep aynı sonuç) β log oran sınırsız büyüyebilir. Bu, DPO'nun bilinen aşırı uyum eğilimidir." : "Bu ayarda DPO, KL cezalı optimuma yakın: aynı amacın ödül modeli ve RL olmadan çözümü.");
    const xs = range(q.K);
    return {
      plots: [
        { title: "Politikalar (yanıtlar ödüle göre sıralı, soldan sağa artan)", xlabel: "yanıt (ödül sırası)", ylabel: "olasılık", yzero: true, xlim: [-0.6, q.K - 0.4], series: [
          { name: "p_ref", x: xs, y: order.map((i) => p[i]), kind: "bar", c: 4, bw: 0.22, off: -0.33 },
          { name: `π* (β = ${q.beta})`, x: xs, y: order.map((i) => star[i]), kind: "bar", c: 0, bw: 0.22, off: -0.11 },
          { name: "DPO", x: xs, y: order.map((i) => D.pi[i]), kind: "bar", c: 2, bw: 0.22, off: 0.11 },
          { name: `best-of-${q.n}`, x: xs, y: order.map((i) => B[i]), kind: "bar", c: 1, bw: 0.22, off: 0.33 }] },
        { title: "Ödül – KL sınırı", xlabel: "KL(π ‖ p_ref) (nat)", ylabel: "E[r]", series: [
          { name: "KL cezalı optimum (β taraması)", x: front.map((f) => f[0]), y: front.map((f) => f[1]), c: 0 },
          { name: "best-of-n (n = 1…128)", x: bf.map((f) => f[0]), y: bf.map((f) => f[1]), c: 1, dots: true, dash: true },
          { name: "seçili β, n ve DPO", x: [kl(star, p), kl(B, p), kl(D.pi, p)], y: [Er(star, rew), Er(B, rew), Er(D.pi, rew)], kind: "scatter", c: 2, r: 5 }] },
      ],
      stats: [
        { label: "E[r]: p_ref / π* / best-of-n / DPO", value: `${fmt(Er(p, rew), 3)} / ${fmt(Er(star, rew), 3)} / ${fmt(Er(B, rew), 3)} / ${fmt(Er(D.pi, rew), 3)}` },
        { label: "KL(·‖p_ref): π* / best-of-n / DPO", value: `${fmt(kl(star, p), 3)} / ${fmt(kl(B, p), 3)} / ${fmt(kl(D.pi, p), 3)}` },
        { label: "KL(π_DPO ‖ π*)", value: kl(D.pi, star) },
        { label: "DPO örtük ödül doğruluğu", value: D.acc },
      ],
      note: P.join(" "),
    };
  },
};
