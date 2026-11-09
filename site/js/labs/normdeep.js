// M3.6 — Kötü başlatılmış derin bir MLP'de normalizasyon: katman başına aktivasyon std'si ve gradyan normu (yok / BatchNorm / LayerNorm / RMSNorm).
import { fmt, range } from "./plot.js";

const ACT = [(z) => Math.tanh(z), (z) => Math.max(0, z)], DACT = [(z) => 1 - Math.tanh(z) ** 2, (z) => (z > 0 ? 1 : 0)];
// normalize: BN = her özellik için batch ekseninde; LN = her örnek için özellik ekseninde; RMS = her örnek için, ortalamasız
export function norm(Z, kind, eps = 1e-5) {
  const B = Z.length, n = Z[0].length;
  if (kind === 0) return { Y: Z, back: (G) => G };
  if (kind === 1) {
    const mu = range(n).map((j) => Z.reduce((s, r) => s + r[j], 0) / B), v = range(n).map((j) => Z.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / B), sd = v.map((x) => Math.sqrt(x + eps));
    const X = Z.map((r) => r.map((z, j) => (z - mu[j]) / sd[j]));
    return { Y: X, back: (G) => { const s1 = range(n).map((j) => G.reduce((s, r) => s + r[j], 0)), s2 = range(n).map((j) => G.reduce((s, r, i) => s + r[j] * X[i][j], 0)); return G.map((r, i) => r.map((g, j) => (B * g - s1[j] - X[i][j] * s2[j]) / (B * sd[j]))); } };
  }
  const rows = Z.map((r) => { const mu = kind === 2 ? r.reduce((s, x) => s + x, 0) / n : 0, v = r.reduce((s, x) => s + (x - mu) ** 2, 0) / n, sd = Math.sqrt(v + eps); return { mu, sd, x: r.map((z) => (z - mu) / sd) }; });
  return { Y: rows.map((o) => o.x), back: (G) => G.map((r, i) => { const { sd, x } = rows[i], s1 = kind === 2 ? r.reduce((s, g) => s + g, 0) : 0, s2 = r.reduce((s, g, j) => s + g * x[j], 0); return r.map((g, j) => (n * g - s1 - x[j] * s2) / (n * sd)); }) };
}

export default {
  params: [
    { id: "norm", label: "Normalizasyon (her doğrusal katmandan sonra, aktivasyondan önce)", options: ["yok", "BatchNorm", "LayerNorm", "RMSNorm"], value: 0 },
    { id: "act", label: "Aktivasyon", options: ["tanh", "ReLU"], value: 1 },
    { id: "scale", label: "Başlatma çarpanı (1 = He/LeCun uygun ölçek)", min: 0.2, max: 3, step: 0.05, value: 0.6, help: "Var(w) = çarpan² × (doğru varyans). 1'den uzaklaştıkça normalizasyonsuz ağda sinyal katmanlar boyunca söner ya da patlar." },
    { id: "L", label: "Derinlik", min: 2, max: 40, value: 20, int: true },
    { id: "B", label: "Batch boyu", min: 2, max: 128, value: 32, int: true, log: true },
  ],
  seed: true,
  run(q, r) {
    const n = 48, L = Math.round(q.L), Bs = Math.round(q.B), base = q.act === 1 ? 2 : 1;
    const W = range(L).map(() => range(n).map(() => range(n).map(() => q.scale * Math.sqrt(base / n) * r.normal())));
    let H = range(Bs).map(() => range(n).map(() => r.normal())); const cache = [], fstd = [];
    for (let l = 0; l < L; l++) {
      const Z = H.map((h) => W[l].map((w) => w.reduce((s, wij, j) => s + wij * h[j], 0))), N = norm(Z, q.norm), A = N.Y.map((rw) => rw.map(ACT[q.act]));
      cache.push({ H, N, Zn: N.Y }); H = A;
      const all = A.flat(), m = all.reduce((s, x) => s + x, 0) / all.length; fstd.push(Math.sqrt(all.reduce((s, x) => s + (x - m) ** 2, 0) / all.length));
    }
    // geri: kayıp = Σ çıktı · rastgele yön (sabit), katman başına ‖∂L/∂W‖
    let G = H.map((h) => h.map(() => r.normal() / Bs)); const gn = new Array(L);
    for (let l = L - 1; l >= 0; l--) {
      const { H: Hin, N, Zn } = cache[l], dZn = G.map((rw, i) => rw.map((g, j) => g * DACT[q.act](Zn[i][j]))), dZ = N.back(dZn);
      let s = 0; for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let g = 0; for (let i = 0; i < Bs; i++) g += dZ[i][a] * Hin[i][b]; s += g * g; } gn[l] = Math.sqrt(s);
      G = dZ.map((dz) => range(n).map((b) => W[l].reduce((t, w, a) => t + w[b] * dz[a], 0)));
    }
    const ls = range(L).map((l) => l + 1), ratio = gn[0] / Math.max(gn[L - 1], 1e-300);
    return {
      plots: [
        { title: "Aktivasyonların std'si, katmana göre", xlabel: "katman", ylabel: "std", ylog: true, series: [{ name: ["normalizasyonsuz", "BatchNorm", "LayerNorm", "RMSNorm"][q.norm], x: ls, y: fstd.map((v) => Math.max(v, 1e-30)), c: 0, dots: true }] },
        { title: "‖∂L/∂Wˡ‖ (mutlak), katmana göre", xlabel: "katman", ylabel: "gradyan normu", ylog: true, series: [{ name: "gradyan normu", x: ls, y: gn.map((v) => Math.max(v, 1e-30)), c: 2, dots: true }] },
      ],
      stats: [
        { label: "Son katman aktivasyon std'si", value: fstd[L - 1] },
        { label: "Gradyan normu: ilk katman / son katman", value: `${fmt(gn[0], 3)} / ${fmt(gn[L - 1], 3)}` },
        { label: "İlk / son oranı (1 = dengeli)", value: ratio },
      ],
      note: `${q.norm === 0 ? `<b>Normalizasyon yok:</b> başlatma çarpanı ${fmt(q.scale, 2)}, her katmanda varyansı yaklaşık ${fmt(q.scale * q.scale, 3)} ile çarpar — ${L} katmanda ${fmt(q.scale ** (2 * L), 2)}. Aktivasyonlar üstel olarak ${q.scale < 0.97 ? "söner" : q.scale > 1.03 ? "patlar" : "korunur"}; gradyan normlarının <b>mutlak</b> ölçeği de onlarla birlikte bozulur (∂L/∂W = δ·aᵀ, iki çarpan da etkilenir). Katmanlar arası oran ~1 kalabilir ama gradyanın kendisi çok küçük ya da çok büyüktür: öğrenme oranını buna göre ayarlamak imkânsızlaşır.` : `<b>${["", "BatchNorm", "LayerNorm", "RMSNorm"][q.norm]}:</b> ön-aktivasyonlar her katmanda yeniden ölçeklenir; aktivasyon std'si ve gradyanların ölçeği başlatma çarpanından <b>bağımsız</b> olur (çarpanı değiştir: std değişmez). Normalize edilmiş katmanın çıktısı ağırlık ölçeğine değişmezdir (W → cW aynı çıktı, ∂L/∂W ise 1/c ile ölçeklenir). ${q.norm === 1 ? "Dikkat: BatchNorm'lu düz (artıksız) derin ağlarda başlangıçta gradyanlar <b>ilk katmanlara doğru büyür</b> (ilk/son oranına bak; Yang ve ark. 2019) — pratikte BN artık bağlantılarla birlikte kullanılır (ResNet, M4.2)." : "LN/RMSNorm'da bu büyüme çok daha ılımlıdır."}`}${q.norm === 1 && Bs <= 4 ? " <b>Küçük batch:</b> BN'in istatistikleri birkaç örnekten kestiriliyor — gürültülü ve örnekler birbirine bağımlı; B = 2'de her özellik ±1'e zorlanır." : ""}`,
    };
  },
};
