// M4.7 — LayerNorm'un yeri: Post-LN (orijinal Transformer), Pre-LN (GPT-2 ve sonrası) ve normalizasyonsuz artık yığın.
// Rastgele başlatılmış L artık blok (her biri MLP: d → 4d → d, ReLU); katman başına ‖∂L/∂W₂‖ ve artık akışın RMS'i. Dikkat alt-katmanı yok (bkz. sınırlar).
import { fmt, range } from "./plot.js";
import { norm } from "./normdeep.js";

const mm = (X, W) => X.map((x) => W.map((w) => w.reduce((s, v, j) => s + v * x[j], 0)));
const mmT = (G, W) => G.map((g) => range(W[0].length).map((j) => W.reduce((s, w, a) => s + w[j] * g[a], 0)));
const gnorm = (G, X) => { let s = 0; for (let a = 0; a < G[0].length; a++) for (let b = 0; b < X[0].length; b++) { let t = 0; for (let i = 0; i < G.length; i++) t += G[i][a] * X[i][b]; s += t * t; } return Math.sqrt(s); };
const rms = (X) => Math.sqrt(X.reduce((s, x) => s + x.reduce((t, v) => t + v * v, 0), 0) / (X.length * X[0].length));

// arch: 0 = Post-LN  x ← LN(x + F(x)),  1 = Pre-LN  x ← x + F(LN(x)) (+ son LN),  2 = LN yok  x ← x + F(x)
export function residualStack(arch, L, ps, r, d = 32, B = 8) {
  const h = 4 * d, W1 = range(L).map(() => range(h).map(() => range(d).map(() => Math.sqrt(2 / d) * r.normal())));
  const W2 = range(L).map(() => range(d).map(() => range(h).map(() => ps * Math.sqrt(1 / h) * r.normal())));
  let X = range(B).map(() => range(d).map(() => r.normal())); const cache = [], xr = [rms(X)];
  for (let l = 0; l < L; l++) {
    const c = {}; c.inp = arch === 1 ? (c.N = norm(X, 2)).Y : X;
    c.Z = mm(c.inp, W1[l]); c.A = c.Z.map((z) => z.map((v) => Math.max(0, v))); const F = mm(c.A, W2[l]);
    let S = X.map((x, i) => x.map((v, j) => v + F[i][j])); if (arch === 0) { c.P = norm(S, 2); S = c.P.Y; }
    cache.push(c); X = S; xr.push(rms(X));
  }
  let fin = null; if (arch === 1) { fin = norm(X, 2); X = fin.Y; }
  let G = X.map((x) => x.map(() => r.normal())); if (fin) G = fin.back(G);
  const g2 = new Array(L), gx = new Array(L);
  for (let l = L - 1; l >= 0; l--) {
    const c = cache[l]; if (arch === 0) G = c.P.back(G);
    g2[l] = gnorm(G, c.A);
    const dZ = mmT(G, W2[l]).map((a, i) => a.map((v, j) => (c.Z[i][j] > 0 ? v : 0)));
    let din = mmT(dZ, W1[l]); if (arch === 1) din = c.N.back(din);
    G = G.map((g, i) => g.map((v, j) => v + din[i][j])); gx[l] = rms(G);
  }
  return { g2, gx, xr };
}
const NAMES = ["Post-LN: x ← LN(x + F(x))", "Pre-LN: x ← x + F(LN(x))", "LN yok: x ← x + F(x)"];

export default {
  params: [
    { id: "arch", label: "Normalizasyonun yeri", options: NAMES, value: 0 },
    { id: "L", label: "Artık blok sayısı L", min: 2, max: 48, value: 24, int: true },
    { id: "ps", label: "Çıkış izdüşümü W₂'nin başlatma ölçeği", options: ["standart (1)", "1/√L (GPT-2 tarzı)"], value: 0, help: "GPT-2 artık dallara yazan katmanların ağırlıklarını 1/√N ile ölçekler (N = artık dal sayısı; blok başına iki dal olduğu için nanoGPT'de 1/√(2L))." },
  ],
  seed: true,
  run(q, r) {
    const L = Math.round(q.L), ps = q.ps === 1 ? 1 / Math.sqrt(L) : 1, R = residualStack(q.arch, L, ps, r), ls = range(L).map((l) => l + 1);
    const ratio = R.g2[0] / R.g2[L - 1], grow = R.xr[L] / R.xr[0];
    return {
      plots: [
        { title: "‖∂L/∂W₂ˡ‖: her bloğun çıkış ağırlığının gradyanı (log)", xlabel: "blok l (1 = girdiye en yakın)", ylabel: "gradyan normu", ylog: true, ylim: [Math.min(...R.g2) / 4, Math.max(...R.g2) * 4], series: [{ name: NAMES[q.arch], x: ls, y: R.g2.map((v) => Math.max(v, 1e-30)), c: 0, dots: L <= 24 }] },
        { title: "Artık akışın RMS'i, bloktan bloğa (log)", xlabel: "blok l (0 = girdi)", ylabel: "RMS(x)", ylog: true, ylim: [Math.min(...R.xr) / 4, Math.max(...R.xr) * 4], series: [{ name: "RMS(xˡ)", x: range(L + 1), y: R.xr, c: 2, dots: L <= 24 }] },
      ],
      stats: [
        { label: "Gradyan: ilk blok / son blok", value: `${fmt(R.g2[0], 3)} / ${fmt(R.g2[L - 1], 3)} (oran ${fmt(ratio, 3)})` },
        { label: "Artık akış RMS'i: girdi → çıkış", value: `${fmt(R.xr[0], 3)} → ${fmt(R.xr[L], 3)} (× ${fmt(grow, 3)})` },
        { label: "√(L + 1) (Pre-LN'de beklenen büyüme)", value: fmt(Math.sqrt(L + 1), 3) },
      ],
      note: q.arch === 0
        ? `<b>Post-LN:</b> her bloktan sonra LN artık akışı RMS 1'e geri çeker; akış büyümez. Son blokların gradyanı derinlikten bağımsız olarak büyüktür (L'yi değiştir: son blok gradyanı ≈ aynı kalır) — Xiong ve ark. (2020) bunun büyük öğrenme oranıyla birleşince başlangıçta kararsızlığa yol açtığını, orijinal Transformer'ın bu yüzden <b>ısınmaya (warmup)</b> muhtaç olduğunu gösterir. Ayrıca özdeşlik yolu her blokta LN'den geçtiği için "temiz" değildir.`
        : q.arch === 1
          ? `<b>Pre-LN:</b> artık akışa hiç dokunulmaz, her blok yalnız normalize edilmiş bir kopyayı okur ve sonucu akışa <b>ekler</b>. ${q.ps === 0 ? `Varyanslar toplandığı için akışın RMS'i ≈ √(L + 1) ile büyür (× ${fmt(grow, 3)}); son LN girdiyi 1/RMS ile küçülttüğü için üst blokların gradyanı ~1/√L ile azalır (L'yi artır: son blok gradyanı düşer). Xiong ve ark.: Pre-LN'de gradyanlar derinlikle iyi huylu, ısınmasız eğitilebilir.` : `1/√L ölçeğiyle her dal küçük katkı yapar: akış ancak ≈ √2 kat büyür ve gradyan profili derinlik boyunca dengelenir. GPT-2'nin "artık katman başlatması" budur.`}`
          : `<b>Normalizasyon yok:</b> ${q.ps === 0 ? `her blok akışın varyansını ≈ 2 ile çarpar (F(x)'in varyansı x'inkine yakın): L blokta RMS ≈ (√2)^L mertebesinde büyür (burada × ${fmt(grow, 3)}), gradyanlar da onunla patlar. LN'siz derin artık ağlar ancak dalları küçültülerek eğitilebilir.` : `1/√L ölçeği varyans büyümesini toplam ≈ 2 ile sınırlıyor: LN olmadan da akış kararlı. Fixup/T-Fixup gibi yöntemler normalizasyonu böyle bir başlatmayla değiştirmeyi dener.`}`,
    };
  },
};
