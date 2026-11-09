// M4.2 — Düz ağ vs artık (residual) ağ: aynı ağırlıklarla, katman katman aktivasyon std'si ve ‖∂L/∂hˡ‖.
// Düz: hˡ⁺¹ = φ(Wˡ hˡ). Artık: hˡ⁺¹ = hˡ + α·Vˡ φ(Wˡ hˡ). Jacobian'a eklenen I, gradyanın çarpımla sönmesini/patlamasını engeller.
import { fmt, range } from "./plot.js";

const ACT = [(z) => Math.tanh(z), (z) => (z > 0 ? z : 0)], DACT = [(z) => 1 - Math.tanh(z) ** 2, (z) => (z > 0 ? 1 : 0)];
const mv = (M, v) => M.map((r) => r.reduce((s, w, j) => s + w * v[j], 0)), mtv = (M, g) => M[0].map((_, j) => M.reduce((s, r, i) => s + r[j] * g[i], 0));
const nrm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export function run(q, r) {
  const n = 32, L = Math.round(q.L), B = 8, base = q.act === 1 ? 2 : 1, a = q.alpha;
  const W = range(L).map(() => range(n).map(() => range(n).map(() => q.c * Math.sqrt(base / n) * r.normal()))), V = range(L).map(() => range(n).map(() => range(n).map(() => Math.sqrt(1 / n) * r.normal())));
  const X = range(B).map(() => range(n).map(() => r.normal())), dir = range(n).map(() => r.normal());
  const go = (res) => {
    const std = [], gn = new Array(L + 1).fill(0);
    for (const x of X) {
      const H = [x], Z = [];
      for (let l = 0; l < L; l++) { const z = mv(W[l], H[l]); Z.push(z); const f = z.map(ACT[q.act]); H.push(res ? H[l].map((h, i) => h + a * mv(V[l], f)[i]) : f); }
      let g = dir.map((d) => d / Math.sqrt(n)); gn[L] += nrm(g) / B;
      for (let l = L - 1; l >= 0; l--) {
        const dz = res ? mtv(V[l], g).map((v, i) => a * v * DACT[q.act](Z[l][i])) : g.map((v, i) => v * DACT[q.act](Z[l][i]));
        const back = mtv(W[l], dz); g = res ? g.map((v, i) => v + back[i]) : back; gn[l] += nrm(g) / B;
      }
      H.forEach((h, l) => { std[l] = (std[l] || 0) + Math.sqrt(h.reduce((s, v) => s + v * v, 0) / n) / B; });
    }
    return { std, gn };
  };
  return { plain: go(false), res: go(true), L };
}

export default {
  params: [
    { id: "L", label: "Derinlik (blok sayısı)", min: 2, max: 60, value: 30, int: true },
    { id: "act", label: "Aktivasyon", options: ["tanh", "ReLU"], value: 1 },
    { id: "c", label: "Başlatma çarpanı (1 = He/LeCun)", min: 0.3, max: 2, step: 0.05, value: 0.8 },
    { id: "alpha", label: "Artık dalın ölçeği α", min: 0, max: 1, step: 0.01, value: 0.2, help: "α = 1 standart toplama; α = 1/√L civarı derinlikle büyümeyi dengeler; α = 0 her bloğu kimlik yapar (Fixup/SkipInit'in başlangıç durumu)." },
  ],
  seed: true,
  run(q, r) {
    const R = run(q, r), ls = range(R.L + 1), lg = (v) => v.map((x) => Math.max(x, 1e-30));
    const rp = R.plain.gn[0] / R.plain.gn[R.L], rr = R.res.gn[0] / R.res.gn[R.L];
    return {
      plots: [
        { title: "‖∂L/∂hˡ‖ (girdiye yakın katman solda)", xlabel: "katman l", ylabel: "gradyan normu", ylog: true, series: [{ name: "düz ağ", x: ls, y: lg(R.plain.gn), c: 2, dots: true }, { name: "artık ağ", x: ls, y: lg(R.res.gn), c: 0, dots: true }] },
        { title: "Aktivasyonların RMS'i, katmana göre", xlabel: "katman l", ylabel: "RMS", ylog: true, series: [{ name: "düz ağ", x: ls, y: lg(R.plain.std), c: 2, dots: true }, { name: "artık ağ", x: ls, y: lg(R.res.std), c: 0, dots: true }] },
      ],
      stats: [
        { label: "Düz ağ: ilk / son katman gradyan oranı", value: rp },
        { label: "Artık ağ: ilk / son katman gradyan oranı", value: rr },
        { label: "Son katman RMS: düz / artık", value: `${fmt(R.plain.std[R.L], 3)} / ${fmt(R.res.std[R.L], 3)}` },
      ],
      note: `Düz ağda gradyan katman katman Jacobian'larla <b>çarpılır</b>: ∂L/∂hˡ = (Wˡ)ᵀ diag(φ′) ⋯; çarpanların tipik büyüklüğü 1'den farklıysa (başlatma çarpanı ${fmt(q.c, 2)}) etki derinlikle üstel büyür — burada ${R.L} katmanda ilk/son oranı ${fmt(rp, 2)}. Artık blokta Jacobian <b>I + α·(…)</b>'dır: gradyanın kimlik yolu hiçbir çarpandan geçmeden ilk katmana ulaşır, oran ${fmt(rr, 2)}. ${q.alpha > 0.5 ? "Ama α büyükken her blok varyans ekler: aktivasyonlar ve gradyanlar derinlikle (yavaşça) büyür — BatchNorm ya da 1/√L benzeri ölçekleme bu yüzden gerekir." : q.alpha === 0 ? "α = 0: her blok tam kimlik, gradyan hiç değişmeden geçer (Fixup'ın sıfırla başlatılan son katmanı)." : "Küçük α ile ağ başta kimliğe yakındır: hem ileri hem geri sinyal korunur."}`,
    };
  },
};
