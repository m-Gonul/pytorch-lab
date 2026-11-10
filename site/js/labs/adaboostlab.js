// M9.16 — Viola–Jones AdaBoost ve kaskad, küçük ölçekte. Rastgele bir Haar öznitelik havuzu (varyans normalize pencerelerde),
// yapay yüz/yüz olmayan eğitim (300 + 300) ve test (300 + 300) kümeleri. AdaBoost (VJ biçimi): ağırlıklar 1/(2m), 1/(2l); her turda
// en düşük ağırlıklı hatalı karar kütüğü; β = ε/(1 − ε), doğru sınıflananların ağırlığı β ile çarpılır; α = ln(1/β); güçlü sınıflandırıcı
// Σα·h ≥ ½Σα. Kaskad: her aşama ayrı AdaBoost, zor negatif madenciliğiyle (aşağıda).
import { fmt, range } from "./plot.js";
import { S, facePatch, nonFacePatch, integral, normed, feature, span } from "./haarfeat.js";

// Zor negatifler: yüzün parçaları (koyu bant + altında açık bant, açık dikey çubuk, koyu kısa çizgi) rastgele alt kümeler ve kaymalarla;
// tek bir öznitelik bunları yüzden ayıramaz, birkaç özniteliğin birleşimi gerekir.
function hardPatch(r, light) {
  const f = new Float64Array(S * S), g = 1 + light * (2 * r() - 1), c = 0.6 + 0.4 * r(), parts = [r() < 0.7, r() < 0.5, r() < 0.5, r() < 0.5], oy = Math.round(5 * r.normal()), ox = Math.round(3 * r.normal()), oy2 = Math.round(6 * r.normal());
  if (parts.every((p) => p)) parts[1 + Math.floor(r() * 3)] = false;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let v = 0.55; const yy = y - oy, xx = x - ox;
    if (parts[0] && yy >= 7 && yy < 11 && xx >= 3 && xx < 21) v = 0.55 - 0.3 * c;
    if (parts[1] && yy >= 11 && yy < 15 && xx >= 3 && xx < 21) v = 0.55 + 0.15 * c;
    if (parts[2] && y - oy2 >= 6 && y - oy2 < 13 && xx >= 10 && xx < 14) v = 0.55 + 0.2 * c;
    if (parts[3] && y - oy2 >= 17 && y - oy2 < 19 && xx >= 7 && xx < 17) v = 0.55 - 0.2 * c;
    f[y * S + x] = Math.max(0, Math.min(1, g * v + 0.04 * r.normal()));
  }
  return f;
}
function dataset(r, nP, nN, light, hard) { const X = [], y = []; for (let i = 0; i < nP; i++) { X.push(facePatch(r, light)); y.push(1); } for (let i = 0; i < nN; i++) { X.push(r() < hard ? hardPatch(r, light) : nonFacePatch(r, light)); y.push(0); } return { X, y }; }

function pool(r, n) { const F = []; while (F.length < n) { const t = Math.floor(r() * 5), [sx, sy] = span(t), w = 1 + Math.floor(r() * (S / sx)), h = 1 + Math.floor(r() * (S / sy)); if (sx * w > S || sy * h > S) continue; const x = Math.floor(r() * (S - sx * w + 1)), y = Math.floor(r() * (S - sy * h + 1)); F.push([t, x, y, w, h]); } return F; }
function values(F, X) { const II = X.map((f) => integral(normed(f))); return F.map((g) => Float64Array.from(II, (ii) => feature(ii, ...g))); }
export function adaboost(V, y, T) {
  const n = y.length, m = y.filter((v) => v).length, l = n - m, order = V.map((v) => range(n).sort((a, b) => v[a] - v[b]));
  let w = y.map((v) => (v ? 1 / (2 * m) : 1 / (2 * l))); const H = [];
  for (let t = 0; t < T; t++) {
    const s = w.reduce((a, b) => a + b, 0); w = w.map((v) => v / s);
    let Tp = 0, Tn = 0; y.forEach((v, i) => (v ? (Tp += w[i]) : (Tn += w[i])));
    let best = { err: Infinity };
    V.forEach((v, j) => { const o = order[j]; let Sp = 0, Sn = 0; for (let k = 0; k <= n; k++) { const th = k === 0 ? v[o[0]] - 1e-9 : k === n ? v[o[n - 1]] + 1e-9 : (v[o[k - 1]] + v[o[k]]) / 2, e1 = Sp + (Tn - Sn), e2 = Sn + (Tp - Sp); if (e1 < best.err) best = { err: e1, j, th, pol: 1 }; if (e2 < best.err) best = { err: e2, j, th, pol: -1 }; if (k < n) { if (y[o[k]]) Sp += w[o[k]]; else Sn += w[o[k]]; } } });
    const eps = Math.max(best.err, 1e-10), beta = eps / (1 - eps), alpha = Math.log(1 / beta);
    const hj = (v) => (best.pol * (v - best.th) > 0 ? 1 : 0);
    w = w.map((wi, i) => (hj(V[best.j][i]) === y[i] ? wi * beta : wi));
    H.push({ ...best, alpha });
  }
  return H;
}
const score = (H, Vt, i, k = H.length) => { let s = 0, a = 0; for (let t = 0; t < k; t++) { const h = H[t]; s += h.alpha * (h.pol * (Vt[h.j][i] - h.th) > 0 ? 1 : 0); a += h.alpha; } return s / a; };
export default {
  params: [
    { id: "T", label: "AdaBoost tur sayısı T (güçlü sınıflandırıcı)", min: 1, max: 31, value: 16, int: true },
    { id: "P", label: "Öznitelik havuzu", min: 50, max: 600, value: 300, int: true, log: true },
    { id: "d", label: "Kaskad: aşama başına en küçük tespit oranı d", min: 0.9, max: 0.999, step: 0.001, value: 0.99 },
    { id: "f", label: "Kaskad: aşama başına en büyük yanlış pozitif oranı f", min: 0.1, max: 0.8, step: 0.05, value: 0.4 },
    { id: "hard", label: "Zor negatif oranı (yüz parçaları)", min: 0, max: 0.9, step: 0.05, value: 0.5 },
    { id: "light", label: "Aydınlatma değişkenliği", min: 0, max: 0.8, step: 0.05, value: 0.4 },
  ],
  seed: true,
  run(q, r) {
    const tr = dataset(r, 300, 300, q.light, q.hard), te = dataset(r, 300, 2000, q.light, q.hard), F = pool(r, q.P), V = values(F, tr.X), Vt = values(F, te.X);
    const H = adaboost(V, tr.y, Math.max(q.T, 31));
    const err = (k, VV, yy) => yy.reduce((s, v, i) => s + ((score(H, VV, i, k) >= 0.5 ? 1 : 0) !== v ? 1 : 0), 0) / yy.length;
    const Ks = range(31).map((k) => k + 1), trE = Ks.map((k) => err(k, V, tr.y)), teE = Ks.map((k) => err(k, Vt, te.y));
    // Kaskad (Viola–Jones gibi): her aşama ayrı bir AdaBoost; pozitifler eğitim yüzleri (önceki aşamaları geçenler), negatifler büyük
    // bir havuzdan şimdiye kadarki kaskadı GEÇEN pencereler (zor negatif madenciliği). Aşamaya, eşik yüzlerin en az d oranını geçirecek
    // biçimde düşürüldüğünde negatiflerin en çok f'si geçene kadar öznitelik eklenir (aşama başına en çok 12).
    const NP = dataset(r, 0, 3000, q.light, q.hard), VN = values(F, NP.X), posIdx = tr.y.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    const evalStage = (st, vals) => { let s2 = 0, a2 = 0; for (const h of st.H) { s2 += h.alpha * (h.pol * (vals[h.j] - h.th) > 0 ? 1 : 0); a2 += h.alpha; } return s2 / a2 >= st.th; };
    const fv = (VV, i) => VV.map((v) => v[i]);
    const stagesC = []; let aliveP = posIdx.slice(), aliveN = range(NP.y.length);
    for (let sIdx = 0; sIdx < 6 && aliveN.length >= 20; sIdx++) {
      const negS = aliveN.slice(0, 300), Vs = V.map((v, j) => Float64Array.from([...aliveP.map((i) => v[i]), ...negS.map((i) => VN[j][i])])), ys = [...aliveP.map(() => 1), ...negS.map(() => 0)];
      const Hs = adaboost(Vs, ys, 12); let st = null;
      for (let k = 1; k <= 12; k++) {
        const sc = ys.map((_, i) => { let s2 = 0, a2 = 0; for (let t = 0; t < k; t++) { const h = Hs[t]; s2 += h.alpha * (h.pol * (Vs[h.j][i] - h.th) > 0 ? 1 : 0); a2 += h.alpha; } return s2 / a2; });
        const sp = sc.filter((_, i) => ys[i]).sort((x, y) => x - y), th = sp[Math.floor((1 - q.d) * sp.length)] - 1e-12, fr = sc.filter((v, i) => !ys[i] && v >= th).length / negS.length;
        st = { H: Hs.slice(0, k), th, k, fr }; if (fr <= q.f) break;
      }
      stagesC.push(st);
      aliveP = aliveP.filter((i) => evalStage(st, fv(V, i))); aliveN = aliveN.filter((i) => evalStage(st, fv(VN, i)));
    }
    const Dtr = aliveP.length / posIdx.length, Ksz = stagesC.map((st) => st.k);
    const runC = (VV, yy) => { const P = yy.filter((v) => v).length, N = yy.length - P, remP = Ksz.map(() => 0), remN = Ksz.map(() => 0); let featN = 0, featP = 0; yy.forEach((v, i) => { const x = fv(VV, i); let used = 0; for (let s = 0; s < stagesC.length; s++) { used += Ksz[s]; if (!evalStage(stagesC[s], x)) break; if (v) remP[s]++; else remN[s]++; } if (v) featP += used; else featN += used; }); return { D: remP[Ksz.length - 1] / P, F: remN[Ksz.length - 1] / N, remP: remP.map((c) => c / P), remN: remN.map((c) => c / N), featN: featN / N, featP: featP / P }; };
    const C = runC(Vt, te.y), dPow = q.d ** stagesC.length, tot = Ksz.reduce((x, y) => x + y, 0);
    const best = H[0], bf = F[best.j], tnames = ["2 yatay", "2 dikey", "3 yatay", "3 dikey", "4 çapraz"];
    const Pn = [`Tek AdaBoost: ilk seçilen öznitelik ${tnames[bf[0]]}, (x, y) = (${bf[1]}, ${bf[2]}), ${bf[3]} × ${bf[4]} (ağırlıklı hata ${fmt(100 * best.err, 3)}%). T = ${q.T} ile hata eğitimde ${fmt(100 * trE[q.T - 1], 3)}%, testte ${fmt(100 * teE[q.T - 1], 3)}%.`];
    Pn.push(`Kaskad: ${stagesC.length} aşama, öznitelik sayıları ${Ksz.join(", ")} (toplam ${tot}); her aşamanın eğitim negatifleri önceki aşamaları geçenlerden. Eğitim tespit oranı ${fmt(Dtr, 3)} (≥ d^${stagesC.length} = ${fmt(dPow, 3)}), test ${fmt(C.D, 3)}${C.D < dPow ? ": eğitimdeki alt sınır teste taşınmadı" : ""}. Test yanlış pozitif oranı ${fmt(C.F, 4)} (2000 negatif). Negatif pencere ortalama ${fmt(C.featN, 3)}, yüz ${fmt(C.featP, 3)} öznitelik hesaplatıyor.`);
    if (stagesC.some((st) => st.k === 12 && st.fr > q.f)) Pn.push("Bazı aşamalar 12 öznitelikle hedef f'ye inemedi: kalan negatifler yüzlere çok benziyor.");
    return {
      plots: [
        { title: "Tek AdaBoost: güçlü sınıflandırıcının hatası", xlabel: "tur sayısı T", ylabel: "hata", side: true, size: { w: 300, h: 240 }, series: [{ name: "eğitim", x: Ks, y: trE, c: 0 }, { name: "test", x: Ks, y: teE, c: 1, dash: true }], marks: [{ x: q.T, label: "seçili T" }] },
        { title: "Kaskad: aşamadan sonra kalan oran (test)", xlabel: "aşama (birikimli öznitelik)", ylabel: "kalan oran", side: true, ylog: true, size: { w: 300, h: 240 }, xticks: Ksz.map((k, s) => [s + 1, String(Ksz.slice(0, s + 1).reduce((x, y) => x + y, 0))]), series: [{ name: "yüz", x: range(Ksz.length + 1), y: [1, ...C.remP].map((v) => Math.max(v, 1e-4)), c: 0 }, { name: "yüz olmayan", x: range(Ksz.length + 1), y: [1, ...C.remN].map((v) => Math.max(v, 1e-4)), c: 1 }] },
      ],
      stats: [{ label: "Eğitim / test hatası (T)", value: `${fmt(100 * trE[q.T - 1], 3)}% / ${fmt(100 * teE[q.T - 1], 3)}%` }, { label: "Aşama öznitelik sayıları", value: Ksz.join(", ") }, { label: "Kaskad tespit: eğitim / test", value: `${fmt(Dtr, 3)} / ${fmt(C.D, 3)}` }, { label: "Kaskad yanlış pozitif (test)", value: fmt(C.F, 4) }, { label: "Ortalama öznitelik: negatif / pozitif pencere", value: `${fmt(C.featN, 3)} / ${fmt(C.featP, 3)}` }],
      note: Pn.join(" "),
    };
  },
};
