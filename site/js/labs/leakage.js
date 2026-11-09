// M3.9 — Veri sızıntısı, sayılarla: (a) grup sızıntısı — aynı "hastanın" örnekleri hem eğitimde hem testte (1-NN);
// (b) öznitelik seçimi sızıntısı — rastgele etiketlerde, seçimi CV'nin dışında yapmak. Rapor edilen doğruluk vs gerçek doğruluk.
import { fmt, range } from "./plot.js";

const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
const nn1 = (Xtr, ytr, x) => { let best = Infinity, y = 0; for (let i = 0; i < Xtr.length; i++) { const d = d2(Xtr[i], x); if (d < best) { best = d; y = ytr[i]; } } return y; };
const acc = (Xtr, ytr, Xte, yte) => Xte.reduce((s, x, i) => s + (nn1(Xtr, ytr, x) === yte[i] ? 1 : 0), 0) / Math.max(1, Xte.length);
const shuffle = (a, r) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

export function groups(G, m, sw, r) {
  const u = [1, 0.5], gen = (k) => { const out = []; for (let g = 0; g < k; g++) { const c = [r.normal(), r.normal()], y = c[0] * u[0] + c[1] * u[1] + 0.8 * r.normal() > 0 ? 1 : 0; for (let j = 0; j < m; j++) out.push({ g, x: [c[0] + sw * r.normal(), c[1] + sw * r.normal()], y }); } return out; };
  const S = gen(G), fresh = gen(300);
  // rastgele bölme (örnek düzeyinde), grup bölmesi (grup düzeyinde), yeni gruplarda gerçek doğruluk
  const idx = shuffle(range(S.length), r), cut = Math.round(0.8 * S.length), A = idx.slice(0, cut).map((i) => S[i]), B = idx.slice(cut).map((i) => S[i]);
  const gs = shuffle(range(G), r), trG = new Set(gs.slice(0, Math.round(0.8 * G))), C = S.filter((s) => trG.has(s.g)), D = S.filter((s) => !trG.has(s.g));
  const sib = B.filter((b) => A.some((a) => a.g === b.g)).length / Math.max(1, B.length);
  return { S, random: acc(A.map((s) => s.x), A.map((s) => s.y), B.map((s) => s.x), B.map((s) => s.y)), group: acc(C.map((s) => s.x), C.map((s) => s.y), D.map((s) => s.x), D.map((s) => s.y)), truth: acc(S.map((s) => s.x), S.map((s) => s.y), fresh.map((s) => s.x), fresh.map((s) => s.y)), sib };
}
// en yakın merkez sınıflandırıcı ve |korelasyon|'a göre en iyi k öznitelik
const topk = (X, y, k) => { const n = X.length, d = X[0].length, my = y.reduce((a, b) => a + b, 0) / n; const sc = range(d).map((j) => { let mx = 0; for (let i = 0; i < n; i++) mx += X[i][j]; mx /= n; let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (X[i][j] - mx) * (y[i] - my); sxx += (X[i][j] - mx) ** 2; } return Math.abs(sxy) / Math.sqrt(sxx + 1e-12); }); return range(d).sort((a, b) => sc[b] - sc[a]).slice(0, k); };
const centroid = (X, y, F, Xt) => { const c = [0, 1].map((cl) => { const R = X.filter((_, i) => y[i] === cl); return F.map((j) => R.reduce((s, r) => s + r[j], 0) / Math.max(1, R.length)); }); return Xt.map((x) => { const v = F.map((j) => x[j]); return d2(v, c[1]) < d2(v, c[0]) ? 1 : 0; }); };
export function featsel(n, d, k, r, K = 5) {
  const X = range(n).map(() => range(d).map(() => r.normal())), y = shuffle(range(n).map((i) => i % 2), r), folds = range(n).map((i) => i % K);
  const Fall = topk(X, y, k); let wrong = 0, right = 0;
  for (let f = 0; f < K; f++) {
    const tr = range(n).filter((i) => folds[i] !== f), te = range(n).filter((i) => folds[i] === f), Xtr = tr.map((i) => X[i]), ytr = tr.map((i) => y[i]), Xte = te.map((i) => X[i]), yte = te.map((i) => y[i]);
    const pw = centroid(Xtr, ytr, Fall, Xte), Fin = topk(Xtr, ytr, k), pr = centroid(Xtr, ytr, Fin, Xte);
    wrong += pw.filter((p, i) => p === yte[i]).length; right += pr.filter((p, i) => p === yte[i]).length;
  }
  return { wrong: wrong / n, right: right / n };
}

export default {
  params: [
    { id: "mode", label: "Sızıntı türü", options: ["grup sızıntısı (aynı hasta iki tarafta)", "öznitelik seçimi CV'nin dışında (rastgele etiketler)"], value: 0 },
    { id: "G", label: "Grup (hasta) sayısı", min: 10, max: 100, value: 40, int: true, on: { mode: [0] } },
    { id: "m", label: "Grup başına örnek", min: 1, max: 15, value: 6, int: true, on: { mode: [0] } },
    { id: "sw", label: "Grup içi yayılım σ_w", min: 0.02, max: 1.5, step: 0.02, value: 0.1, on: { mode: [0] } },
    { id: "n", label: "Örnek sayısı", min: 20, max: 100, value: 40, int: true, on: { mode: [1] } },
    { id: "d", label: "Öznitelik sayısı (hepsi saf gürültü)", min: 50, max: 3000, value: 1000, int: true, log: true, on: { mode: [1] } },
    { id: "k", label: "Seçilen öznitelik sayısı k", min: 1, max: 100, value: 20, int: true, log: true, on: { mode: [1] } },
  ],
  seed: true,
  run(q, r) {
    if (q.mode === 0) {
      const R = groups(Math.round(q.G), Math.round(q.m), q.sw, r), S = R.S;
      const sws = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5], sweep = sws.map((s) => groups(Math.round(q.G), Math.round(q.m), s, r));
      const cols = [0, 2];
      return {
        plots: [
          { title: "Örnekler (renk: etiket; aynı hastanın örnekleri kümelenir)", xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 360 }, series: [0, 1].map((c) => ({ name: `y = ${c}`, x: S.filter((s) => s.y === c).map((s) => s.x[0]), y: S.filter((s) => s.y === c).map((s) => s.x[1]), kind: "scatter", c: cols[c], r: 3 })) },
          { title: "Doğruluk: rapor edilen (iki bölme) ve gerçek (yeni hastalar)", xlabel: "grup içi yayılım σ_w", ylabel: "doğruluk", ylim: [0.3, 1.02], series: [{ name: "rastgele bölme", x: sws, y: sweep.map((s) => s.random), c: 5, dots: true }, { name: "grup bölmesi", x: sws, y: sweep.map((s) => s.group), c: 0, dots: true }, { name: "gerçek (yeni hastalar)", x: sws, y: sweep.map((s) => s.truth), c: 1, dash: true }], marks: [{ x: q.sw, label: "seçili" }] },
        ],
        stats: [
          { label: "Rastgele bölme ile rapor edilen", value: `%${fmt(R.random * 100, 3)}` },
          { label: "Grup bölmesi ile rapor edilen", value: `%${fmt(R.group * 100, 3)}` },
          { label: "Gerçek doğruluk (300 yeni hasta)", value: `%${fmt(R.truth * 100, 3)}` },
          { label: "Testte eğitimde kardeşi olan örnek oranı", value: `%${fmt(R.sib * 100, 3)}` },
        ],
        note: `Bir hastanın örnekleri birbirine çok benzer (σ_w küçük). Rastgele bölmede test örneklerinin %${fmt(R.sib * 100, 2)}'inin eğitimde bir "kardeşi" var; 1-NN onu bulup etiketini kopyalar: doğruluk etiketi tahmin etme yeteneğini değil, <b>hastayı tanıma</b> yeteneğini ölçer. Grup bölmesi (her hasta yalnız bir tarafta; scikit-learn <code>GroupKFold</code>) yeni hastadaki gerçek performansa yakın bir kestirim verir. ${Math.round(q.m) === 1 ? "Grup başına tek örnek: sızacak kardeş yok, iki bölme istatistiksel olarak eşdeğer (farkları yalnız küçük test kümesinin gürültüsü)." : ""} Not: grup bölmesinin test kümesi yalnız ≈ %20 hastadan oluşur; kestirimi gürültülüdür — pratikte GroupKFold ile tüm katların ortalaması alınır.`,
      };
    }
    const n = Math.round(q.n), d = Math.round(q.d), k = Math.min(Math.round(q.k), d), R = featsel(n, d, k, r), ks = [1, 2, 5, 10, 20, 50, 100].filter((v) => v <= d), sweep = ks.map((kk) => featsel(n, d, kk, r));
    return {
      plots: [
        { title: "5-katlı CV doğruluğu, seçilen öznitelik sayısına göre (etiketler rastgele!)", xlabel: "k (log)", ylabel: "CV doğruluğu", ylim: [0.2, 1.02], xticks: [1, 2, 5, 10, 20, 50, 100].map((v) => [Math.log10(v), String(v)]), series: [{ name: "yanlış: seçim tüm veride, sonra CV", x: ks.map(Math.log10), y: sweep.map((s) => s.wrong), c: 5, dots: true }, { name: "doğru: seçim her katın içinde", x: ks.map(Math.log10), y: sweep.map((s) => s.right), c: 0, dots: true }, { name: "gerçek: %50 (şans)", x: [0, 2], y: [0.5, 0.5], c: 1, dash: true }], marks: [{ x: Math.log10(k), label: "seçili k" }] },
      ],
      stats: [
        { label: "Yanlış protokolle CV doğruluğu", value: `%${fmt(R.wrong * 100, 3)}` },
        { label: "Doğru protokolle CV doğruluğu", value: `%${fmt(R.right * 100, 3)}` },
        { label: "Gerçek doğruluk", value: "%50 (etiketler öznitelikten bağımsız)" },
      ],
      note: `${d} saf gürültü öznitelik arasından, <b>tüm verinin</b> etiketleriyle en çok korelasyonlu ${k} tanesi seçilince, test katlarının etiketleri de seçime katılmış olur: şans eseri o etiketlerle ilişkili öznitelikler seçilir ve CV %${fmt(R.wrong * 100, 2)} gibi bir doğruluk "bulur". Seçim (ve her türlü veriye bakan ön işleme: ölçekleme istatistikleri, eksik değer doldurma, PCA) her katın <b>içinde</b>, yalnız eğitim kısmıyla yapılmalıdır (scikit-learn <code>Pipeline</code>). Öznitelik sayısı örnek sayısından çok büyükken sızıntı en şiddetlidir.`,
    };
  },
};
