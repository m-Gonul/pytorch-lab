// M10.4 — Warp ayrışması (divergence): bir bloğun thread'leri 32'lik warp'lara bölünür; bir warp'taki thread'ler bir dalda farklı yollara
// giderse yollar sırayla yürütülür (yolda olmayan thread'ler maskelenir). Desenler: if (tid % 2), if (tid < k), warp hizalı
// if ((tid / 32) % 2), veriye bağlı if (rastgele, olasılık p), değişken döngü for (i < tid % L) (warp en uzun döngüyü bekler).
// Maliyet: then yolu A, else yolu B komut; warp maliyeti = yürütülen yolların toplamı (döngüde en uzun turun A katı). SIMT verimi =
// yararlı thread·komut / (32 × warp'ın yürüttüğü komut).
import { fmt, range } from "./plot.js";

const PAT = ["if (tid % 2 == 0)", "if (tid < k)", "if ((tid / 32) % 2 == 0)  (warp hizalı)", "veriye bağlı: if (x[tid] > eşik), olasılık p", "for (i = 0; i < tid % L; i++)"];
export default {
  params: [
    { id: "pat", label: "Dal deseni", options: PAT, value: 0 },
    { id: "T", label: "Blok başına thread", min: 32, max: 1024, step: 32, value: 256, int: true },
    { id: "A", label: "then yolu (komut)", min: 1, max: 100, value: 20, int: true },
    { id: "B", label: "else yolu (komut)", min: 0, max: 100, value: 20, int: true, on: { pat: [0, 1, 2, 3] } },
    { id: "k", label: "Eşik k", min: 0, max: 1024, value: 100, int: true, on: { pat: [1] } },
    { id: "p", label: "Koşulun doğru olma olasılığı p", min: 0, max: 1, step: 0.01, value: 0.5, on: { pat: [3] } },
    { id: "L", label: "Döngü periyodu L", min: 1, max: 64, value: 32, int: true, on: { pat: [4] } },
  ],
  seed: true,
  seedOn: { pat: 3, p: 0.05 },
  run(q, r) {
    const W = Math.ceil(q.T / 32), cost = [], useful = [], split = [], cond = (t) => (q.pat === 0 ? t % 2 === 0 : q.pat === 1 ? t < q.k : q.pat === 2 ? Math.floor(t / 32) % 2 === 0 : r() < q.p);
    for (let w = 0; w < W; w++) {
      const lanes = range(32).map((l) => w * 32 + l).filter((t) => t < q.T);
      if (q.pat === 4) { const it = lanes.map((t) => t % q.L), mx = Math.max(...it); cost.push(mx * q.A); useful.push(it.reduce((a, b) => a + b, 0) * q.A); continue; }
      const c = lanes.map(cond), nT = c.filter(Boolean).length, nF = c.length - nT;
      cost.push((nT ? q.A : 0) + (nF ? q.B : 0)); useful.push(nT * q.A + nF * q.B); split.push(nT > 0 && nF > 0);
    }
    const tot = cost.reduce((a, b) => a + b, 0), use = useful.reduce((a, b) => a + b, 0), eff = tot ? use / (32 * tot) : 1, div = split.filter(Boolean).length;
    const P = [`${W} warp; ${q.pat === 4 ? "warp'lar en uzun döngüyü bekliyor" : `${div} warp ayrışıyor`}. SIMT verimi ${fmt(100 * eff, 3)}% (yürütülen ${tot} warp-komutunun yararlı kısmı).`];
    if (q.pat === 0) P.push(q.B === 0 ? "else yolu boş, ama then yolunda her warp'ın thread'lerinin yarısı maskeli bekliyor: verim yine %50. Koşulu warp'a hizalamak (tid / 32) ayrışmayı yok eder." : "Her warp'ta çift ve tek thread'ler var: bütün warp'lar iki yolu da sırayla yürütüyor. Koşulu warp'a hizalamak (tid / 32) ayrışmayı yok eder.");
    if (q.pat === 1) P.push(`Yalnız k'yi içeren warp ayrışıyor (k = ${q.k}${q.k % 32 === 0 ? ": 32'nin katı, hiç ayrışma yok" : ""}).`);
    if (q.pat === 3) P.push("Veriye bağlı koşulda neredeyse her warp iki yolu da yürütür (p 0 ya da 1'e yakın değilse); veriyi önce sıralamak ya da gruplamak warp'ları tek tipleştirir.");
    if (q.pat === 4) P.push("Döngü sayısı thread'e göre değişince warp en uzun thread kadar döner; işi thread'ler arasında dengelemek ya da döngü sayısını warp içinde sabit tutmak (örneğin tid / 32'ye bağlamak) verimi artırır.");
    return {
      plots: [{ title: "Warp başına yürütülen ve yararlı komut", xlabel: "warp", ylabel: "komut", wide: true, size: { w: 620, h: 210 }, series: [{ name: "yürütülen (warp)", x: range(W), y: cost, kind: "bar", c: 1 }, { name: "yararlı / 32", x: range(W), y: useful.map((u) => u / 32), kind: "bar", c: 0, bw: 0.4 }] }],
      stats: [{ label: "SIMT verimi", value: `${fmt(100 * eff, 3)}%` }, { label: "Ayrışan warp", value: q.pat === 4 ? "—" : `${div} / ${W}` }, { label: "Toplam warp-komutu", value: tot }],
      note: P.join(" "),
    };
  },
};
