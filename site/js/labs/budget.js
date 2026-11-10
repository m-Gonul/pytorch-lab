// M8.1 — Hesap bütçesi. Eğitim FLOP'u C ≈ 6ND (N parametre, D token). Chinchilla (Hoffmann ve ark. 2022) uydurulmuş kayıp yasası
// (yaklaşım 3): L(N, D) = E + A/N^α + B/D^β, E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28. Sabit C'de D = C/(6N); eş-FLOP
// eğrisi L(N) ve en iyi N*. Kural olarak D ≈ 20N (yaklaşım 1–2). Süre = C / (tepe · MFU).
import { fmt, range } from "./plot.js";

const E = 1.69, A = 406.4, B = 410.7, al = 0.34, be = 0.28;
export const loss = (N, D) => E + A / N ** al + B / D ** be;
export default {
  params: [
    { id: "logN", label: "Parametre sayısı: 10^a, a", min: 6, max: 12, step: 0.05, value: 7 },
    { id: "ratio", label: "Token / parametre oranı D/N", min: 1, max: 1000, value: 20, log: true },
    { id: "peak", label: "Tepe hız (TFLOP/s)", min: 1, max: 2000, value: 100, log: true },
    { id: "mfu", label: "MFU (tepenin kullanılan oranı)", min: 0.05, max: 0.8, step: 0.01, value: 0.4 },
  ],
  run(q) {
    const N = 10 ** q.logN, D = q.ratio * N, C = 6 * N * D, secs = C / (q.peak * 1e12 * q.mfu), L = loss(N, D);
    const Ns = range(121).map((i) => 10 ** (q.logN - 3 + i * 0.05)), iso = Ns.map((n) => loss(n, C / (6 * n))), bi = iso.indexOf(Math.min(...iso)), Nopt = Ns[bi], Dopt = C / (6 * Nopt);
    const N20 = Math.sqrt(C / 120), L20 = loss(N20, 20 * N20);
    const hms = secs < 3600 ? `${fmt(secs, 3)} s` : secs < 86400 * 2 ? `${fmt(secs / 3600, 3)} saat` : `${fmt(secs / 86400, 3)} gün`;
    const P = [`N = ${fmt(N, 3)}, D = ${fmt(D, 3)} token (D/N = ${q.ratio}): C = 6ND = ${fmt(C, 3)} FLOP. ${q.peak} TFLOP/s ve %${fmt(100 * q.mfu, 3)} MFU ile süre ${hms}. Uydurulmuş yasaya göre kayıp ${fmt(L, 4)} nat/token.`];
    P.push(`Aynı C ile en iyi bölüşüm (eş-FLOP eğrisinin dibi): N* = ${fmt(Nopt, 3)}, D* = ${fmt(Dopt, 3)} (D/N = ${fmt(Dopt / Nopt, 3)}), kayıp ${fmt(iso[bi], 4)}. "20 token/parametre" kuralı N = ${fmt(N20, 3)} verir, kayıp ${fmt(L20, 4)}. Bu bütçede yaklaşım 3'ün katsayıları kurala göre daha ${Nopt > N20 ? "büyük" : "küçük"} bir model öneriyor: yaklaşım 3'te en iyi D/N oranı hesapla büyür (makale Gopher bütçesi için yaklaşım 1–2'de ≈ 63–67 milyar, yaklaşım 3'te 40 milyar parametre bildirir). Kayıp eğrisi dipte düz olduğu için iki seçimin kayıp farkı küçüktür.`);
    if (q.ratio > 100) P.push("Çok fazla token/parametre: hesap bütçesine göre küçük bir model uzun eğitiliyor. Eğitim hesabı açısından optimal değil ama çıkarımı ucuz bir model verir; LLaMA gibi modeller bilerek bu bölgede eğitilir.");
    if (q.ratio < 5) P.push("Çok az token/parametre: büyük bir model az veriyle eğitiliyor; aynı hesapla daha küçük bir model daha düşük kayıp verirdi (Kaplan ve ark.'nın ölçekleme önerisinin Chinchilla'da düzeltilen yönü).");
    return {
      plots: [{ title: "Eş-FLOP eğrisi (kesikli çizgiler: seçili N ve en iyi N*)", xlabel: "log₁₀ N", ylabel: "kayıp (nat/token)", wide: true, size: { w: 620, h: 240 }, series: [{ name: `C = ${fmt(C, 2)} FLOP`, x: Ns.map(Math.log10), y: iso, c: 0 }],
        marks: [{ x: q.logN, label: "N" }, { x: Math.log10(Nopt), label: "" }] }],
      stats: [
        { label: "C = 6ND (FLOP)", value: C },
        { label: "Süre (s)", value: secs },
        { label: "N*, D* (eş-FLOP en iyisi)", value: `${fmt(Nopt, 3)}, ${fmt(Dopt, 3)}` },
        { label: "Kayıp: seçili / en iyi", value: `${fmt(L, 4)} / ${fmt(iso[bi], 4)}` },
      ],
      note: P.join(" "),
    };
  },
};
