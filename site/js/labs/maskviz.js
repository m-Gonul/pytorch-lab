// M4.14 — Dikkat maskeleri: kodlayıcı öz-dikkati (kaynak dolgusu), kod çözücü öz-dikkati (nedensel ∧ hedef dolgusu), çapraz
// dikkat (kaynak dolgusu), kayan pencere, önek-LM (ilk P konum çift yönlü), seyrek (stride). İzin matrisi A (satır: sorgu,
// sütun: anahtar) ve L katman sonra erişim (A ∨ I)^L: hangi girdi hangi çıktıyı etkileyebilir.
import { fmt, range } from "./plot.js";

export function mask(kind, q) {
  const T = q.T, S = kind === 2 ? q.S : q.T, tl = q.T - q.tpad, sl = q.S - q.spad;
  return range(T).map((i) => range(S).map((j) => {
    if (kind === 0) return i < sl && j < sl;            // kodlayıcı öz-dikkati (kaynak = T uzunlukta gösterilir)
    if (kind === 1) return j <= i && j < tl;            // kod çözücü öz-dikkati
    if (kind === 2) return j < sl;                      // çapraz dikkat
    if (kind === 3) return j <= i && i - j < q.w;       // kayan pencere
    if (kind === 4) return j < q.P ? true : j <= i && i >= q.P ? true : j <= i && i < q.P && j < q.P; // önek-LM
    return j <= i && (i - j < q.w || (i - j) % q.w === 0); // seyrek: yerel + stride
  }));
}
export function reach(A, L) {
  const n = A.length; let R = range(n).map((i) => range(n).map((j) => i === j || A[i][j]));
  const B = R.map((r) => [...r]);
  for (let l = 1; l < L; l++) R = R.map((r, i) => range(n).map((j) => r[j] || range(n).some((k) => r[k] && B[k][j])));
  return R;
}

export default {
  params: [
    { id: "kind", label: "Maske", options: ["kodlayıcı öz-dikkati (dolgu)", "kod çözücü öz-dikkati (nedensel + dolgu)", "çapraz dikkat (kaynak dolgusu)", "kayan pencere (nedensel, son w)", "önek-LM (ilk P çift yönlü)", "seyrek: yerel w + stride w"], value: 1 },
    { id: "T", label: "Hedef (sorgu) uzunluğu T", min: 2, max: 48, value: 12, int: true },
    { id: "S", label: "Kaynak uzunluğu S", min: 2, max: 48, value: 9, int: true, on: { kind: [2] } },
    { id: "tpad", label: "Hedef dolgusu (son kaç konum)", min: 0, max: 20, value: 2, int: true, on: { kind: [1] } },
    { id: "spad", label: "Kaynak dolgusu", min: 0, max: 20, value: 2, int: true, on: { kind: [0, 2] } },
    { id: "w", label: "Pencere / stride w", min: 1, max: 16, value: 3, int: true, on: { kind: [3, 5] } },
    { id: "P", label: "Önek uzunluğu P", min: 0, max: 24, value: 4, int: true, on: { kind: [4] } },
    { id: "L", label: "Katman sayısı L (erişim)", min: 1, max: 8, value: 2, int: true, on: { kind: [3, 5] } },
  ],
  validate(q) {
    const e = [];
    if (q.tpad >= q.T) e.push({ level: "error", msg: "Hedef dolgusu T'den küçük olmalı." });
    if (q.spad >= (q.kind === 2 ? q.S : q.T)) e.push({ level: "error", msg: "Kaynak dolgusu uzunluktan küçük olmalı." });
    if (q.P > q.T) e.push({ level: "warn", msg: "P > T: tüm dizi çift yönlü." });
    return e;
  },
  run(q) {
    const A = mask(q.kind, q), T = A.length, S = A[0].length, allowed = A.flat().filter(Boolean).length, sq = q.kind !== 2;
    const R = sq ? reach(A, q.L) : null, last = sq ? R[T - 1].filter(Boolean).length : 0, maxRow = Math.max(...A.map((r) => r.filter(Boolean).length));
    const P = [`${T} × ${S} skor matrisinde ${allowed} izinli (sorgu, anahtar) çifti (%${fmt((100 * allowed) / (T * S), 3)}); satır başına en fazla ${maxRow} anahtar. Yasak hücrelerin skoru −∞ yapılır: e^−∞ = 0, softmax'tan sonra ağırlık sıfırdır.`];
    if (q.kind === 1) P.push(`Nedensel maske tek başına T(T+1)/2 = ${(T * (T + 1)) / 2} çifte izin verir; hedef dolgusu (son ${q.tpad} konum) anahtar olarak da yasaklanır. Dolgu konumlarının sorgu satırları yine hesaplanır ama kayıpta maskelenir. Bu maske sayesinde öğretmen zorlamasıyla tüm hedef konumları tek geçişte, paralel eğitilir.`);
    if (q.kind === 0) P.push("Kodlayıcıda her kelime her kelimeye bakar (çift yönlü); yalnız dolgu anahtarları yasak. Dolgu satırlarının tamamen yasak olması softmax'ı 0/0 yapar: uygulamalar bu satırlara en az bir anahtar bırakır ya da çıktıyı sonradan sıfırlar.");
    if (q.kind === 2) P.push("Çapraz dikkatte sorgular kod çözücüden (T), anahtar ve değerler kodlayıcının son katmanından (S) gelir. Nedensellik yoktur: kaynak cümlenin tamamı baştan bilinir; yalnız kaynak dolgusu yasaktır.");
    if (q.kind === 3 || q.kind === 5) P.push(`Tek katmanda bir konum en fazla ${maxRow} anahtar görür; ${q.L} katman sonra son konumun çıktısını etkileyebilen girdi sayısı ${last} (sağdaki erişim matrisi, (A ∨ I)^L). Kayan pencerede erişim katman başına w − 1 büyür (Longformer, Mistral); stride eklemek (Child ve ark. 2019) az katmanda tüm diziye ulaşır.`);
    if (q.kind === 4) P.push(`İlk P = ${q.P} konum (önek, ör. talimat ya da kaynak cümle) kendi aralarında çift yönlü, sonrası nedensel: tek bir yığında kodlayıcı + kod çözücü davranışı (UniLM, T5'in prefix-LM varyantı).`);
    const img = (M) => ({ w: M[0].length, h: M.length, data: M.flat().map((x) => (x ? 1 : 0.12)) });
    return {
      plots: [
        { title: "İzin matrisi A (satır: sorgu, sütun: anahtar; açık = izinli)", xlabel: "anahtar", ylabel: "sorgu", size: { w: 310, h: 300 }, img: img(A), side: true, series: [] },
        ...(sq ? [{ title: `Erişim ${q.L} katman sonra: (A ∨ I)^L`, xlabel: "girdi konumu", ylabel: "çıktı konumu", size: { w: 310, h: 300 }, img: img(R), side: true, series: [] }] : []),
      ],
      stats: [
        { label: "İzinli çift", value: allowed },
        { label: "Oran", value: allowed / (T * S) },
        { label: "Satır başına en fazla anahtar", value: maxRow },
        ...(sq ? [{ label: `Son konumun ${q.L} katmanda eriştiği girdi`, value: last }] : []),
      ],
      note: P.join(" "),
    };
  },
};
