// M6.1 — PyTorch autograd motorunun küçük bir benzetimi: a, b yaprak; c = a·b, d = c + a, L = d². Düğümler: PowBackward0 (d'yi saklar),
// AddBackward0 (hiçbir şey saklamaz), MulBackward0 (a ve b'yi saklar), AccumulateGrad(a), AccumulateGrad(b). Motor bağımlılık sayarak
// topolojik sırada çalışır; bir düğüme gelen gradyanlar toplanır. c üzerinde gradyanı h ile çarpan bir tensor hook'u, zero_grad olmadan
// k kez backward (retain_graph=True) ve sürüm sayacı (version counter) ile in-place denetimi. Sonuçlar PyTorch 2.x ile doğrulandı.
import { fmt, range } from "./plot.js";

const INPL = ["yok", "c.add_(1) (L'den sonra)", "d.add_(1) (L'den sonra)", "a.add_(1) (yaprak)"];
export function simulate(q) {
  const c = q.a * q.b, d = c + q.a, L = d * d;
  if (q.inpl === 2) return { err: "one of the variables needed for gradient computation has been modified by an inplace operation: … output 0 of AddBackward0, is at version 1; expected version 0 instead.", L, c, d };
  if (q.inpl === 3) return { err: "a leaf Variable that requires grad is being used in an in-place operation.", L, c, d };
  const gd = 2 * d, gc = q.h * gd, viaC = gc * q.b, direct = gd, gb1 = gc * q.a;
  const ga = range(q.calls).map((k) => (k + 1) * (viaC + direct)), gb = range(q.calls).map((k) => (k + 1) * gb1);
  return { L, c, d, gd, gc, viaC, direct, ga, gb };
}
export default {
  params: [
    { id: "a", label: "a", min: -3, max: 3, step: 0.5, value: 2 },
    { id: "b", label: "b", min: -3, max: 3, step: 0.5, value: 3 },
    { id: "h", label: "c'deki hook'un çarpanı h (1 = hook yok)", min: 0, max: 4, step: 0.5, value: 1 },
    { id: "calls", label: "zero_grad olmadan backward sayısı", min: 1, max: 5, value: 1, int: true },
    { id: "inpl", label: "In-place işlem", options: INPL, value: 0 },
  ],
  run(q) {
    const s = simulate(q);
    if (s.err) {
      const who = q.inpl === 2 ? "PowBackward0, türevi 2d için d'yi saklamıştı (sürüm 0). d.add_(1) sürümü 1'e çıkardı; geri geçişte saklanan sürüm ile şimdiki uyuşmuyor ve motor yanlış gradyan hesaplamak yerine hata verir." : "Gradyan isteyen bir yaprağı yerinde değiştirmek yasaktır: a'nın değeri grafın kurulduğu değerden farklı olur ve AccumulateGrad'ın biriktireceği gradyan hangi a'ya ait olduğu belirsizleşir. Parametre güncellemesi bu yüzden torch.no_grad() içinde yapılır.";
      return {
        plots: [{ title: "Gradyan hesaplanamadı", xlabel: "", ylabel: "", series: [{ name: "—", x: [0, 1], y: [0, 0], c: 4 }] }],
        stats: [{ label: "L = (ab + a)²", value: s.L }, { label: "Sonuç", value: "RuntimeError" }],
        note: `<b>RuntimeError:</b> ${s.err} ${who}`,
      };
    }
    const P = [`İleri geçiş: c = ${fmt(s.c, 3)}, d = ${fmt(s.d, 3)}, L = ${fmt(s.L, 3)}. Motor L'den başlar: PowBackward0 ∂L/∂d = 2d = ${fmt(s.gd, 3)} üretir, AddBackward0 bunu c'ye ve a'ya kopyalar, MulBackward0 c'den gelen ${fmt(s.gc, 3)}'i saklanan a ve b ile çarpıp dağıtır.`];
    P.push(`a'nın AccumulateGrad düğümüne iki gradyan gelir (Mul'dan ${fmt(s.viaC, 3)}, Add'den ${fmt(s.direct, 3)}). Motor bu düğümün bağımlılık sayısını 2 tutar ve ikisi de gelince toplayıp bir kez çalıştırır: a.grad = ${fmt(s.ga[0], 4)}, b.grad = ${fmt(s.gb[0], 4)}.`);
    if (q.h !== 1) P.push(`c'deki hook gradyanı ${q.h} ile çarpıyor; yalnız c'den geçen yol etkilenir, a'nın doğrudan yolu değişmez.`);
    if (q.calls > 1) P.push(`zero_grad yapılmadan ${q.calls} kez backward: AccumulateGrad .grad'a <b>ekler</b>, a.grad ${fmt(s.ga.at(-1), 4)} oldu. (İkinci çağrı için ilkinde retain_graph=True gerekir; yoksa saklanan tensörler serbest bırakılmış olur.)`);
    if (q.inpl === 1) P.push("c.add_(1) c'nin sürümünü 1'e çıkardı, ama c'yi saklayan bir düğüm yok (Mul a ve b'yi, Pow d'yi saklar, Add hiçbir şey saklamaz). Bu yüzden hata yok ve gradyanlar değişmedi: in-place işlem ancak değiştirilen tensör bir düğüm tarafından saklanmışsa sorun olur.");
    const ks = range(q.calls + 1);
    return {
      plots: [
        { title: "∂L/∂a'nın iki yolu (bir backward)", xlabel: "yol", ylabel: "katkı", yzero: true, xlim: [-0.6, 1.6], series: [{ name: "katkı", x: [0, 1], y: [s.viaC, s.direct], kind: "bar", c: 0 }], xticks: [[0, "c üzerinden"], [1, "doğrudan (Add)"]] },
        { title: "Birikim: her backward'dan sonra .grad", xlabel: "backward çağrısı", ylabel: ".grad", yzero: true, series: [
          { name: "a.grad", x: ks, y: [0, ...s.ga], c: 0 }, { name: "b.grad", x: ks, y: [0, ...s.gb], c: 1 }], xticks: ks.map((k) => [k, String(k)]) },
      ],
      stats: [
        { label: "L", value: s.L },
        { label: "a.grad / b.grad (son)", value: `${fmt(s.ga.at(-1), 4)} / ${fmt(s.gb.at(-1), 4)}` },
        { label: "Topolojik sıra", value: "Pow → Add → Mul → AccumulateGrad(a, b)" },
      ],
      note: P.join(" "),
    };
  },
};
