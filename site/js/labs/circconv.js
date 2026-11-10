// M9.10 — Konvolüsyon teoremi ve sarmalanma. Uzunluğu A olan f (dikdörtgen + rampa) ile uzunluğu B olan h (kutu ya da Gauss),
// P noktalı DFT'lerin çarpımının tersiyle (dairesel konvolüsyon) ve doğrudan doğrusal konvolüsyonla karşılaştırılır.
// P ≥ A + B − 1 olunca ikisi aynıdır; daha küçük P'de kuyruk başa sarmalanır. Ortak FFT yardımcıları da burada.
import { fmt, range } from "./plot.js";

// Karmaşık diziler { re, im }; dft her uzunlukta (O(N²)), fft 2'nin kuvvetlerinde (yinelemeli Cooley–Tukey).
export function dft(re, im, inv = false) {
  const N = re.length, R = new Float64Array(N), I = new Float64Array(N), s = inv ? 1 : -1;
  for (let u = 0; u < N; u++) { let a = 0, b = 0; for (let x = 0; x < N; x++) { const t = (s * 2 * Math.PI * u * x) / N, c = Math.cos(t), si = Math.sin(t); a += re[x] * c - im[x] * si; b += re[x] * si + im[x] * c; } R[u] = inv ? a / N : a; I[u] = inv ? b / N : b; }
  return { re: R, im: I };
}
export function fft(re, im, inv = false) {
  const N = re.length, R = Float64Array.from(re), I = Float64Array.from(im);
  for (let i = 1, j = 0; i < N; i++) { let bit = N >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [R[i], R[j]] = [R[j], R[i]]; [I[i], I[j]] = [I[j], I[i]]; } }
  for (let len = 2; len <= N; len <<= 1) {
    const a = ((inv ? 2 : -2) * Math.PI) / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < N; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const p = i + k, q = p + len / 2, tr = R[q] * cr - I[q] * ci, ti = R[q] * ci + I[q] * cr; R[q] = R[p] - tr; I[q] = I[p] - ti; R[p] += tr; I[p] += ti; const n = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = n; } }
  }
  if (inv) for (let i = 0; i < N; i++) { R[i] /= N; I[i] /= N; }
  return { re: R, im: I };
}
export function fft2(re, im, n, inv = false) { // n × n, n 2'nin kuvveti
  const R = Float64Array.from(re), I = Float64Array.from(im);
  for (let y = 0; y < n; y++) { const o = fft(R.subarray(y * n, y * n + n), I.subarray(y * n, y * n + n), inv); R.set(o.re, y * n); I.set(o.im, y * n); }
  for (let x = 0; x < n; x++) { const cr = new Float64Array(n), ci = new Float64Array(n); for (let y = 0; y < n; y++) { cr[y] = R[y * n + x]; ci[y] = I[y * n + x]; } const o = fft(cr, ci, inv); for (let y = 0; y < n; y++) { R[y * n + x] = o.re[y]; I[y * n + x] = o.im[y]; } }
  return { re: R, im: I };
}
export default {
  params: [
    { id: "A", label: "f'nin uzunluğu A", min: 4, max: 40, value: 24, int: true },
    { id: "B", label: "h'nin uzunluğu B", min: 2, max: 30, value: 12, int: true },
    { id: "kind", label: "h", options: ["kutu (ortalama)", "Gauss"], value: 0 },
    { id: "P", label: "DFT uzunluğu P", min: 8, max: 80, value: 24, int: true },
  ],
  run(q) {
    const f = range(q.A).map((x) => (x < q.A / 2 ? 1 : 0.3 + (0.7 * (x - q.A / 2)) / (q.A / 2))), h0 = range(q.B).map((k) => (q.kind === 0 ? 1 : Math.exp(-0.5 * ((k - (q.B - 1) / 2) / (q.B / 5)) ** 2))), hs = h0.reduce((a, b) => a + b, 0), h = h0.map((v) => v / hs);
    const L = q.A + q.B - 1, lin = range(L).map((n) => h.reduce((s, v, k) => s + (n - k >= 0 && n - k < q.A ? v * f[n - k] : 0), 0));
    const pad = (a) => { const o = new Float64Array(q.P); a.forEach((v, i) => { o[i % q.P] += v; }); return o; };
    const F = dft(pad(f), new Float64Array(q.P)), H = dft(pad(h), new Float64Array(q.P));
    const G = dft(F.re.map((a, u) => a * H.re[u] - F.im[u] * H.im[u]), F.re.map((a, u) => a * H.im[u] + F.im[u] * H.re[u]), true), circ = Array.from(G.re);
    const cmp = range(Math.min(L, q.P)).map((n) => Math.abs(circ[n] - lin[n])), err = Math.max(...cmp);
    const ok = q.P >= L, folded = q.A > q.P || q.B > q.P;
    const P = [`Doğrusal konvolüsyonun uzunluğu A + B − 1 = ${L}; P = ${q.P}. ${ok ? `P ≥ ${L}: dairesel sonuç doğrusal sonuçla aynı (en büyük fark ${fmt(err, 3)})${q.P > L ? `; fazladan ${q.P - L} örnek sıfır kalır` : ""}.` : L - q.P <= q.P ? `P &lt; ${L}: doğrusal sonucun son ${L - q.P} örneği başa sarmalanır ve ilk ${L - q.P} örneğe eklenir (en büyük fark ${fmt(err, 3)}).` : `P &lt; ${L}: doğrusal sonuç P'den iki kat uzun; kuyruk birden çok kez sarmalanıp üst üste biner (en büyük fark ${fmt(err, 3)}).`}`];
    if (folded) P.push("<b>Dikkat:</b> P, dizilerden birinden bile kısa: dizi DFT'den önce P uzunluğuna katlanıyor (zamanda örtüşme).");
    P.push("Görüntülerde aynı kural her eksende geçerlidir: M × N görüntü ve m × n çekirdek için P ≥ M + m − 1, Q ≥ N + n − 1. Gonzalez çoğunlukla P = 2M, Q = 2N alır.");
    return {
      plots: [
        { title: "Doğrusal konvolüsyon ve P noktalı DFT ile dairesel konvolüsyon", xlabel: "n", ylabel: "değer", wide: true, size: { w: 620, h: 220 }, series: [{ name: "f", x: range(q.A), y: f, c: 4, dash: true }, { name: "doğrusal f ∗ h", x: range(L), y: lin, c: 0 }, { name: "IDFT(F·H), P noktalı", x: range(q.P), y: circ, c: 1, dash: true }], marks: [{ x: q.P - 0.5, label: "P" }, { x: L - 0.5, label: "A+B−1" }] },
        { title: "|F(u)| ve |H(u)| (P noktalı)", xlabel: "u", ylabel: "genlik", wide: true, size: { w: 620, h: 160 }, series: [{ name: "|F|", x: range(q.P), y: Array.from(F.re, (a, u) => Math.hypot(a, F.im[u])), kind: "bar", c: 0 }, { name: "|H|·max|F|", x: range(q.P), y: Array.from(H.re, (a, u) => Math.hypot(a, H.im[u]) * Math.hypot(F.re[0], F.im[0])), c: 1 }] },
      ],
      stats: [{ label: "A + B − 1", value: L }, { label: "En büyük fark |dairesel − doğrusal|", value: err }, { label: "Sarmalanma", value: ok ? "yok" : `${L - q.P} örnek` }],
      note: P.join(" "),
    };
  },
};
