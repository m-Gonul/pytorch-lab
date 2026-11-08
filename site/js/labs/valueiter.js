// M11.2 — 5×5 ızgara dünyasında değer iterasyonu: büzülme γ^k ve kayma olasılığı
import { fmt, range } from "./plot.js";
const N = 5, GOAL = 4, PIT = 9, WALLS = [12, 17];
const MOV = [[0, 1], [1, 0], [0, -1], [-1, 0]]; // →, ↓, ←, ↑ (satır, sütun değil: [dx, dy] olarak x sütun)
const ARR = ["→", "↓", "←", "↑"];
export default {
  params: [
    { id: "g", label: "İndirim γ", min: 0, max: 0.99, step: 0.01, value: 0.9, hardMax: 0.999, why: "γ = 1 sonsuz ufukta toplamı sonsuz yapabilir; büzülme garantisi kalkar." },
    { id: "slip", label: "Kayma olasılığı (yana sapma)", min: 0, max: 0.6, step: 0.01, value: 0.2 },
    { id: "step", label: "Adım ödülü", min: -1, max: 0.2, step: 0.01, value: -0.04 },
    { id: "k", label: "İterasyon k", min: 0, max: 80, step: 1, value: 10, int: true },
  ],
  run(p) {
    const S = N * N, term = (s) => s === GOAL || s === PIT;
    const nxt = (s, a) => { const x = s % N, y = Math.floor(s / N), nx = x + MOV[a][0], ny = y + MOV[a][1]; const t = ny * N + nx; return nx < 0 || ny < 0 || nx >= N || ny >= N || WALLS.includes(t) ? s : t; };
    const Q = (V, s, a) => { let q = 0; for (const [b, pr] of [[a, 1 - p.slip], [(a + 1) % 4, p.slip / 2], [(a + 3) % 4, p.slip / 2]]) { const t = nxt(s, b); const r = t === GOAL ? 1 : t === PIT ? -1 : p.step; q += pr * (r + (term(t) ? 0 : p.g * V[t])); } return q; };
    let V = new Array(S).fill(0); const dif = [], Vs = [V];
    for (let k = 0; k < 80; k++) { const W = V.map((_, s) => (term(s) || WALLS.includes(s) ? 0 : Math.max(...range(4).map((a) => Q(V, s, a))))); dif.push(Math.max(...W.map((v, i) => Math.abs(v - V[i])))); V = W; Vs.push(V); }
    const Vk = Vs[p.k];
    const pol = range(S).map((s) => (term(s) ? (s === GOAL ? "★" : "✕") : WALLS.includes(s) ? "■" : ARR[range(4).reduce((b, a) => (Q(Vk, s, a) > Q(Vk, s, b) ? a : b), 0)]));
    const grid = range(N).map((y) => range(N).map((x) => pol[y * N + x]).join(" ")).join("\n");
    const z = []; for (let y = N - 1; y >= 0; y--) for (let x = 0; x < N; x++) z.push(-(Vk[y * N + x]));
    const bound = (p.g ** p.k / (1 - p.g)) * Math.max(1, Math.abs(p.step));
    return {
      plots: [
        { title: "‖V_{k+1} − V_k‖∞ (log)", xlabel: "iterasyon k", ylabel: "fark", ylog: true, series: [{ name: "gerçek fark", x: range(80), y: dif.map((v) => Math.max(v, 1e-16)) }, { name: "γ^k · ‖V₁ − V₀‖∞", x: range(80), y: range(80).map((k) => Math.max(p.g ** k * dif[0], 1e-16)), c: 2, dash: true }], marks: [{ x: p.k, label: "k" }] },
        { title: `V_k (k = ${p.k}; koyu = yüksek değer)`, xlabel: "sol alt: başlangıç · sağ üst: hedef ★ · altındaki: çukur ✕", ylabel: "", field: { x0: 0, x1: N, y0: 0, y1: N, nx: N, ny: N, z }, xlim: [0, N], ylim: [0, N], series: [] },
      ],
      stats: [{ label: "‖V_{k+1} − V_k‖∞", value: dif[p.k] ?? 0 }, { label: "Başlangıç durumu (sol alt) V_k", value: Vk[20] }, { label: "Kalan hata üst sınırı γ^k R_max/(1−γ)", value: bound }, { label: "Hedefe ε = 10⁻⁶ için gereken k ≈ ln ε / ln γ", value: p.g > 0 ? Math.ceil(Math.log(1e-6) / Math.log(p.g)) : 1 }],
      note: `Greedy politika (k = ${p.k}):<pre class="lab-grid">${grid}</pre>Bellman operatörü ∞-normda γ-büzülme: fark en kötü durumda γ^k ile söner (log grafikte doğru). Kayma arttıkça politika çukurdan (✕) uzak, daha uzun ama güvenli yolu seçer; adım ödülü çok negatifse risk almaya başlar.`,
    };
  },
};
