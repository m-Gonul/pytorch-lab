// M2.6 — PCA'nın iki eşdeğer tanımı: izdüşüm varyansını en büyüklemek = yeniden kurma hatasını en küçüklemek.
// Bir yön θ seç; izdüşümler, varyans v(θ) = uᵀCu ve hata = toplam varyans − v(θ) canlı.
import { fmt, linspace } from "./plot.js";

export function eig2(C) { const a = C[0][0], b = C[0][1], d = C[1][1], tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0)); const l1 = tr / 2 + disc, l2 = tr / 2 - disc; const v = Math.abs(b) > 1e-12 ? [l1 - d, b] : a >= d ? [1, 0] : [0, 1]; const n = Math.hypot(...v); return { l1, l2, v1: [v[0] / n, v[1] / n] }; }

export default {
  params: [
    { id: "s1", label: "Uzun eksen std", min: 0.2, max: 3, step: 0.05, value: 2 },
    { id: "s2", label: "Kısa eksen std", min: 0.05, max: 3, step: 0.05, value: 0.6 },
    { id: "rot", label: "Verinin açısı (derece)", min: 0, max: 180, step: 1, value: 30 },
    { id: "th", label: "Seçtiğin izdüşüm yönü θ (derece)", min: 0, max: 180, step: 1, value: 80 },
    { id: "n", label: "Örnek sayısı", min: 10, max: 400, value: 120, int: true },
    { id: "out", label: "Aykırı nokta (x₂ = 6'da)", options: ["yok", "1 tane"], value: 0 },
  ],
  seed: true,
  run(q, r) {
    const a = (q.rot * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), n = Math.round(q.n);
    let X = Array.from({ length: n }, () => { const u = q.s1 * r.normal(), v = q.s2 * r.normal(); return [c * u - s * v, s * u + c * v]; });
    if (q.out === 1) X.push([0, 6]);
    const m = [0, 1].map((j) => X.reduce((t, p) => t + p[j], 0) / X.length); X = X.map((p) => [p[0] - m[0], p[1] - m[1]]);
    const N = X.length, C = [[0, 0], [0, 0]]; for (const p of X) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) C[i][j] += (p[i] * p[j]) / (N - 1);
    const E = eig2(C), tot = C[0][0] + C[1][1];
    const t = (q.th * Math.PI) / 180, u = [Math.cos(t), Math.sin(t)];
    const vth = u[0] * u[0] * C[0][0] + 2 * u[0] * u[1] * C[0][1] + u[1] * u[1] * C[1][1];
    const proj = X.map((p) => { const z = p[0] * u[0] + p[1] * u[1]; return [z * u[0], z * u[1]]; });
    const segx = [], segy = []; X.forEach((p, i) => { if (i % 2 === 0) { segx.push(p[0], proj[i][0], NaN); segy.push(p[1], proj[i][1], NaN); } });
    const L = 5, pcAng = ((Math.atan2(E.v1[1], E.v1[0]) * 180) / Math.PI + 180) % 180;
    const ths = linspace(0, 180, 181), vv = ths.map((d) => { const w = [Math.cos((d * Math.PI) / 180), Math.sin((d * Math.PI) / 180)]; return w[0] * w[0] * C[0][0] + 2 * w[0] * w[1] * C[0][1] + w[1] * w[1] * C[1][1]; });
    return {
      plots: [
        { title: "Veri (ortalanmış), seçilen yön ve izdüşümler", xlabel: "x₁", ylabel: "x₂", size: { w: 380, h: 370 }, xlim: [-L, L], ylim: [-L, L], series: [
          { name: "izdüşüm artıkları (her 2 noktadan 1)", x: segx, y: segy, c: 4 },
          { name: "veri", x: X.map((p) => p[0]), y: X.map((p) => p[1]), kind: "scatter", c: 0, r: 2.4 },
          { name: `seçilen yön θ = ${q.th}°`, x: [-L * u[0], L * u[0]], y: [-L * u[1], L * u[1]], c: 2 },
          { name: "izdüşümler", x: proj.map((p) => p[0]), y: proj.map((p) => p[1]), kind: "scatter", c: 2, r: 1.8 },
          { name: "1. temel bileşen", x: [0, 2 * Math.sqrt(E.l1) * E.v1[0]], y: [0, 2 * Math.sqrt(E.l1) * E.v1[1]], c: 1 },
          { name: "", x: [0, -2 * Math.sqrt(E.l1) * E.v1[0]], y: [0, -2 * Math.sqrt(E.l1) * E.v1[1]], c: 1 },
        ] },
        { title: "Yöne göre korunan varyans ve kayıp (yeniden kurma hatası)", xlabel: "θ (derece)", ylabel: "varyans", yzero: true, series: [
          { name: "korunan: uᵀCu", x: ths, y: vv, c: 0 },
          { name: "kayıp: toplam − uᵀCu", x: ths, y: vv.map((v) => tot - v), c: 2, dash: true },
          { name: "seçili θ", x: [q.th, q.th], y: [vth, tot - vth], kind: "scatter", c: 3, r: 4.5 },
        ], marks: [{ x: pcAng, label: "PC1" }] },
      ],
      stats: [
        { label: "Kovaryans C", value: `[[${fmt(C[0][0], 3)}, ${fmt(C[0][1], 3)}], [${fmt(C[1][0], 3)}, ${fmt(C[1][1], 3)}]]` },
        { label: "Özdeğerler λ₁, λ₂", value: `${fmt(E.l1, 3)}, ${fmt(E.l2, 3)}` },
        { label: "PC1 açısı", value: `${fmt(pcAng, 3)}°` },
        { label: "θ'da korunan varyans / toplam", value: `${fmt(vth, 3)} / ${fmt(tot, 3)} (%${fmt((100 * vth) / tot, 3)})` },
        { label: "PC1'in açıkladığı oran λ₁/(λ₁ + λ₂)", value: `%${fmt((100 * E.l1) / tot, 3)}` },
      ],
      note: `Her nokta için Pisagor: ‖x‖² = (izdüşüm)² + (artık)². Toplam varyans sabit olduğu için izdüşüm varyansını <b>en büyüklemek</b> ile kesikli artıkların kare toplamını (yeniden kurma hatası) <b>en küçüklemek</b> aynı problemdir: iki eğri birbirinin aynası ve tepe/dip noktası aynı açıda, PC1'de. Korunan varyans θ'nın fonksiyonu olarak λ₁cos²(θ − θ₁) + λ₂sin²(θ − θ₁) biçimindedir. ${q.out === 1 ? "<b>Aykırı nokta:</b> tek bir uzak nokta kare uzaklıklar yüzünden kovaryansı ve PC1'i kendine doğru çeker — PCA aykırılara duyarlıdır (dayanıklı PCA yöntemleri vardır)." : ""}${Math.abs(q.s1 - q.s2) < 0.1 ? " İki std neredeyse eşit: λ₁ ≈ λ₂, tüm yönler aynı varyanslı ve PC1 kararsız (her tohumda başka yöne döner)." : ""}`,
    };
  },
};
