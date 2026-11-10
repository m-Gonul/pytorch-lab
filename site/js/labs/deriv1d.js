// M9.4 — Birinci ve ikinci türev ayrık farklarla: f'(x) = f(x+1) − f(x), f''(x) = f(x+1) + f(x−1) − 2f(x). Profil: düz bölge, aşağı rampa,
// düz, ince çizgi (1 piksel), düz, basamak, tek nokta. Gürültü eklenebilir ve türevden önce σ'lık Gauss ile yumuşatılabilir.
import { fmt, range } from "./plot.js";

export function profile(kind) {
  if (kind === 0) return [5, 5, 4, 3, 2, 1, 0, 0, 0, 6, 0, 0];  // modülün el hesabı
  const f = [];
  for (let i = 0; i < 80; i++) f.push(i < 10 ? 6 : i < 20 ? 6 - 0.6 * (i - 10) : i < 35 ? 0 : i === 35 ? 5 : i < 55 ? 0 : i < 70 ? 4 : i === 72 ? 2 : 4);
  return f;
}
const gsm = (f, s) => { if (s <= 0) return f; const R = Math.ceil(3 * s), w = range(2 * R + 1).map((j) => Math.exp(-0.5 * ((j - R) / s) ** 2)), S = w.reduce((a, b) => a + b, 0); return f.map((_, i) => w.reduce((a, v, j) => a + v * f[Math.max(0, Math.min(f.length - 1, i - R + j))], 0) / S); };
export default {
  params: [
    { id: "kind", label: "Profil", options: ["el hesabı (12 örnek)", "rampa, ince çizgi, basamak, tek nokta (80 örnek)"], value: 1 },
    { id: "noise", label: "Gürültü σ", min: 0, max: 1, step: 0.02, value: 0 },
    { id: "s", label: "Türevden önce Gauss σ (0 = yok)", min: 0, max: 4, step: 0.1, value: 0 },
  ],
  seed: true,
  seedOn: { noise: 0.1 },
  run(q, r) {
    const f0 = profile(q.kind), f = gsm(f0.map((v) => v + q.noise * r.normal()), q.s), n = f.length;
    const d1 = range(n - 1).map((i) => f[i + 1] - f[i]), d2 = range(n).map((i) => (i === 0 || i === n - 1 ? 0 : f[i + 1] + f[i - 1] - 2 * f[i]));
    const M2 = Math.max(...d2.map(Math.abs)), zc = range(n - 2).filter((i) => i > 0 && d2[i] * d2[i + 1] < 0 && Math.abs(d2[i] - d2[i + 1]) > 0.1 * M2).length;
    const P = [];
    if (q.kind === 0) P.push(`El hesabı: rampa içinde f'(3) = ${fmt(d1[3], 3)}, f''(3) = ${fmt(d2[3], 3)}; rampa başında f''(1) = ${fmt(d2[1], 3)}; ince çizgide f''(9) = ${fmt(d2[9], 3)} (birinci türev çizginin iki yanında +6 ve −6).`);
    else {
      const c0 = (v) => (Math.abs(v) < 1e-9 ? 0 : v), m1 = (a, b) => c0(Math.max(...d1.slice(a, b).map(Math.abs))), m2 = (a, b) => c0(Math.max(...d2.slice(a, b).map(Math.abs))), rl = m2(33, 38) / Math.max(m1(33, 37), 1e-12);
      P.push(`En büyük |f'|: rampada ${fmt(m1(10, 20), 3)}, ince çizgide ${fmt(m1(33, 37), 3)}, basamakta ${fmt(m1(53, 57), 3)}. En büyük |f''|: rampada ${fmt(m2(11, 19), 3)} ${q.noise > 0 || q.s > 0 ? "(gürültüsüz ve yumuşatmasız durumda 0)" : ""}, uçlarında ${fmt(Math.max(m2(9, 11), m2(19, 22)), 3)}, ince çizgide ${fmt(m2(33, 38), 3)}, basamakta ${fmt(m2(53, 57), 3)}.`);
      P.push(`İnce çizgide |f''| / |f'| = ${fmt(rl, 3)}. ` + (q.s === 0 ? "İkinci türev rampa boyunca sıfırdır, yalnız rampanın başında ve sonunda yanıt verir (çift kenar). İnce çizgi ve tek noktada ikinci türevin yanıtı birinci türevinkinin iki katıdır: ince ayrıntıyı daha çok vurgular, bu yüzden keskinleştirmede tercih edilir." : "Yumuşatma ince çizgiyi yaydığı için ikinci türevin ince ayrıntıdaki üstünlüğü kayboluyor: σ büyüdükçe oran 2'nin altına düşer."));
    }
    if (q.noise > 0) P.push(`Gürültü σ = ${q.noise}: türevler gürültüyü yükseltir; bağımsız gürültüde f'nin std'si √2σ, f''nin √6σ ≈ ${fmt(Math.sqrt(6) * q.noise, 3)}. ${q.s > 0 ? "Türevden önce yumuşatmak gürültüyü bastırır ama kenarları yayar (M9.6'daki Gauss türevi ve LoG'un fikri)." : "Yumuşatmadan ikinci türev neredeyse kullanılamaz hâle gelir."} Sıfır geçişi sayısı: ${zc}.`);
    return {
      plots: [
        { title: "Profil f", xlabel: "x", ylabel: "f", wide: true, size: { w: 620, h: 150 }, series: [{ name: "f", x: range(n), y: f, c: 4 }] },
        { title: "Birinci ve ikinci türev", xlabel: "x", ylabel: "değer", wide: true, size: { w: 620, h: 190 }, series: [{ name: "f′", x: range(n - 1).map((i) => i + 0.5), y: d1, c: 0 }, { name: "f″", x: range(n), y: d2, c: 1 }] },
      ],
      stats: [{ label: "f″'nün güçlü sıfır geçişleri", value: zc }, { label: "max |f′| / max |f″|", value: `${fmt(Math.max(...d1.map(Math.abs)), 3)} / ${fmt(Math.max(...d2.map(Math.abs)), 3)}` }],
      note: P.join(" "),
    };
  },
};
