// M4.15 — Görüntüyü yamalara bölmek: H × W görüntü, P × P yamalar → N = HW/P² token. Yama ızgarası, konum bilgisi olmadan
// modelin gördüğü "yama torbası" (yamalar karıştırılmış), dizi uzunluğu ve öz-dikkat maliyetinin P ile değişimi.
import { fmt, range } from "./plot.js";

export function scene(H, r) {
  const img = range(H * H).map(() => 0.15), put = (x, y, v) => { if (x >= 0 && y >= 0 && x < H && y < H) img[y * H + x] = v; };
  const cx = H * (0.3 + 0.1 * r()), cy = H * (0.35 + 0.1 * r()), R = H * 0.18;
  for (let y = 0; y < H; y++) for (let x = 0; x < H; x++) {
    img[y * H + x] = 0.12 + 0.25 * (y / H);
    if ((x - cx) ** 2 + (y - cy) ** 2 < R * R) put(x, y, 0.95);
    if (x > H * 0.55 && x < H * 0.9 && y > H * 0.55 && y < H * 0.85) put(x, y, 0.6 + 0.3 * ((x + y) % 2));
    if (Math.abs(y - (H * 0.9 - 0.6 * x)) < H * 0.02) put(x, y, 0.8);
  }
  return img;
}
export function shuffle(img, H, P, r) {
  const g = H / P, order = range(g * g); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const out = new Array(H * H).fill(0);
  order.forEach((src, dst) => { const sy = Math.floor(src / g) * P, sx = (src % g) * P, dy = Math.floor(dst / g) * P, dx = (dst % g) * P; for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) out[(dy + y) * H + dx + x] = img[(sy + y) * H + sx + x]; });
  return out;
}

export default {
  params: [
    { id: "H", label: "Görüntü kenarı (piksel)", options: ["32", "64", "96", "224"], value: 1 },
    { id: "P", label: "Yama kenarı P", options: ["2", "4", "8", "16", "32"], value: 2 },
    { id: "D", label: "Gömme boyutu D", min: 16, max: 1280, value: 768, int: true, log: true },
    { id: "C", label: "Kanal sayısı C", min: 1, max: 4, value: 3, int: true },
    { id: "shuf", label: "Gösterim", options: ["yamalar yerinde", "karıştırılmış (konumsuz ViT'in gördüğü)"], value: 0 },
  ],
  seed: true,
  seedOn: { shuf: 1 },
  validate(q) { const H = +["32", "64", "96", "224"][q.H], P = +["2", "4", "8", "16", "32"][q.P]; return H % P ? [{ level: "error", msg: `${H}, P = ${P}'ye tam bölünmüyor; yamalara bölmek için H, P'nin katı olmalı (ya da görüntü kırpılır/doldurulur).` }] : []; },
  run(q, r) {
    const H = +["32", "64", "96", "224"][q.H], P = +["2", "4", "8", "16", "32"][q.P], g = H / P, N = g * g, T = N + 1;
    const base = scene(H, () => 0.5), img = q.shuf ? shuffle(base, H, P, r) : base;
    const lines = []; for (let k = 0; k <= g; k++) { lines.push({ x: [k * P, k * P], y: [0, H] }, { x: [0, H], y: [k * P, k * P] }); }
    const grid = { name: "yama sınırları", x: lines.flatMap((l) => [...l.x, NaN]), y: lines.flatMap((l) => [...l.y, NaN]), c: 2 };
    const peParams = P * P * q.C * q.D + q.D, attn = 2 * T * T * q.D, proj = 4 * T * q.D * q.D, mlp = 8 * T * q.D * q.D; // çarpma-toplama (MAC)
    const Ps = [2, 4, 8, 16, 32].filter((p) => H % p === 0), seq = Ps.map((p) => (H / p) ** 2 + 1);
    const P1 = [`${H} × ${H} görüntü, P = ${P}: ${g} × ${g} = ${N} yama, [class] ile dizi uzunluğu ${T}. Her yama ${P}·${P}·${q.C} = ${P * P * q.C} sayılık bir vektöre düzleşir ve ${q.D} boyuta izdüşürülür (yama embedding'i ${peParams.toLocaleString("tr-TR")} parametre). Bir katmanda (çarpma-toplama): dikkat skorları ve ağırlıklı toplam 2T²D = ${fmt(attn, 3)}, Q/K/V/O izdüşümleri 4TD² = ${fmt(proj, 3)}, MLP (4D genişlik) 8TD² = ${fmt(mlp, 3)}. Dikkat / MLP = T/(4D) = ${fmt(attn / mlp, 3)}${attn > mlp ? ": <b>dikkat terimi baskın</b>, dizi uzunluğu 4D'yi aşmış." : "."}`];
    P1.push(`P yarıya inerse N dört katına, dikkat maliyeti on altı katına çıkar (T² ∝ P⁻⁴). ${q.shuf ? "<b>Karıştırılmış yamalar:</b> konum embedding'i olmayan bir ViT için bu görüntü öncekiyle aynıdır: öz-dikkat ve MLP permütasyona eş-değişkendir, [class] çıktısı yamaların sırasından bağımsızdır (ek x2). Modelin elindeki tek uzamsal bilgi yama <i>içi</i> düzendir." : ""}`);
    return {
      plots: [
        { title: q.shuf ? "Yamalar karıştırılmış" : "Görüntü ve yama ızgarası", xlabel: "x", ylabel: "y", size: { w: 310, h: 300 }, img: { w: H, h: H, data: img }, side: true, series: [grid] },
        { title: "Dizi uzunluğu ve P (log ekseni)", xlabel: "yama kenarı P", ylabel: "token sayısı T", ylog: true, ylim: [1, Math.max(...seq) * 2], xticks: Ps.map((p) => [Math.log2(p), String(p)]), series: [
          { name: "T = (H/P)² + 1", x: Ps.map(Math.log2), y: seq, c: 0, dots: true }], marks: [{ x: Math.log2(P), label: `P = ${P}` }] },
      ],
      stats: [
        { label: "Yama sayısı N", value: N },
        { label: "Dizi uzunluğu ([class] dahil)", value: T },
        { label: "Yama embedding'i parametresi", value: peParams },
        { label: "Katman başına dikkat 2T²D (MAC)", value: attn },
        { label: "Dikkat / MLP = T/(4D)", value: attn / mlp },
      ],
      note: P1.join(" "),
    };
  },
};
