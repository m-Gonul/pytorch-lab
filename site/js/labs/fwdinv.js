// M9.1 — İleri ve ters eşleme. W × W'lik bir test görüntüsü (dama + daire) merkez etrafında θ döndürülüp s ile ölçekleniyor.
// İleri eşleme: her girdi pikseli için çıkış konumunu hesapla ve en yakın piksele yaz; bazı çıkış pikselleri hiç değer almaz (delik),
// bazıları birden çok değer alır (çakışma). Ters eşleme: her çıkış pikseli için A⁻¹ ile kaynağı hesapla ve oradan oku (en yakın komşu
// ya da bilineer). Delikler, kaynağı görüntünün içinde kalan çıkış pikselleri arasında sayılır.
import { fmt, range } from "./plot.js";

function test(W) { return range(W * W).map((k) => { const x = k % W, y = Math.floor(k / W), c = (Math.floor(x / 4) + Math.floor(y / 4)) % 2 ? 0.85 : 0.25, r = Math.hypot(x - W / 2 + 0.5, y - W / 2 + 0.5); return Math.abs(r - W / 3) < 1 ? 1 : c; }); }
export default {
  params: [
    { id: "th", label: "Döndürme θ (derece)", min: -180, max: 180, step: 1, value: 30 },
    { id: "s", label: "Ölçek s (> 1 büyütür)", min: 0.3, max: 3, step: 0.05, value: 1.6 },
    { id: "m", label: "Yöntem", options: ["ileri eşleme (en yakın piksele yaz)", "ters eşleme, en yakın komşu", "ters eşleme, bilineer"], value: 0 },
    { id: "W", label: "Görüntü boyu W", min: 16, max: 96, value: 48, int: true },
  ],
  run(q) {
    const W = q.W, img = test(W), c = (W - 1) / 2, t = (q.th * Math.PI) / 180, co = Math.cos(t), si = Math.sin(t);
    const fwd = (x, y) => [c + q.s * (co * (x - c) - si * (y - c)), c + q.s * (si * (x - c) + co * (y - c))];
    const inv = (x, y) => [c + (co * (x - c) + si * (y - c)) / q.s, c + (-si * (x - c) + co * (y - c)) / q.s];
    const out = new Array(W * W).fill(0), cnt = new Array(W * W).fill(0);
    if (q.m === 0) for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const [u, v] = fwd(x, y), X = Math.round(u), Y = Math.round(v); if (X >= 0 && X < W && Y >= 0 && Y < W) { out[Y * W + X] = img[y * W + x]; cnt[Y * W + X]++; } }
    else for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const [u, v] = inv(x, y); if (u < -0.5 || u > W - 0.5 || v < -0.5 || v > W - 0.5) continue;
      if (q.m === 1) { out[y * W + x] = img[Math.min(W - 1, Math.max(0, Math.round(v))) * W + Math.min(W - 1, Math.max(0, Math.round(u)))]; }
      else { const x0 = Math.max(0, Math.min(W - 2, Math.floor(u))), y0 = Math.max(0, Math.min(W - 2, Math.floor(v))), a = Math.min(1, Math.max(0, u - x0)), b = Math.min(1, Math.max(0, v - y0)); out[y * W + x] = (1 - a) * (1 - b) * img[y0 * W + x0] + a * (1 - b) * img[y0 * W + x0 + 1] + (1 - a) * b * img[(y0 + 1) * W + x0] + a * b * img[(y0 + 1) * W + x0 + 1]; }
      cnt[y * W + x] = 1;
    }
    let inside = 0, holes = 0, coll = 0;
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const [u, v] = inv(x, y); if (u >= 0 && u <= W - 1 && v >= 0 && v <= W - 1) { inside++; if (cnt[y * W + x] === 0) holes++; } if (cnt[y * W + x] > 1) coll++; }
    const P = [q.m === 0 ? `İleri eşleme: görüntünün içine düşmesi gereken ${inside} çıkış pikselinin ${holes} tanesi hiç değer almadı (delik, siyah noktalar); ${coll} çıkış pikseline birden çok girdi pikseli yazıldı (çakışma; son yazılan kazanır). ${q.s > 1 ? "Büyütmede girdi pikselleri çıkışı örtmeye yetmez: delikler kaçınılmazdır." : "Küçültmede çakışmalar baskındır: birçok girdi pikseli aynı çıkışa düşer ve biri dışında hepsi atılır (aliasing)."}` : `Ters eşleme: her çıkış pikseli kaynağını A⁻¹ ile buldu; delik ve çakışma yok (delik ${holes}, çakışma ${coll}). ${q.m === 1 ? "En yakın komşu basamaklı kenarlar üretir." : "Bilineer değerleri dört komşudan karıştırır; kenarlar yumuşar."}`];
    P.push(`Dönüşüm x' = s·R(θ)(x − c) + c; ters eşleme için A⁻¹ = R(−θ)/s gerekir. Afin dönüşümlerin tersi her zaman vardır (det ≠ 0), bu yüzden ters eşleme her afin dönüşümde uygulanabilir.`);
    return {
      plots: [
        { title: "Girdi", img: { w: W, h: W, data: img }, side: true, size: { w: 300, h: 290 }, series: [] },
        { title: `Çıkış (${["ileri", "ters, en yakın", "ters, bilineer"][q.m]})`, img: { w: W, h: W, data: out }, side: true, size: { w: 300, h: 290 }, series: [] },
      ],
      stats: [{ label: "Delik / içerideki çıkış pikseli", value: `${holes} / ${inside}` }, { label: "Çakışan çıkış pikseli", value: coll }],
      note: P.join(" "),
    };
  },
};
