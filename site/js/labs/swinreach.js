// M4.15 — Endüktif yanlılık ve erişim: g × g yama ızgarasında, seçilen bir çıktı yamasını L katman sonra hangi girdi yamaları
// etkileyebilir? 3 × 3 konvolüsyon (yerel, alıcı alan katman başına 1 büyür), ViT (ilk katmanda küresel), Swin (M × M pencere,
// katmanlar arasında M/2 kaydırma; kaydırılmış pencereler orijinal ızgarada sarmalanmaz) ve kaydırmasız pencere (hiç büyümez).
// Ayrıca katman başına dikkat maliyeti: küresel 4hwC² + 2(hw)²C, pencere 4hwC² + 2M²hwC (Swin makalesi, Denk. 1–2).
import { fmt, range } from "./plot.js";

export function reachMap(model, g, M, L, pr, pc) {
  let R = range(g * g).map((k) => k === pr * g + pc);
  const s = Math.floor(M / 2);
  for (let l = 0; l < L; l++) {
    if (model === 1) return range(g * g).map(() => true);
    if (model === 0) { R = R.map((_, k) => { const r = Math.floor(k / g), c = k % g; for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) { const rr = r + dr, cc = c + dc; if (rr >= 0 && cc >= 0 && rr < g && cc < g && R[rr * g + cc]) return true; } return false; }); continue; }
    const sh = model === 2 && l % 2 === 1 ? s : 0, id = (k) => `${Math.floor((Math.floor(k / g) - sh) / M)},${Math.floor(((k % g) - sh) / M)}`;
    const hit = new Set(); R.forEach((v, k) => { if (v) hit.add(id(k)); });
    R = R.map((_, k) => hit.has(id(k)));
  }
  return R;
}

export default {
  params: [
    { id: "model", label: "Model", options: ["CNN (3 × 3 konvolüsyon)", "ViT (küresel öz-dikkat)", "Swin (pencere + kaydırma)", "yalnız pencere (kaydırmasız)"], value: 2 },
    { id: "g", label: "Yama ızgarası g (g × g token)", min: 4, max: 32, value: 16, int: true },
    { id: "M", label: "Pencere kenarı M (Swin'de 7)", min: 2, max: 8, value: 4, int: true, on: { model: [2, 3] } },
    { id: "L", label: "Katman sayısı L", min: 1, max: 12, value: 2, int: true, on: { model: [0, 2] } },
    { id: "pos", label: "Çıktı yamasının konumu (köşegen üzerinde, 0–1)", min: 0, max: 1, step: 0.05, value: 0.4, on: { model: [0, 2, 3] } },
    { id: "C", label: "Kanal boyutu C (maliyet için)", min: 32, max: 1024, value: 96, int: true, log: true },
  ],
  validate(q) { return (q.model === 2 || q.model === 3) && q.g % q.M ? [{ level: "error", msg: `g = ${q.g}, M = ${q.M}'ye tam bölünmeli (pencereler ızgarayı tam kaplamalı).` }] : []; },
  run(q) {
    const p = Math.min(q.g - 1, Math.floor(q.pos * q.g)), R = reachMap(q.model, q.g, q.M, q.L, p, p), n = R.filter(Boolean).length;
    const Ls = range(12).map((i) => i + 1), curve = (m) => Ls.map((l) => (m >= 2 && q.g % q.M ? null : reachMap(m, q.g, q.M, l, p, p).filter(Boolean).length));
    const hw = q.g * q.g, glob = 4 * hw * q.C ** 2 + 2 * hw * hw * q.C, win = 4 * hw * q.C ** 2 + 2 * q.M ** 2 * hw * q.C;
    const full = Ls.find((l) => curve(q.model)[l - 1] === hw);
    const name = ["CNN 3 × 3", "ViT", "Swin", "kaydırmasız pencere"][q.model];
    const P = [`${q.g} × ${q.g} ızgarada (${p}, ${p}) konumundaki çıktıyı ${q.L} katman sonra ${n} girdi yaması etkileyebilir (%${fmt((100 * n) / hw, 3)}). ${full ? `${name} tüm ızgaraya ${full}. katmanda ulaşıyor.` : `${name} 12 katmanda bile tüm ızgaraya ulaşmıyor.`}`];
    if (q.model === 0) P.push("CNN'in yerellik ve kaydırma eşdeğerliği her katmana gömülüdür: alıcı alan katman başına 1 yama büyür, (2L + 1)². Bu, az veride yararlı bir varsayımdır.");
    if (q.model === 1) P.push("ViT'in öz-dikkati ilk katmanda küreseldir: yerellik varsayımı yok, modelin bunu veriden öğrenmesi gerekir. Makalenin Şekil 7'sinde bazı başlar ilk katmanlarda yerel kalmayı öğrenir, bazıları baştan küresel bakar.");
    if (q.model === 2) P.push(`Swin: ${q.M} × ${q.M} pencereler, her ikinci katmanda ⌊M/2⌋ = ${Math.floor(q.M / 2)} kaydırma. Kaydırma pencere sınırlarını aşan bilgi akışını sağlar; erişim katman çifti başına yaklaşık M büyür.`);
    if (q.model === 3) P.push("Kaydırma olmadan pencereler birbirinden yalıtılmıştır: erişim kaç katman olursa olsun kendi penceresinde kalır. Swin'in kaydırılmış penceresi bu yüzden vardır.");
    P.push(`Katman başına dikkat maliyeti (C = ${q.C}, hw = ${hw}): küresel ${fmt(glob, 3)}, ${q.M} × ${q.M} pencere ${fmt(win, 3)}. Karesel terim 2(hw)²C yerine 2M²hwC: ızgara büyüdükçe pencere dikkati doğrusal kalır.`);
    return {
      plots: [
        { title: `Erişim haritası: ${name}, L = ${q.L}`, xlabel: "sütun", ylabel: "satır", size: { w: 310, h: 300 }, img: { w: q.g, h: q.g, data: R.map((v, k) => (k === p * q.g + p ? 0.55 : v ? 1 : 0.1)) }, side: true, series: [] },
        { title: "Erişilen girdi sayısı ve derinlik", xlabel: "katman L", ylabel: "girdi yaması", yzero: true, series: [
          { name: "CNN 3 × 3", x: Ls, y: curve(0), c: 1, dots: true },
          { name: "ViT", x: Ls, y: curve(1), c: 3, dash: true },
          { name: "Swin", x: Ls, y: curve(2), c: 0, dots: true },
          { name: "kaydırmasız pencere", x: Ls, y: curve(3), c: 4, dash: true }], marks: [{ x: q.L, label: `L = ${q.L}` }] },
      ],
      stats: [
        { label: "Erişilen girdi yaması", value: n },
        { label: "Izgaranın oranı", value: n / hw },
        { label: "Tüm ızgaraya ulaşılan katman", value: full ?? "> 12" },
        { label: "Dikkat maliyeti: küresel / pencere", value: `${fmt(glob, 3)} / ${fmt(win, 3)}` },
      ],
      note: P.join(" "),
    };
  },
};
