// M4.16 — Ön eğitim amaçlarının gürültü fonksiyonları, aynı Türkçe paragraf üzerinde: BERT MLM (%p seç; %80 [MASK], %10 rastgele,
// %10 aynı), BART token maskeleme, token silme, metin doldurma (Poisson(λ) uzunluklu aralıklar tek <mask> ile), cümle karıştırma,
// belge döndürme ve ELECTRA (seçilen konumlara "üreteç" örneği; ayırt edici her konumda gerçek/değiştirilmiş tahmin eder).
// Ölçüt: kaybın kaç konumdan sinyal aldığı (nedensel LM her konumdan, MLM yalnız seçilenlerden).
import { fmt, range } from "./plot.js";

const TEXT = "kedi sabah erkenden bahçeye çıktı . güneş yeni doğmuştu ve çimenler ıslaktı . kuşlar ağaçta ötüyordu . kedi sessizce yaklaştı ama kuşlar uçup gitti . sonra eve dönüp kapının önünde uyudu .";
const TOK = TEXT.split(" "), VOC = [...new Set(TOK.filter((t) => t !== "."))];
export function poisson(lam, r) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= r(); } while (p > L); return k - 1; }
const esc = (s) => s.replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function corrupt(mode, q, r) {
  const T = TOK.slice(), n = T.length, out = [], lossPos = [];
  if (mode === 0 || mode === 6) {
    let sel = 0, kinds = [0, 0, 0];
    T.forEach((t, i) => {
      if (t !== "." && r() < q.p) {
        sel++;
        if (mode === 6) { const g = r() < q.gacc ? t : VOC[Math.floor(r() * VOC.length)]; out.push({ t: g, cls: g === t ? "same" : "repl" }); return; }
        const u = r(); if (u < 0.8) { kinds[0]++; out.push({ t: "[MASK]", cls: "mask" }); } else if (u < 0.9) { kinds[1]++; out.push({ t: VOC[Math.floor(r() * VOC.length)], cls: "repl" }); } else { kinds[2]++; out.push({ t, cls: "same" }); }
        lossPos.push(i);
      } else out.push({ t, cls: "" });
    });
    return { out, signal: mode === 6 ? n : lossPos.length, sel, kinds, n };
  }
  if (mode === 1) { let s = 0; T.forEach((t) => { if (t !== "." && r() < q.p) { s++; out.push({ t: "&lt;mask&gt;", cls: "mask" }); } else out.push({ t, cls: "" }); }); return { out, signal: n, sel: s, n }; }
  if (mode === 2) { let s = 0; T.forEach((t) => { if (t !== "." && r() < q.p) s++; else out.push({ t, cls: "" }); }); return { out, signal: n, sel: s, n }; }
  if (mode === 3) {
    const masked = new Array(n).fill(false), spans = []; let m = 0, guard = 0;
    while (m < q.p * n && guard++ < 500) { const L = poisson(q.lam, r), s = Math.floor(r() * n); if (range(L).some((k) => s + k >= n || masked[s + k])) continue; spans.push({ s, L }); for (let k = 0; k < L; k++) masked[s + k] = true; m += L; }
    const starts = new Map(spans.map((x) => [x.s, x.L]));
    for (let i = 0; i < n; i++) { if (starts.has(i)) { out.push({ t: "&lt;mask&gt;", cls: "mask", L: starts.get(i) }); if (starts.get(i) === 0) out.push({ t: T[i], cls: "" }); continue; } if (!masked[i]) out.push({ t: T[i], cls: "" }); }
    return { out, signal: n, sel: m, spans, n };
  }
  if (mode === 4) {
    const S = []; let cur = []; T.forEach((t) => { cur.push(t); if (t === ".") { S.push(cur); cur = []; } });
    for (let i = S.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [S[i], S[j]] = [S[j], S[i]]; }
    return { out: S.flat().map((t) => ({ t, cls: "" })), signal: n, sel: 0, n, nS: S.length };
  }
  const k = Math.floor(r() * n); return { out: [...T.slice(k), ...T.slice(0, k)].map((t, i) => ({ t, cls: i === 0 ? "mask" : "" })), signal: n, sel: 0, n, rot: k };
}

export default {
  params: [
    { id: "mode", label: "Gürültü / amaç", options: ["BERT MLM (80/10/10)", "BART: token maskeleme", "BART: token silme", "BART: metin doldurma (span)", "BART: cümle karıştırma", "BART: belge döndürme", "ELECTRA: değiştirilmiş token tespiti"], value: 0 },
    { id: "p", label: "Bozma oranı p", min: 0.05, max: 0.6, step: 0.01, value: 0.15, on: { mode: [0, 1, 2, 3, 6] } },
    { id: "lam", label: "Aralık uzunluğu Poisson λ", min: 0.5, max: 6, step: 0.5, value: 3, on: { mode: [3] } },
    { id: "gacc", label: "ELECTRA üretecinin doğru token önerme olasılığı", min: 0, max: 0.9, step: 0.05, value: 0.3, on: { mode: [6] } },
  ],
  seed: true,
  run(q, r) {
    const R = corrupt(q.mode, q, r), html = R.out.map((o) => (o.cls === "mask" ? (o.L != null ? `<b>&lt;mask:${o.L}&gt;</b>` : `<b>${o.t}</b>`) : o.cls === "repl" ? `<u>${o.t}</u>` : o.cls === "same" ? `<i>${o.t}</i>` : o.t)).join(" ");
    const P = [`<span class="small">Özgün: ${TOK.join(" ")}</span><br>Bozulmuş: ${html}`];
    const desc = [
      `${R.sel} token seçildi (beklenti p·n = ${fmt(q.p * TOK.filter((t) => t !== ".").length, 3)}): ${R.kinds?.[0] ?? 0} [MASK], ${R.kinds?.[1] ?? 0} rastgele (<u>altı çizili</u>), ${R.kinds?.[2] ?? 0} aynı (<i>eğik</i>). Kayıp yalnız seçilen ${R.signal} konumda: dizinin %${fmt((100 * R.signal) / R.n, 3)} kadarı. Rastgele ve aynı bırakılan tokenlar, ince ayarda hiç görülmeyen [MASK]'in yarattığı uyumsuzluğu azaltır; model her tokenın temsilini doğru tutmak zorunda kalır.`,
      `${R.sel} token &lt;mask&gt; ile değiştirildi. BART'ta kod çözücü özgün metnin tamamını üretir: kayıp tüm ${R.n} konumda.`,
      `${R.sel} token silindi ve yerine hiçbir şey konmadı: model hem neyin hem <b>nerede</b> eksik olduğunu bulmalı.`,
      `${R.spans?.length ?? 0} aralık, toplam ${R.sel} token maskelendi; her aralık <b>tek</b> bir &lt;mask&gt; ile değişir (gösterimde &lt;mask:L&gt;, L = aralık uzunluğu; L = 0, araya eklenen boş maske). Model eksik token sayısını da tahmin etmeli. BART makalesinde en iyi sonuç λ = 3 ile metin doldurmadan geldi.`,
      `${R.nS} cümlenin sırası karıştırıldı; model özgün sırayı üretmeli.`,
      `Belge ${R.rot}. tokendan döndürüldü (<b>kalın</b>: yeni ilk token); model belgenin gerçek başını bulmalı.`,
      `${R.sel} konum üreteçten gelen bir tokenla değiştirildi; üreteç bunların bir kısmında doğru tokenı önerdi (<i>eğik</i>: "özgün" sayılır), diğerleri <u>altı çizili</u>. Ayırt edici <b>her</b> konumda "değiştirildi mi?" sorusunu yanıtlar: sinyal ${R.n} konumdan, MLM'nin ≈ ${fmt(1 / q.p, 2)} katı.`,
    ][q.mode];
    P.push(desc);
    const lens = range(9), pois = lens.map((k) => Math.exp(-q.lam) * q.lam ** k / [1, 1, 2, 6, 24, 120, 720, 5040, 40320][k]);
    const plots = [{ title: "Kaybın sinyal aldığı konumların oranı", xlabel: "", ylabel: "oran", ylim: [0, 1.05], xticks: [[0, "nedensel LM"], [1, "bu amaç"], [2, "MLM (p = 0.15)"]], series: [{ name: "oran", x: [0, 1, 2], y: [1, R.signal / R.n, 0.15], kind: "bar", c: 0 }] }];
    if (q.mode === 3) plots.push({ title: `Aralık uzunluğu: Poisson(λ = ${q.lam})`, xlabel: "uzunluk", ylabel: "olasılık", yzero: true, series: [{ name: "Poisson", x: lens, y: pois, kind: "bar", c: 2 }, { name: "bu örnekteki aralıklar", x: lens, y: lens.map((k) => (R.spans.filter((s) => s.L === k).length) / Math.max(1, R.spans.length)), kind: "scatter", c: 3, r: 5 }] });
    return {
      plots,
      stats: [
        { label: "Dizi uzunluğu", value: R.n },
        { label: "Bozulan token", value: R.sel },
        { label: "Kayıp sinyali olan konum", value: R.signal },
        { label: "Bozulmuş dizinin uzunluğu", value: R.out.length },
      ],
      note: P.join("<br>"),
    };
  },
};
