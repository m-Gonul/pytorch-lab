// M4.12 — Skip-gram + negatif örnekleme (SGNS) küçük, yapısı bilinen sentetik bir derlemde. Her "varlık" kelimesi (kral, kadın,
// Paris, Fransa…) birkaç özelliğe (cinsiyet, statü, yaş, şehir/ülke, dil) bağlı bağlam kelimeleriyle birlikte geçer. Eğitim
// sonrası vektörler PCA ile 2-B'ye indirilip çizilir; a − b + c analojisi kosinüs benzerliğiyle sorgulanır.
const ENT = {
  kral: ["erkek", "soylu", "yetişkin"], kraliçe: ["kadın", "soylu", "yetişkin"], prens: ["erkek", "soylu", "çocuk"], prenses: ["kadın", "soylu", "çocuk"],
  adam: ["erkek", "halk", "yetişkin"], kadın: ["kadın", "halk", "yetişkin"], oğlan: ["erkek", "halk", "çocuk"], kız: ["kadın", "halk", "çocuk"],
  paris: ["şehir", "fr"], fransa: ["ülke", "fr"], roma: ["şehir", "it"], italya: ["ülke", "it"], madrid: ["şehir", "es"], ispanya: ["ülke", "es"], berlin: ["şehir", "de"], almanya: ["ülke", "de"],
};
const CTX = { erkek: ["bey", "bıyık"], kadın: ["hanım", "etek"], soylu: ["taç", "saray", "taht"], halk: ["köy", "tarla", "pazar"], yetişkin: ["iş", "vergi"], çocuk: ["oyuncak", "okul"],
  şehir: ["sokak", "belediye", "metro"], ülke: ["sınır", "bayrak", "ordu"], fr: ["fransızca", "şarap"], it: ["italyanca", "makarna"], es: ["ispanyolca", "flamenko"], de: ["almanca", "bira"] };
const FILL = ["ve", "bir", "çok", "da"];
export function corpus(r, n) {
  const ents = Object.keys(ENT), S = [];
  for (let i = 0; i < n; i++) { const e = ents[Math.floor(r() * ents.length)], w = ENT[e].map((a) => CTX[a][Math.floor(r() * CTX[a].length)]); if (r() < 0.5) w.push(FILL[Math.floor(r() * FILL.length)]); for (let j = w.length - 1; j > 0; j--) { const k = Math.floor(r() * (j + 1)); [w[j], w[k]] = [w[k], w[j]]; } w.splice(Math.floor(r() * (w.length + 1)), 0, e); S.push(w); }
  return S;
}
export function train(o, r) {
  const S = corpus(r, o.n), vocab = [...new Set(S.flat())], id = Object.fromEntries(vocab.map((w, i) => [w, i])), V = vocab.length, d = o.d;
  const cnt = new Array(V).fill(0); S.flat().forEach((w) => cnt[id[w]]++);
  const tot = cnt.reduce((a, b) => a + b, 0), pn = cnt.map((c) => c ** o.alpha), z = pn.reduce((a, b) => a + b, 0), cdf = []; pn.reduce((a, p, i) => (cdf[i] = a + p / z), 0);
  const neg = () => { const u = r(); let i = cdf.findIndex((c) => u <= c); return i < 0 ? V - 1 : i; };
  const keep = cnt.map((c) => (o.sub ? Math.min(1, Math.sqrt(1e-2 / (c / tot)) + 1e-2 / (c / tot)) : 1));
  const W = vocab.map(() => Array.from({ length: d }, () => (r() - 0.5) / d)), C = vocab.map(() => new Array(d).fill(0)), sig = (x) => 1 / (1 + Math.exp(-x)), loss = [];
  for (let ep = 0; ep < o.ep; ep++) {
    let L = 0, m = 0; const lr = o.lr * (1 - ep / o.ep) + 1e-4;
    for (const s0 of S) {
      const s = s0.filter((w) => r() < keep[id[w]]);
      s.forEach((w, i) => {
        for (let j = Math.max(0, i - o.win); j <= Math.min(s.length - 1, i + o.win); j++) {
          if (j === i) continue;
          const a = id[w], gW = new Array(d).fill(0), tg = [[id[s[j]], 1], ...Array.from({ length: o.k }, () => [neg(), 0])];
          for (const [c, y] of tg) { let dot = 0; for (let t = 0; t < d; t++) dot += W[a][t] * C[c][t]; const p = sig(dot), g = p - y; L += y ? -Math.log(p + 1e-12) : -Math.log(1 - p + 1e-12); for (let t = 0; t < d; t++) { gW[t] += g * C[c][t]; C[c][t] -= lr * g * W[a][t]; } }
          for (let t = 0; t < d; t++) W[a][t] -= lr * gW[t]; m++;
        }
      });
    }
    loss.push(L / Math.max(1, m));
  }
  return { vocab, id, W, cnt, loss };
}
const norm = (v) => Math.hypot(...v), cos = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (norm(a) * norm(b) + 1e-12);
export function analogy(R, a, b, c, top = 5) {
  const va = R.W[R.id[a]], vb = R.W[R.id[b]], vc = R.W[R.id[c]], q = va.map((x, i) => x - vb[i] + vc[i]);
  return R.vocab.filter((w) => ![a, b, c].includes(w)).map((w) => [w, cos(q, R.W[R.id[w]])]).sort((x, y) => y[1] - x[1]).slice(0, top);
}
export function pca2(X) {
  const n = X.length, d = X[0].length, mu = Array.from({ length: d }, (_, j) => X.reduce((s, x) => s + x[j], 0) / n), Y = X.map((x) => x.map((v, j) => v - mu[j]));
  const comps = [];
  for (let c = 0; c < Math.min(2, d); c++) {
    let v = Array.from({ length: d }, (_, j) => Math.sin(j * 1.7 + c + 1));
    for (let it = 0; it < 200; it++) { const Yv = Y.map((y) => y.reduce((s, x, j) => s + x * v[j], 0)); let w = Array.from({ length: d }, (_, j) => Y.reduce((s, y, i) => s + y[j] * Yv[i], 0)); for (const p of comps) { const pr = w.reduce((s, x, j) => s + x * p[j], 0); w = w.map((x, j) => x - pr * p[j]); } const nw = norm(w) || 1; v = w.map((x) => x / nw); }
    comps.push(v);
  }
  return Y.map((y) => comps.map((p) => y.reduce((s, x, j) => s + x * p[j], 0)).concat(d < 2 ? [0] : []));
}
const ENTS = Object.keys(ENT), GROUP = (w) => (["kral", "kraliçe", "prens", "prenses"].includes(w) ? "wv-a" : ["adam", "kadın", "oğlan", "kız"].includes(w) ? "wv-b" : ["paris", "roma", "madrid", "berlin"].includes(w) ? "wv-p" : "wv-q");

export default {
  custom: true,
  keys: ["d", "win", "k", "ep", "alpha", "sub", "a", "b", "c", "show"],
  selfTest(assert) {
    let s = 7; const r = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); r.normal = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
    const R = train({ n: 1500, d: 16, win: 3, k: 5, ep: 6, lr: 0.05, alpha: 0.75, sub: false }, r);
    assert.ok(R.loss[R.loss.length - 1] < R.loss[0], "kayıp azalır");
    const top = analogy(R, "kral", "adam", "kadın", 3).map((x) => x[0]); assert.ok(top.includes("kraliçe"), `kral − adam + kadın: ${top}`);
    const top2 = analogy(R, "paris", "fransa", "italya", 3).map((x) => x[0]); assert.ok(top2.includes("roma"), `paris − fransa + italya: ${top2}`);
    const P = pca2(R.W.slice(0, 5)); assert.equal(P[0].length, 2);
  },
  mount(el, ctx) {
    const st = { d: 16, win: 3, k: 5, ep: 6, alpha: 0.75, sub: 0, a: "kral", b: "adam", c: "kadın", show: 0 }; let R = null, key = "";
    const opt = (arr, v) => arr.map((x) => `<option value="${x}"${x === v ? " selected" : ""}>${x}</option>`).join("");
    el.innerHTML = `<div class="lab-bar">
      <label>Boyut d <select data-k="d">${opt([2, 4, 8, 16, 32], 16)}</select></label>
      <label>Pencere <select data-k="win">${opt([1, 2, 3, 4], 3)}</select></label>
      <label>Negatif k <select data-k="k">${opt([1, 2, 5, 10, 15], 5)}</select></label>
      <label>Epoch <select data-k="ep">${opt([1, 2, 4, 6, 10], 6)}</select></label>
      <label>Gürültü üssü α <select data-k="alpha">${opt([0, 0.5, 0.75, 1], 0.75)}</select></label>
      <label>Alt-örnekleme <select data-k="sub"><option value="0">kapalı</option><option value="1">açık (t = 10⁻²)</option></select></label>
      <button type="button" class="btn small" data-k="go">Eğit</button></div>
      <div class="lab-bar"><label>Analoji: <select data-k="a"></select></label> − <select data-k="b" aria-label="çıkarılan"></select> + <select data-k="c" aria-label="eklenen"></select>
      <label>Göster <select data-k="show"><option value="0">yalnız varlıklar</option><option value="1">tüm kelimeler</option></select></label></div>
      <div class="wv-wrap"><svg class="wv-svg" viewBox="0 0 420 340" role="img" aria-label="Kelime vektörlerinin PCA ile iki boyuta izdüşümü"></svg><div class="nx-read" aria-live="polite"></div></div>`;
    const $ = (s) => el.querySelector(s);
    for (const k of ["a", "b", "c"]) $(`[data-k=${k}]`).innerHTML = opt(ENTS, st[k]);
    function fit() {
      const kk = JSON.stringify([st.d, st.win, st.k, st.ep, st.alpha, st.sub]); if (kk === key && R) return; key = kk;
      R = train({ n: 1500, d: +st.d, win: +st.win, k: +st.k, ep: +st.ep, lr: 0.05, alpha: +st.alpha, sub: !!+st.sub }, ctx.rng(1));
    }
    function draw() {
      for (const k of Object.keys(st)) { const e = $(`[data-k=${k}]`); if (e) e.value = String(st[k]); }
      fit();
      const words = st.show ? R.vocab : ENTS, P = pca2(words.map((w) => R.W[R.id[w]])), xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), sx = (v) => 30 + ((v - x0) / (x1 - x0 || 1)) * 340, sy = (v) => 310 - ((v - y0) / (y1 - y0 || 1)) * 280;
      const pos = Object.fromEntries(words.map((w, i) => [w, [sx(xs[i]), sy(ys[i])]]));
      const pairs = [["kral", "kraliçe"], ["adam", "kadın"], ["prens", "prenses"], ["oğlan", "kız"], ["fransa", "paris"], ["italya", "roma"], ["ispanya", "madrid"], ["almanya", "berlin"]];
      $(".wv-svg").innerHTML = pairs.filter(([a, b]) => pos[a] && pos[b]).map(([a, b]) => `<line x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" class="wv-pair"/>`).join("") +
        words.map((w) => `<circle cx="${pos[w][0]}" cy="${pos[w][1]}" r="3.5" class="${ENT[w] ? GROUP(w) : "wv-o"}"/><text x="${pos[w][0] + 5}" y="${pos[w][1] - 4}" class="wv-t${ENT[w] ? "" : " wv-dim"}">${w}</text>`).join("");
      const res = analogy(R, st.a, st.b, st.c), exp = { "kral|adam|kadın": "kraliçe", "paris|fransa|italya": "roma", "prens|oğlan|kız": "prenses", "kraliçe|kadın|adam": "kral", "madrid|ispanya|almanya": "berlin" }[`${st.a}|${st.b}|${st.c}`];
      $(".nx-read").innerHTML = `<p><b>${st.a} − ${st.b} + ${st.c}</b> ≈ ? (kosinüs benzerliği, sorgudaki üç kelime hariç)</p><ol class="small">${res.map(([w, s]) => `<li>${w === exp ? `<b>${w}</b>` : w} — ${ctx.fmt(s, 3)}</li>`).join("")}</ol>
        <p class="small">Kayıp (çift başına, epoch'lara göre): ${R.loss.map((l) => ctx.fmt(l, 3)).join(" → ")}. Sözlük ${R.vocab.length} kelime; d = ${st.d}. Çizgiler çiftleri birleştiriyor (kral–kraliçe, ülke–başkent…): analoji, bu fark vektörlerinin <b>paralel</b> olmasıdır. ${+st.d === 2 ? "<b>d = 2</b>: iki boyut, cinsiyet, statü, yaş, şehir/ülke ve dil özelliklerinin hepsini ayrı eksenlere koymaya yetmiyor; analojiler bozuluyor." : ""}${+st.alpha === 0 ? " α = 0: negatifler düzgün dağılımdan; sık kelimeler (dolgu ve bağlam kelimeleri) az cezalanıyor." : ""} Bu küçük derlemde her varlık yalnız özelliklerinin bağlam kelimeleriyle geçtiği için analoji yapısı kendiliğinden ortaya çıkar; gerçek metinde ilişkiler çok daha gürültülüdür.</p>`;
    }
    for (const k of ["d", "win", "k", "ep", "alpha", "sub", "a", "b", "c", "show"]) $(`[data-k=${k}]`).addEventListener("change", (e) => { st[k] = ["a", "b", "c"].includes(k) ? e.target.value : +e.target.value; draw(); });
    $("[data-k=go]").addEventListener("click", () => { key = ""; draw(); });
    const set = (p) => { for (const k of Object.keys(st)) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
