// M4.5 — BPE adım adım: en sık komşu çifti birleştir, tekrarla. Derlem (kelime: sıklık), çift sıklık tablosu, birleştirme listesi;
// öğrenilen birleştirmelerle yeni bir kelimeyi kodlama (birleştirmeler öğrenildikleri sırayla uygulanır).
export const CORPORA = {
  "Sennrich ve ark. örneği (İngilizce)": "low 5\nlower 2\nnewest 6\nwidest 3",
  "Türkçe ekler": "evler 4\nevlerde 3\nevde 5\nkitaplar 3\nkitaplarda 2\nkitapta 2\nokullarda 2\nokulda 3",
  "Tekrar eden heceler": "banana 4\nbandana 2\nanana 3",
};
export const EOW = "</w>";
export function parse(text) {
  const words = [];
  for (const line of text.split(/\n+/)) {
    const m = line.trim().match(/^(\S+)(?:\s+(\d+))?$/);
    if (m) words.push({ w: m[1], f: m[2] ? +m[2] : 1, s: [...m[1], EOW] });
  }
  return words;
}
export function pairCounts(words) {
  const c = new Map();
  for (const { s, f } of words) for (let i = 0; i + 1 < s.length; i++) { const k = s[i] + "\u0000" + s[i + 1]; c.set(k, (c.get(k) || 0) + f); }
  return c;
}
export const applyMerge = (s, a, b) => { const r = []; for (let i = 0; i < s.length; i++) { if (i + 1 < s.length && s[i] === a && s[i + 1] === b) { r.push(a + b); i++; } else r.push(s[i]); } return r; };
// en sık çift; eşitlikte ilk karşılaşılan (derlem sırası, soldan sağa)
export function train(words, K) {
  const W = words.map((x) => ({ ...x, s: [...x.s] })), merges = [];
  for (let k = 0; k < K; k++) {
    const c = pairCounts(W); let best = null, bf = 0;
    for (const [key, f] of c) if (f > bf) { best = key; bf = f; }
    if (!best || bf < 1) break;
    const [a, b] = best.split("\u0000"); merges.push({ a, b, f: bf });
    for (const x of W) x.s = applyMerge(x.s, a, b);
  }
  return { W, merges };
}
export const encode = (word, merges) => { let s = [...word, EOW]; for (const { a, b } of merges) s = applyMerge(s, a, b); return s; };

export default {
  custom: true,
  keys: ["corpus", "k", "test"],
  selfTest(assert) {
    const W = parse(CORPORA["Sennrich ve ark. örneği (İngilizce)"]), T = train(W, 10);
    assert.deepEqual(T.merges.slice(0, 5).map((m) => m.a + "+" + m.b), ["e+s", "es+t", "est+</w>", "l+o", "lo+w"]);
    assert.equal(T.merges[0].f, 9);
    assert.deepEqual(T.W.map((x) => x.s.join(" ")), ["low</w>", "low e r </w>", "newest</w>", "wi d est</w>"]);
    // kodlama: öğrenilmemiş kelime alt birimlere bölünür, OOV yok
    assert.deepEqual(encode("lowest", T.merges), ["low", "est</w>"]);
    assert.deepEqual(encode("xyz", T.merges), ["x", "y", "z", "</w>"]);
  },
  mount(el, ctx) {
    const st = { corpus: 0, k: 3, test: "lowest" }, names = Object.keys(CORPORA);
    el.innerHTML = `<div class="lab-bar"><label>Derlem <select data-k="corpus">${names.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select></label>
      <label>Birleştirme sayısı <input type="number" data-k="k" min="0" max="40" step="1" style="width:5em"></label>
      <button type="button" data-a="prev">− 1</button><button type="button" data-a="next">+ 1 birleştir</button>
      <label>Kodlanacak kelime <input type="text" data-k="test" maxlength="24" style="width:9em"></label></div>
      <label class="small">Derlem (her satır: kelime sıklık) — düzenleyebilirsin<textarea data-k="text" rows="4" style="width:100%;font-family:var(--mono)"></textarea></label>
      <div class="bp-wrap"><div class="bp-words"></div><div class="bp-side"></div></div><div class="nx-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s), ta = $("[data-k=text]");
    ta.value = CORPORA[names[st.corpus]];
    const tok = (t, i) => `<span class="bp-t bp-c${i % 6}">${t.replace("</w>", "␣")}</span>`;
    function draw() {
      $("[data-k=corpus]").value = st.corpus; $("[data-k=k]").value = st.k; if (document.activeElement !== $("[data-k=test]")) $("[data-k=test]").value = st.test;
      const words = parse(ta.value), T = train(words, st.k), next = train(words, st.k + 1).merges[st.k], c = [...pairCounts(T.W)].sort((x, y) => y[1] - x[1]).slice(0, 8);
      const vocab = new Set(T.W.flatMap((x) => x.s)), base = new Set(words.flatMap((x) => x.s)), nTok = T.W.reduce((a, x) => a + x.s.length * x.f, 0), nChar = words.reduce((a, x) => a + x.s.length * x.f, 0);
      $(".bp-words").innerHTML = `<table class="ic-t"><thead><tr><th>kelime</th><th>sıklık</th><th>${st.k} birleştirmeden sonraki tokenlar</th></tr></thead><tbody>${T.W.map((x) => `<tr><td>${x.w}</td><td>${x.f}</td><td style="text-align:left">${x.s.map(tok).join("")}</td></tr>`).join("")}</tbody></table>`;
      $(".bp-side").innerHTML = `<p class="nx-cap">en sık komşu çiftler (şimdi)</p><table class="ic-t"><tbody>${c.map(([k, f], i) => `<tr class="${i === 0 ? "bp-best" : ""}"><td>${k.split("\u0000").map((t) => t.replace("</w>", "␣")).join(" + ")}</td><td>${f}</td></tr>`).join("") || "<tr><td>— (birleştirilecek çift yok)</td></tr>"}</tbody></table>
        <p class="nx-cap">öğrenilen birleştirmeler (sırayla)</p><ol class="bp-m">${T.merges.map((m) => `<li>${m.a.replace("</w>", "␣")} + ${m.b.replace("</w>", "␣")} → <b>${(m.a + m.b).replace("</w>", "␣")}</b> <span class="muted">(${m.f})</span></li>`).join("") || "<li class='muted'>henüz yok</li>"}</ol>`;
      const enc = encode(st.test || "", T.merges);
      $(".nx-read").innerHTML = `<p><b>"${st.test}"</b> → ${enc.map(tok).join("")} (${enc.length} token). ${enc.every((t) => vocab.has(t) || base.has(t)) ? "" : "Bazı karakterler derlemde yok: gerçek uygulamada bilinmeyen karakter tokenı ya da bayt düzeyi BPE gerekir."}</p>
        <p class="small">Sözlük: ${base.size} başlangıç sembolü + ${T.merges.length} birleştirme = en fazla ${base.size + T.merges.length} token (şu an kullanılan ${vocab.size}). Derlem ${nChar} karakter-token'dan ${nTok} token'a indi (sıklıkla ağırlıklı). ${next ? `Sonraki birleştirme: <b>${next.a.replace("</w>", "␣")} + ${next.b.replace("</w>", "␣")}</b> (${next.f} kez).` : "Birleştirilecek çift kalmadı: her kelime tek token."} ␣ = kelime sonu işareti &lt;/w&gt;: "est␣" kelime sonundaki "est"tir, ortadakinden ayrı bir token.</p>`;
    }
    $("[data-k=corpus]").addEventListener("change", (e) => { st.corpus = +e.target.value; ta.value = CORPORA[names[st.corpus]]; st.k = 0; draw(); });
    $("[data-k=k]").addEventListener("input", (e) => { st.k = Math.max(0, Math.min(40, Math.round(+e.target.value || 0))); draw(); });
    $("[data-k=test]").addEventListener("input", (e) => { st.test = e.target.value.trim(); draw(); });
    ta.addEventListener("input", () => draw());
    $("[data-a=prev]").addEventListener("click", () => { st.k = Math.max(0, st.k - 1); draw(); });
    $("[data-a=next]").addEventListener("click", () => { st.k = Math.min(40, st.k + 1); draw(); });
    const set = (p) => { if (p.corpus != null && p.corpus !== st.corpus) { st.corpus = p.corpus; ta.value = CORPORA[names[st.corpus]]; } for (const k of ["k", "test"]) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
