// M0.5 — Storage + (shape, stride, offset) = tensör görünümü. 24 elemanlık düz bir bellek ve onun üzerine kurulan görünümler.
// Bir hücreye odaklan/üzerine gel: adres formülü ve storage'daki yeri vurgulanır.
const STORE = 24;
export function contiguousStrides(shape) { const s = new Array(shape.length); let a = 1; for (let i = shape.length - 1; i >= 0; i--) { s[i] = a; a *= shape[i]; } return s; }
export function addresses(shape, stride, offset) {
  const out = [], idx = new Array(shape.length).fill(0), n = shape.reduce((a, b) => a * b, 1);
  for (let k = 0; k < n; k++) {
    out.push({ idx: [...idx], addr: offset + idx.reduce((a, v, j) => a + v * stride[j], 0) });
    for (let j = shape.length - 1; j >= 0; j--) { if (++idx[j] < shape[j]) break; idx[j] = 0; }
  }
  return out;
}
export function isContiguous(shape, stride) { const c = contiguousStrides(shape); return shape.every((n, i) => n === 1 || stride[i] === c[i]); }

export const VIEWS = [
  { name: "x: (4, 6) bitişik", base: true, shape: [4, 6], stride: [6, 1], offset: 0, code: "x = torch.arange(24).view(4, 6)", note: "Satır-öncelikli (row-major): son eksen bellekte ardışık (stride 1), bir satır aşağı inmek 6 eleman atlamak." },
  { name: "x.t() — transpoz", shape: [6, 4], stride: [1, 6], offset: 0, code: "x.t()", note: "Aynı storage, stride'lar yer değiştirdi. Artık bitişik değil: satır boyunca okumak belleği 6'şar atlıyor. Kopya yok, O(1)." },
  { name: "x[1:3, ::2] — dilim", shape: [2, 3], stride: [6, 2], offset: 6, code: "x[1:3, ::2]", note: "Başlangıç → offset (1·6 = 6), adım 2 → stride 2·1. Basit dilimleme hep görünümdür." },
  { name: "x[:, 2] — tek sütun", shape: [4], stride: [6], offset: 2, code: "x[:, 2]", note: "Bir ekseni seçmek o ekseni düşürür: 1 boyutlu, stride 6, offset 2." },
  { name: "x.view(2, 3, 4)", shape: [2, 3, 4], stride: [12, 4, 1], offset: 0, code: "x.view(2, 3, 4)", note: "Bitişik bir tensörün shape'i, eleman sırası (row-major) korunduğu sürece kopyasız değişir." },
  { name: "expand — yayın (stride 0)", shape: [3, 6], stride: [0, 1], offset: 0, code: "x[0].expand(3, 6)", note: "Birinci eksende stride 0: üç satırın hepsi aynı belleği okur. Yayın (broadcasting) bellekte kopya yapmadan böyle çalışır. Bu görünüme yazmak tanımsızdır (aynı hücreye üç yerden)." },
  { name: "unfold — kayan pencere", shape: [5, 4], stride: [4, 1], offset: 0, code: "x.flatten()[:20].unfold(0, 4, 4) ↔ pencere 4, adım 4", note: "Pencere boyu 4, adım 4: çakışmayan bloklar. Adımı 1 yap (stride (1, 1)) — pencereler çakışır, aynı eleman birden çok pencerede görünür." },
  { name: "çakışan pencereler (stride 1, 1)", shape: [5, 4], stride: [1, 1], offset: 0, code: "x.flatten().unfold(0, 4, 1)[:5]", note: "Her satır bir öncekinden 1 eleman kaymış pencere: hareketli ortalama, 1-B konvolüsyon (im2col) için kopyasız matris." },
  { name: "x.t().reshape(...) — kopya!", shape: [6, 4], stride: [1, 6], offset: 0, copy: true, code: "x.t().reshape(24)", note: "Bitişik olmayan bir görünümü düzleştirmek tek bir stride ile ifade edilemez (0, 6, 12, 18, 1, 7, … sırası): reshape kopya yapar, view hata verir (RuntimeError: view size is not compatible …)." },
];

export default {
  custom: true,
  keys: ["view", "s0", "s1", "s2", "n0", "n1", "n2", "off"],
  selfTest(assert) {
    assert.deepEqual(contiguousStrides([2, 3, 4]), [12, 4, 1]);
    const a = addresses([6, 4], [1, 6], 0); assert.equal(a[1].addr, 6, "transpozda (0,1) → adres 6"); assert.equal(a[4].addr, 1);
    assert.equal(isContiguous([6, 4], [1, 6]), false); assert.equal(isContiguous([4, 6], [6, 1]), true);
    assert.equal(isContiguous([1, 6], [99, 1]), true, "boyu 1 olan eksenin stride'ı önemsiz");
    for (const V of VIEWS) { const ad = addresses(V.shape, V.stride, V.offset).map((x) => x.addr); assert.ok(Math.min(...ad) >= 0 && Math.max(...ad) < STORE, V.name); }
    const sl = addresses([2, 3], [6, 2], 6).map((x) => x.addr); assert.deepEqual(sl, [6, 8, 10, 12, 14, 16], "x[1:3, ::2]");
  },
  mount(el, ctx) {
    const st = { v: 0, shape: [...VIEWS[0].shape], stride: [...VIEWS[0].stride], offset: 0, sel: null };
    const load = (i) => { const V = VIEWS[i]; st.v = i; st.shape = [...V.shape]; st.stride = [...V.stride]; st.offset = V.offset; st.sel = null; };
    const set = (p) => {
      if (p.view != null) load(p.view);
      const nd = [p.n0, p.n1, p.n2].filter((v) => v != null);
      if (nd.length) { st.shape = nd; st.v = -1; }
      const sd = [p.s0, p.s1, p.s2].filter((v) => v != null);
      if (sd.length) { st.stride = sd; st.v = -1; }
      if (p.off != null) { st.offset = p.off; st.v = -1; }
    };
    load(0); set(ctx.init);
    el.innerHTML = `<div class="lab-bar"><label>Görünüm <select data-k="view">${VIEWS.map((V, i) => `<option value="${i}">${V.name}</option>`).join("")}<option value="-1">— elle ayarla —</option></select></label></div>
      <div class="sv-edit"><span class="sv-lbl">shape</span><span data-k="shape"></span><span class="sv-lbl">stride</span><span data-k="stride"></span><label class="sv-lbl">offset <input type="number" data-k="off" min="0" max="23" step="1"></label>
      <button class="btn ghost small" type="button" data-k="dims">Eksen sayısı: <b></b></button></div>
      <p class="sv-code"><code></code></p>
      <div class="sv-issue" aria-live="polite"></div>
      <figure class="sv-fig"><figcaption>Storage: 24 elemanlık düz bellek (indis = adres, değer = <code>torch.arange(24)</code>)</figcaption><div class="sv-store" role="list"></div></figure>
      <figure class="sv-fig"><figcaption>Görünüm: <span data-k="shapecap"></span> — bir hücreye gel ya da Tab ile odaklan</figcaption><div class="sv-view"></div></figure>
      <div class="sv-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s);
    const numIn = (k, i, v) => `<input type="number" class="sv-n" data-${k}="${i}" value="${v}" min="${k === "n" ? 1 : 0}" max="${k === "n" ? 24 : 24}" step="1" aria-label="${k === "n" ? "shape" : "stride"}[${i}]">`;
    function controls() {
      $("[data-k=shape]").innerHTML = "(" + st.shape.map((v, i) => numIn("n", i, v)).join(", ") + ")";
      $("[data-k=stride]").innerHTML = "(" + st.stride.map((v, i) => numIn("s", i, v)).join(", ") + ")";
      $("[data-k=off]").value = st.offset; $("[data-k=view]").value = st.v; $("[data-k=dims] b").textContent = st.shape.length;
      for (const inp of el.querySelectorAll(".sv-n")) inp.addEventListener("change", (e) => {
        const v = Math.round(Number(e.target.value)); if (!Number.isFinite(v)) return;
        if (e.target.dataset.n != null) st.shape[+e.target.dataset.n] = Math.max(1, v); else st.stride[+e.target.dataset.s] = Math.max(0, v);
        st.v = -1; $("[data-k=view]").value = -1; draw();
      });
    }
    function draw() {
      const V = VIEWS[st.v], ad = addresses(st.shape, st.stride, st.offset), n = ad.length;
      const bad = ad.filter((a) => a.addr < 0 || a.addr >= STORE), cnt = new Array(STORE).fill(0);
      for (const a of ad) if (a.addr >= 0 && a.addr < STORE) cnt[a.addr]++;
      const overlap = cnt.some((c) => c > 1), contig = isContiguous(st.shape, st.stride);
      $("[data-k=shapecap]").textContent = `shape (${st.shape.join(", ")}), ${n} eleman`;
      $("p.sv-code code").textContent = V ? V.code : `torch.as_strided(storage, size=(${st.shape.join(", ")}), stride=(${st.stride.join(", ")}), storage_offset=${st.offset})`;
      $(".sv-issue").innerHTML = bad.length ? `<p class="lab-iss error">${bad.length} eleman storage'ın dışına düşüyor (en büyük adres ${Math.max(...ad.map((a) => a.addr))} ≥ 24). <code>as_strided</code> bunu denetlemez: gerçek bellekte rastgele veri okur ya da çöker. Güvenli sarmalayıcılar (<code>unfold</code>, <code>sliding_window_view</code>) bu yüzden tercih edilir.</p>` : "";
      $(".sv-store").innerHTML = Array.from({ length: STORE }, (_, i) => `<span role="listitem" class="sv-c${cnt[i] ? " used" : ""}${cnt[i] > 1 ? " multi" : ""}" data-a="${i}" title="adres ${i}${cnt[i] ? `, görünümde ${cnt[i]} kez` : ""}">${i}${cnt[i] > 1 ? `<sup>×${cnt[i]}</sup>` : ""}</span>`).join("");
      // görünüm: 1-B → tek satır; 2-B → ızgara; 3-B → yan yana bloklar
      const cell = (a) => `<button type="button" class="sv-v${a.addr < 0 || a.addr >= STORE ? " oob" : ""}" data-addr="${a.addr}" data-idx="${a.idx.join(",")}">${a.addr >= 0 && a.addr < STORE ? a.addr : "?"}</button>`;
      let html = "";
      if (st.shape.length === 1) html = `<div class="sv-grid" style="grid-template-columns:repeat(${st.shape[0]},2.3em)">${ad.map(cell).join("")}</div>`;
      else if (st.shape.length === 2) html = `<div class="sv-grid" style="grid-template-columns:repeat(${st.shape[1]},2.3em)">${ad.map(cell).join("")}</div>`;
      else { const per = st.shape[1] * st.shape[2]; for (let b = 0; b < st.shape[0]; b++) html += `<div class="sv-blk"><span class="small muted">[${b}]</span><div class="sv-grid" style="grid-template-columns:repeat(${st.shape[2]},2.3em)">${ad.slice(b * per, (b + 1) * per).map(cell).join("")}</div></div>`; }
      $(".sv-view").innerHTML = html;
      const kind = V?.copy ? "<b>reshape burada kopya yapar</b> (aşağıya bak)" : overlap ? "<b>çakışan</b> görünüm: bazı adresler birden çok konumdan okunuyor (yalnız okuma için güvenli)" : contig ? "<b>bitişik</b> (contiguous): row-major sırayla ardışık" : "bitişik <b>değil</b>: sıradaki eleman bellekte ardışık değil";
      $(".sv-read").innerHTML = `<p>${kind}. <code>is_contiguous()</code> = ${contig}. Bu görünüm ${new Set(ad.map((a) => a.addr)).size} farklı adres kullanıyor; storage 24 eleman.</p>${V ? `<p class="small">${V.note}</p>` : `<p class="small">Elle ayar: stride ve offset'i değiştir, adreslerin nereye düştüğünü izle. Kural: adres = offset + Σ indisₖ · strideₖ.</p>`}<p class="sv-addr small muted">Bir hücre seç: adres hesabı burada görünür.</p>`;
      const show = (b) => {
        const idx = b.dataset.idx.split(",").map(Number), addr = +b.dataset.addr;
        for (const c of el.querySelectorAll(".sv-c.hl, .sv-v.hl")) c.classList.remove("hl");
        b.classList.add("hl"); el.querySelector(`.sv-c[data-a="${addr}"]`)?.classList.add("hl");
        $(".sv-addr").innerHTML = `x[${idx.join(", ")}] → adres = ${st.offset} + ${idx.map((v, j) => `${v}·${st.stride[j]}`).join(" + ")} = <b>${addr}</b>`;
      };
      for (const b of el.querySelectorAll(".sv-v")) { b.addEventListener("mouseenter", () => show(b)); b.addEventListener("focus", () => show(b)); b.addEventListener("click", () => show(b)); }
    }
    $("[data-k=view]").addEventListener("change", (e) => { const i = Number(e.target.value); if (i >= 0) load(i); else st.v = -1; controls(); draw(); });
    $("[data-k=off]").addEventListener("change", (e) => { const v = Math.round(Number(e.target.value)); if (Number.isFinite(v) && v >= 0) { st.offset = v; st.v = -1; controls(); draw(); } });
    $("[data-k=dims]").addEventListener("click", () => {
      const d = st.shape.length % 3 + 1, n = 24; // 1→2→3→1 boyut, bitişik varsayılanlar
      st.shape = d === 1 ? [n] : d === 2 ? [4, 6] : [2, 3, 4]; st.stride = contiguousStrides(st.shape); st.offset = 0; st.v = -1; controls(); draw();
    });
    controls(); draw();
    return { set(p) { set(p); controls(); draw(); } };
  },
};
