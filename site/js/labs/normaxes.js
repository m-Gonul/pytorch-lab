// M3.6 — "Hangi eksenlerde normalize?" — (N, C, S) bir tensörde BN / LN / IN / GN / RMSNorm'un istatistik gruplarını gösterir.
// Bir hücre seç: aynı μ, σ'yı paylaşan hücreler vurgulanır; normalize değerler ve gruplar hesaplanır.
const N = 4, C = 6, S = 4;
export const KINDS = ["BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm", "RMSNorm"];
// deterministik örnek tensör: kanal başına güçlü kayma, örnek başına ölçek farkı — BN ile LN'nin neyi sildiği görünsün
export function sample() {
  const chOff = [3, -2, 0.5, 6, -4, 1], smpScale = [0.5, 1, 2, 4];
  let k = 7; const u = () => { k = (k * 16807) % 2147483647; return k / 2147483647 - 0.5; };
  return Array.from({ length: N }, (_, n) => Array.from({ length: C }, (_, c) => Array.from({ length: S }, () => +(smpScale[n] * (chOff[c] / 2 + 2 * u()) + chOff[c] / 2).toFixed(2))));
}
// grup anahtarı: aynı anahtar → aynı istatistik
export function groupKey(kind, G, n, c, s) {
  switch (kind) {
    case 0: return `c${c}`;
    case 1: case 4: return `n${n}`;
    case 2: return `n${n}c${c}`;
    case 3: return `n${n}g${Math.floor(c / (C / G))}`;
  }
}
export function normalize(X, kind, G, eps = 0) {
  const groups = new Map();
  for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let s = 0; s < S; s++) { const key = groupKey(kind, G, n, c, s); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(X[n][c][s]); }
  const st = new Map();
  for (const [key, v] of groups) { const mu = kind === 4 ? 0 : v.reduce((a, b) => a + b, 0) / v.length, var_ = v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length; st.set(key, { mu, var: var_, n: v.length }); }
  const Y = X.map((P, n) => P.map((R, c) => R.map((x, s) => { const g = st.get(groupKey(kind, G, n, c, s)); return (x - g.mu) / Math.sqrt(g.var + eps); })));
  return { Y, st };
}
const PT = ["nn.BatchNorm2d(C)", "nn.LayerNorm([C, H, W])", "nn.InstanceNorm2d(C)", "nn.GroupNorm(G, C)", "nn.RMSNorm([C, H, W])"];
const DESC = [
  "her <b>kanal</b> için: tüm örnekler (N) ve tüm konumlar (H·W) üzerinden",
  "her <b>örnek</b> için: tüm kanallar ve konumlar üzerinden",
  "her <b>örnek × kanal</b> için: yalnız konumlar üzerinden",
  "her <b>örnek × kanal grubu</b> için: grubun kanalları ve konumlar üzerinden",
  "LayerNorm'un eksenleri, ama <b>ortalama çıkarılmaz</b>: yalnız RMS'e bölünür",
];
const AFF = ["C (kanal başına γ, β)", "C·H·W (eleman başına γ, β — varsayılan)", "C (affine=True ise; varsayılan kapalı)", "C (kanal başına γ, β)", "C·H·W (yalnız γ; β yok)"];

export default {
  custom: true,
  keys: ["kind", "G", "n", "c", "s", "view"],
  selfTest(assert) {
    const X = sample(), close = (a, b, m) => assert.ok(Math.abs(a - b) < 1e-9, `${m}: ${a} ≠ ${b}`);
    // GN(G=1) ≡ LN, GN(G=C) ≡ IN
    const ln = normalize(X, 1, 1).Y, g1 = normalize(X, 3, 1).Y, inn = normalize(X, 2, 1).Y, gC = normalize(X, 3, C).Y;
    close(ln[2][3][1], g1[2][3][1], "GN(1) = LN"); close(inn[1][4][2], gC[1][4][2], "GN(C) = IN");
    // BN: her kanalın normalize ortalaması 0, varyansı 1
    const bn = normalize(X, 0, 1).Y;
    for (let c = 0; c < C; c++) { const v = []; for (let n = 0; n < N; n++) for (let s = 0; s < S; s++) v.push(bn[n][c][s]); const m = v.reduce((a, b) => a + b, 0) / v.length; close(m, 0, "BN kanal ortalaması"); close(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length, 1, "BN kanal varyansı"); }
    // grup sayıları
    assert.equal(normalize(X, 0, 1).st.size, C); assert.equal(normalize(X, 1, 1).st.size, N); assert.equal(normalize(X, 2, 1).st.size, N * C); assert.equal(normalize(X, 3, 2).st.size, N * 2);
    // RMSNorm: her örnekte ortalama kare 1
    const rm = normalize(X, 4, 1).Y; for (let n = 0; n < N; n++) close(rm[n].flat().reduce((a, b) => a + b * b, 0) / (C * S), 1, "RMS");
  },
  mount(el, ctx) {
    const X = sample(), st = { kind: 0, G: 2, n: 0, c: 3, s: 1, view: 0 };
    el.innerHTML = `<div class="lab-bar"><label>Normalizasyon <select data-k="kind">${KINDS.map((k, i) => `<option value="${i}">${k}</option>`).join("")}</select></label>
      <label data-gwrap>Grup sayısı G <select data-k="G">${[1, 2, 3, 6].map((g) => `<option value="${g}">${g}</option>`).join("")}</select></label>
      <label>Göster <select data-k="view"><option value="0">ham değerler x</option><option value="1">normalize x̂</option></select></label></div>
      <p class="small muted">Tensör shape'i (N, C, H·W) = (${N}, ${C}, ${S}): her blok bir örnek, satırlar kanallar, sütunlar uzamsal konumlar. Bir hücreye tıkla ya da Tab ile odaklanıp Enter'a bas — aynı μ ve σ'yı paylaşan hücreler çerçevelenir.</p>
      <div class="nx-grid"></div><div class="nx-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s);
    function draw() {
      const { Y, st: S_ } = normalize(X, st.kind, st.G, 1e-5), key = groupKey(st.kind, st.G, st.n, st.c, st.s), V = st.view ? Y : X;
      const mx = Math.max(...V.flat(2).map(Math.abs)) || 1;
      $("[data-gwrap]").hidden = st.kind !== 3; $("[data-k=kind]").value = st.kind; $("[data-k=G]").value = st.G; $("[data-k=view]").value = st.view;
      $(".nx-grid").innerHTML = X.map((P, n) => `<div class="nx-blk"><p class="nx-cap">örnek ${n}</p><div class="nx-cells" style="grid-template-columns:repeat(${S},1fr)">${P.map((R, c) => R.map((_, s) => {
        const v = V[n][c][s], inG = groupKey(st.kind, st.G, n, c, s) === key, sel = n === st.n && c === st.c && s === st.s, col = v >= 0 ? "--c-a" : "--c-b";
        return `<button type="button" class="nx-c${inG ? " grp" : ""}${sel ? " sel" : ""}" data-n="${n}" data-c="${c}" data-s="${s}" style="background:color-mix(in srgb, var(${col}) ${Math.round((Math.abs(v) / mx) * 55)}%, transparent)" aria-label="örnek ${n}, kanal ${c}, konum ${s}: ${v.toFixed(2)}">${v.toFixed(st.view ? 2 : 1)}</button>`;
      }).join("")).join("")}</div></div>`).join("");
      const g = S_.get(key), xh = Y[st.n][st.c][st.s];
      // her yöntemin neyi sildiğini göster: normalize sonrası kanal ortalamaları ve örnek std'leri
      const chM = Array.from({ length: C }, (_, c) => { let a = 0; for (let n = 0; n < N; n++) for (let s = 0; s < S; s++) a += Y[n][c][s]; return a / (N * S); });
      const smS = Array.from({ length: N }, (_, n) => { const v = Y[n].flat(), m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); });
      const f = (v) => ctx.fmt(Math.abs(v) < 5e-10 ? 0 : v, 3);
      $(".nx-read").innerHTML = `<p><b>${KINDS[st.kind]}</b> — istatistik ${DESC[st.kind]}. Bu tensörde <b>${S_.size}</b> ayrı (μ, σ) çifti var, her biri <b>${g.n}</b> elemandan hesaplanıyor.</p>
        <p>Seçili hücre x = ${f(X[st.n][st.c][st.s])}: grubunun ${st.kind === 4 ? "RMS'i" : "μ = " + f(g.mu) + ", σ ="} ${f(Math.sqrt(g.var))} → x̂ = ${st.kind === 4 ? "x / RMS" : "(x − μ) / σ"} = <b>${f(xh)}</b> (ε = 10⁻⁵).</p>
        <p class="small">Normalize sonrası <b>kanal ortalamaları</b> (N ve konumlar üzerinden): ${chM.map(f).join(", ")}<br>Normalize sonrası <b>örnek std'leri</b>: ${smS.map(f).join(", ")}</p>
        <p class="small">PyTorch: <code>${PT[st.kind].replace("G, C", `${st.G}, C`)}</code>; öğrenilen afin parametre sayısı: ${AFF[st.kind]}.${st.kind === 0 ? " Çıkarımda μ, σ yerine kayan ortalamalar kullanılır; diğerleri eğitim ve çıkarımda aynı hesabı yapar." : ""}${st.kind === 3 && st.G === 1 ? " <b>G = 1: LayerNorm ile aynı gruplar.</b>" : ""}${st.kind === 3 && st.G === C ? " <b>G = C: InstanceNorm ile aynı gruplar.</b>" : ""}</p>`;
      for (const b of el.querySelectorAll(".nx-c")) b.addEventListener("click", () => { st.n = +b.dataset.n; st.c = +b.dataset.c; st.s = +b.dataset.s; draw(); el.querySelector(`.nx-c[data-n="${st.n}"][data-c="${st.c}"][data-s="${st.s}"]`)?.focus(); });
    }
    for (const k of ["kind", "G", "view"]) $(`[data-k=${k}]`).addEventListener("change", (e) => { st[k] = +e.target.value; draw(); });
    const set = (p) => { for (const k of ["kind", "G", "n", "c", "s", "view"]) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
