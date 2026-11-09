// M4.1 — im2col: her alıcı alanı bir sütuna aç, konvolüsyonu tek matris çarpımı yap. Bir sütuna tıkla → girdideki penceresi;
// bir girdi hücresine tıkla → matristeki tüm kopyaları (bellek K² kat şişer) ve col2im'in geri geçişte o hücreye kaç terim topladığı.
export function im2col(X, K, S) {
  const C = X.length, H = X[0].length, W = X[0][0].length, Ho = Math.floor((H - K) / S) + 1, Wo = Math.floor((W - K) / S) + 1, rows = [], src = [];
  for (let c = 0; c < C; c++) for (let u = 0; u < K; u++) for (let v = 0; v < K; v++) {
    const row = [], srow = [];
    for (let a = 0; a < Ho; a++) for (let b = 0; b < Wo; b++) { const i = a * S + u, j = b * S + v; row.push(X[c][i][j]); srow.push([c, i, j]); }
    rows.push(row); src.push(srow);
  }
  return { cols: rows, src, Ho, Wo };
}
// col2im: sütun matrisini girdi şekline geri topla (im2col'un eşleniği/adjoint'i)
export function col2im(M, src, C, H, W) {
  const X = Array.from({ length: C }, () => Array.from({ length: H }, () => new Array(W).fill(0)));
  M.forEach((row, r) => row.forEach((v, t) => { const [c, i, j] = src[r][t]; X[c][i][j] += v; }));
  return X;
}
const C = 2, H = 5, W = 5;
const X0 = Array.from({ length: C }, (_, c) => Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => (c === 0 ? i * W + j + 1 : -(i + j) % 3))));
const Wf = (K) => [Array.from({ length: C * K * K }, (_, t) => (t % 3 === 0 ? 1 : t % 3 === 1 ? 0 : -1)), Array.from({ length: C * K * K }, (_, t) => (t < K * K ? 1 / (K * K) : 0))];

export default {
  custom: true,
  keys: ["K", "S", "col", "ci", "ii", "jj"],
  selfTest(assert) {
    for (const [K, S] of [[2, 1], [3, 1], [3, 2]]) {
      const R = im2col(X0, K, S), Wm = Wf(K);
      // matris çarpımı = doğrudan korelasyon
      const direct = (f, a, b) => { let s = 0; for (let c = 0; c < C; c++) for (let u = 0; u < K; u++) for (let v = 0; v < K; v++) s += Wm[f][c * K * K + u * K + v] * X0[c][a * S + u][b * S + v]; return s; };
      for (let f = 0; f < 2; f++) for (let t = 0; t < R.Ho * R.Wo; t++) { const mm = R.cols.reduce((s, row, r) => s + Wm[f][r] * row[t], 0); assert.ok(Math.abs(mm - direct(f, Math.floor(t / R.Wo), t % R.Wo)) < 1e-12, "im2col · W = conv"); }
      // eşlenik: ⟨im2col(X), M⟩ = ⟨X, col2im(M)⟩
      const M = R.cols.map((row, r) => row.map((_, t) => Math.sin(1 + r * 7 + t * 3))), Y = col2im(M, R.src, C, H, W);
      const lhs = R.cols.reduce((s, row, r) => s + row.reduce((p, v, t) => p + v * M[r][t], 0), 0), rhs = X0.reduce((s, ch, c) => s + ch.reduce((p, row, i) => p + row.reduce((q, v, j) => q + v * Y[c][i][j], 0), 0), 0);
      assert.ok(Math.abs(lhs - rhs) < 1e-9, "col2im, im2col'un eşleniği");
    }
  },
  mount(el, ctx) {
    const st = { K: 2, S: 1, col: 0, ci: -1, ii: 0, jj: 0 };
    el.innerHTML = `<div class="lab-bar"><label>Çekirdek K <select data-k="K"><option value="2">2 × 2</option><option value="3">3 × 3</option></select></label>
      <label>Adım S <select data-k="S"><option value="1">1</option><option value="2">2</option></select></label></div>
      <p class="small muted">Girdi: C = ${C} kanal, ${H} × ${W}, dolgu yok. İki filtre: F₀ (+1, 0, −1 deseni), F₁ (yalnız 0. kanalın ortalaması). Bir <b>sütuna</b> ya da bir <b>girdi hücresine</b> tıkla.</p>
      <div class="ic-wrap"><div class="ic-in"></div><div class="ic-mat"></div></div><div class="nx-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s), f = (v) => ctx.fmt(Math.abs(v) < 1e-12 ? 0 : v, 3);
    function draw() {
      $("[data-k=K]").value = st.K; $("[data-k=S]").value = st.S;
      const K = st.K, S = st.S, R = im2col(X0, K, S), T = R.Ho * R.Wo, Wm = Wf(K); st.col = Math.min(st.col, T - 1);
      const selIn = st.ci >= 0, a = Math.floor(st.col / R.Wo), b = st.col % R.Wo;
      const inWin = (c, i, j) => !selIn && i >= a * S && i < a * S + K && j >= b * S && j < b * S + K;
      const cover = (c, i, j) => R.src.reduce((n, row) => n + row.filter(([cc, ii, jj]) => cc === c && ii === i && jj === j).length, 0);
      $(".ic-in").innerHTML = X0.map((ch, c) => `<p class="nx-cap">girdi kanalı ${c}</p><div class="cv-grid" style="grid-template-columns:repeat(${W},1fr)">${ch.map((row, i) => row.map((v, j) => `<button type="button" class="cv-c${inWin(c, i, j) ? " win" : ""}${selIn && st.ci === c && st.ii === i && st.jj === j ? " sel" : ""}" data-c="${c}" data-i="${i}" data-j="${j}" aria-label="kanal ${c}, (${i}, ${j}) = ${v}">${f(v)}</button>`).join("")).join("")}</div>`).join("");
      const out = [0, 1].map((fi) => Array.from({ length: T }, (_, t) => R.cols.reduce((s, row, r) => s + Wm[fi][r] * row[t], 0)));
      let h = `<p class="nx-cap">im2col matrisi (${C}·${K}·${K} = ${C * K * K} satır × ${T} sütun) ve çıktı = W · sütunlar</p><div class="ic-scroll"><table class="ic-t"><thead><tr><th>W₀</th><th>W₁</th><th></th>${Array.from({ length: T }, (_, t) => `<th><button type="button" class="ic-h${!selIn && t === st.col ? " sel" : ""}" data-t="${t}" aria-label="sütun ${t}">${t}</button></th>`).join("")}</tr></thead><tbody>`;
      R.cols.forEach((row, r) => { const [c, u, v] = [Math.floor(r / (K * K)), Math.floor((r % (K * K)) / K), r % K]; h += `<tr><td>${f(Wm[0][r])}</td><td>${f(Wm[1][r])}</td><th class="small">c${c} (${u},${v})</th>${row.map((val, t) => { const [cc, ii, jj] = R.src[r][t], hit = selIn ? cc === st.ci && ii === st.ii && jj === st.jj : t === st.col; return `<td class="${hit ? "hit" : ""}">${f(val)}</td>`; }).join("")}</tr>`; });
      h += `<tr class="ic-out"><td colspan="3">y₀ = W₀·sütun</td>${out[0].map((v, t) => `<td class="${!selIn && t === st.col ? "hit" : ""}">${f(v)}</td>`).join("")}</tr><tr class="ic-out"><td colspan="3">y₁ = W₁·sütun</td>${out[1].map((v, t) => `<td class="${!selIn && t === st.col ? "hit" : ""}">${f(v)}</td>`).join("")}</tr></tbody></table></div>`;
      $(".ic-mat").innerHTML = h;
      const total = R.cols.length * T, inN = C * H * W;
      $(".nx-read").innerHTML = selIn
        ? `<p>Girdi hücresi (kanal ${st.ci}, ${st.ii}, ${st.jj}) im2col matrisinde <b>${cover(st.ci, st.ii, st.jj)}</b> kez kopyalanmış (vurgulu hücreler). Geri geçişte col2im (PyTorch <code>F.fold</code>) bu ${cover(st.ci, st.ii, st.jj)} konumdaki gradyanları <b>toplayarak</b> bu hücrenin gradyanını verir — im2col bir kopyalama, col2im onun eşleniği olan toplamadır.</p>`
        : `<p>Sütun ${st.col} = çıktı konumu (${a}, ${b}) için alıcı alan (girdide çerçeveli), kanal kanal ve satır satır düzleştirilmiş. Çıktı y<sub>f</sub>[${a}, ${b}] = W<sub>f</sub> · sütun: y₀ = <b>${f(out[0][st.col])}</b>, y₁ = <b>${f(out[1][st.col])}</b>. Tüm çıktı tek bir (F × C·K²) · (C·K² × H_out·W_out) matris çarpımıdır — GPU'ların en iyi yaptığı iş.</p>`;
      $(".nx-read").innerHTML += `<p class="small">Bellek: girdi ${inN} sayı, im2col matrisi ${total} sayı (≈ ${f(total / inN)} kat). Adım 1'de bu oran ≈ K² = ${K * K}'ye yaklaşır: im2col hızı bellekle satın alır. cuDNN gibi kütüphaneler bu yüzden matrisi parça parça kurar ya da Winograd/FFT gibi başka algoritmalar seçer.</p>`;
      for (const btn of el.querySelectorAll(".ic-h")) btn.addEventListener("click", () => { st.col = +btn.dataset.t; st.ci = -1; draw(); });
      for (const btn of el.querySelectorAll(".ic-in .cv-c")) btn.addEventListener("click", () => { st.ci = +btn.dataset.c; st.ii = +btn.dataset.i; st.jj = +btn.dataset.j; draw(); });
    }
    for (const k of ["K", "S"]) $(`[data-k=${k}]`).addEventListener("change", (e) => { st[k] = +e.target.value; st.col = 0; draw(); });
    const set = (p) => { for (const k of Object.keys(st)) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
