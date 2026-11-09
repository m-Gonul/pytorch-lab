// M4.1 — 2-B konvolüsyon, hücre hücre: girdi (dolgu dahil), çekirdek, çıktı. Bir çıktı hücresine tıkla: alıcı alanı ve
// Σ w·x toplamı gösterilir. Korelasyon (PyTorch conv2d) ile matematiksel konvolüsyon (çekirdek 180° döndürülür) farkı.
export const IMAGES = {
  "kare çerçeve": (i, j) => ((i === 2 || i === 6) && j >= 2 && j <= 6) || ((j === 2 || j === 6) && i >= 2 && i <= 6) ? 1 : 0,
  "artı": (i, j) => (i === 4 || j === 4 ? 1 : 0),
  "köşegen": (i, j) => (i === j ? 1 : 0),
  "tek nokta (dürtü)": (i, j) => (i === 4 && j === 4 ? 1 : 0),
  "yatay kenar (üst yarı parlak)": (i) => (i < 4 ? 1 : 0),
  "dama": (i, j) => ((i + j) % 2),
};
export const KERNELS = {
  "özdeşlik": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  "kutu bulanıklığı (1/9)": [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)),
  "Gauss (1/16)": [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map((r) => r.map((v) => v / 16)),
  "keskinleştirme (½ · [−1 … 10 … −1])": [[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]].map((r) => r.map((v) => v / 2)),
  "Sobel yatay kenar": [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
  "Sobel dikey kenar": [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
  "kaydırma (asimetrik: sağdaki komşu)": [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
};
const N = 9;
// dolgu: 0 geçerli (valid, P = 0), 1 sıfır, 2 kenar tekrarı (replicate), 3 aynalama (reflect)
export function padAt(X, i, j, mode) {
  const H = X.length, W = X[0].length;
  if (i >= 0 && i < H && j >= 0 && j < W) return X[i][j];
  if (mode === 1 || mode === 0) return 0;
  if (mode === 2) return X[Math.min(H - 1, Math.max(0, i))][Math.min(W - 1, Math.max(0, j))];
  const ref = (k, n) => (k < 0 ? -k : k >= n ? 2 * (n - 1) - k : k);
  return X[ref(i, H)][ref(j, W)];
}
export function conv2d(X, K, { pad = 1, stride = 1, flip = false } = {}) {
  const k = K.length, P = pad === 0 ? 0 : (k - 1) / 2, H = X.length, W = X[0].length;
  const Ho = Math.floor((H + 2 * P - k) / stride) + 1, Wo = Math.floor((W + 2 * P - k) / stride) + 1;
  const w = (u, v) => (flip ? K[k - 1 - u][k - 1 - v] : K[u][v]);
  const Y = Array.from({ length: Ho }, (_, a) => Array.from({ length: Wo }, (_, b) => { let s = 0; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) s += w(u, v) * padAt(X, a * stride - P + u, b * stride - P + v, pad); return s; }));
  return { Y, P, Ho, Wo, w };
}
const PADS = ["geçerli (dolgu yok)", "sıfır dolgu", "kenar tekrarı (replicate)", "aynalama (reflect)"];

export default {
  custom: true,
  keys: ["img", "ker", "pad", "stride", "flip", "a", "b"],
  selfTest(assert) {
    const X = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => IMAGES["kare çerçeve"](i, j)));
    // çıktı boyu ⌊(H + 2P − K)/S⌋ + 1
    assert.equal(conv2d(X, KERNELS["özdeşlik"], { pad: 0 }).Ho, 7); assert.equal(conv2d(X, KERNELS["özdeşlik"], { pad: 1, stride: 2 }).Ho, 5);
    // özdeşlik çekirdeği sıfır dolguyla görüntüyü korur
    const I = conv2d(X, KERNELS["özdeşlik"], { pad: 1 }).Y; assert.ok(I.every((r, i) => r.every((v, j) => v === X[i][j])));
    // konvolüsyon = çevrilmiş çekirdekle korelasyon; kaydırma çekirdeği korelasyonda sola, konvolüsyonda sağa kaydırır
    const D = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => IMAGES["tek nokta (dürtü)"](i, j)));
    const c = conv2d(D, KERNELS["kaydırma (asimetrik: sağdaki komşu)"], { pad: 1 }).Y, f = conv2d(D, KERNELS["kaydırma (asimetrik: sağdaki komşu)"], { pad: 1, flip: true }).Y;
    assert.equal(c[4][3], 1); assert.equal(f[4][5], 1);
    // dürtü yanıtı: konvolüsyonda çekirdeğin kendisi
    const G = conv2d(D, KERNELS["Sobel dikey kenar"], { pad: 1, flip: true }).Y; assert.equal(G[3][3], 1); assert.equal(G[4][3], 2); assert.equal(G[3][5], -1);
    // aynalama: indeks −1 → 1
    assert.equal(padAt([[1, 2, 3]], 0, -1, 3), 2); assert.equal(padAt([[1, 2, 3]], 0, -1, 2), 1);
  },
  mount(el, ctx) {
    const st = { img: 0, ker: 4, pad: 1, stride: 1, flip: 0, a: 2, b: 2 }, imgs = Object.keys(IMAGES), kers = Object.keys(KERNELS);
    el.innerHTML = `<div class="lab-bar">
      <label>Girdi <select data-k="img">${imgs.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select></label>
      <label>Çekirdek <select data-k="ker">${kers.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select></label>
      <label>Dolgu <select data-k="pad">${PADS.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select></label>
      <label>Adım (stride) <select data-k="stride"><option value="1">1</option><option value="2">2</option></select></label>
      <label>İşlem <select data-k="flip"><option value="0">çapraz korelasyon (PyTorch conv2d)</option><option value="1">konvolüsyon (çekirdek 180° döner)</option></select></label></div>
      <div class="cv-wrap"><div><p class="nx-cap">girdi x (kesikli: dolgu)</p><div class="cv-in"></div></div><div><p class="nx-cap">uygulanan çekirdek</p><div class="cv-k"></div></div><div><p class="nx-cap">çıktı y — bir hücreye tıkla</p><div class="cv-out"></div></div></div>
      <div class="nx-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s), f = (v) => ctx.fmt(Math.abs(v) < 1e-12 ? 0 : v, 3);
    const shade = (v, m) => `background:color-mix(in srgb, var(${v >= 0 ? "--c-a" : "--c-b"}) ${Math.round((Math.min(1, Math.abs(v) / (m || 1))) * 70)}%, transparent)`;
    function draw() {
      for (const k of ["img", "ker", "pad", "stride", "flip"]) $(`[data-k=${k}]`).value = st[k];
      const X = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => IMAGES[imgs[st.img]](i, j))), K = KERNELS[kers[st.ker]];
      const R = conv2d(X, K, { pad: st.pad, stride: st.stride, flip: !!st.flip }), P = R.P, k = K.length;
      st.a = Math.min(st.a, R.Ho - 1); st.b = Math.min(st.b, R.Wo - 1);
      const i0 = st.a * st.stride - P, j0 = st.b * st.stride - P, inWin = (i, j) => i >= i0 && i < i0 + k && j >= j0 && j < j0 + k;
      // girdi (dolgu halkasıyla)
      let h = `<div class="cv-grid" style="grid-template-columns:repeat(${N + 2 * P},1fr)">`;
      for (let i = -P; i < N + P; i++) for (let j = -P; j < N + P; j++) { const v = padAt(X, i, j, st.pad), out = i < 0 || j < 0 || i >= N || j >= N; h += `<span class="cv-c${out ? " pad" : ""}${inWin(i, j) ? " win" : ""}" style="${shade(v, 1)}">${f(v)}</span>`; }
      $(".cv-in").innerHTML = h + "</div>";
      $(".cv-k").innerHTML = `<div class="cv-grid" style="grid-template-columns:repeat(${k},1fr)">${Array.from({ length: k * k }, (_, t) => { const w = R.w(Math.floor(t / k), t % k); return `<span class="cv-c win" style="${shade(w, Math.max(...K.flat().map(Math.abs)))}">${f(w)}</span>`; }).join("")}</div>`;
      const m = Math.max(1e-9, ...R.Y.flat().map(Math.abs));
      $(".cv-out").innerHTML = `<div class="cv-grid" style="grid-template-columns:repeat(${R.Wo},1fr)">${R.Y.map((row, a) => row.map((v, b) => `<button type="button" class="cv-c${a === st.a && b === st.b ? " sel" : ""}" data-a="${a}" data-b="${b}" style="${shade(v, m)}" aria-label="çıktı (${a}, ${b}) = ${f(v)}">${f(v)}</button>`).join("")).join("")}</div>`;
      const terms = []; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) { const w = R.w(u, v), x = padAt(X, i0 + u, j0 + v, st.pad); if (w !== 0 && x !== 0) terms.push(`${f(w)}·${f(x)}`); }
      $(".nx-read").innerHTML = `<p>Çıktı boyu ⌊(H + 2P − K)/S⌋ + 1 = ⌊(${N} + ${2 * P} − ${k})/${st.stride}⌋ + 1 = <b>${R.Ho} × ${R.Wo}</b>.</p>
        <p>y[${st.a}, ${st.b}] = Σ<sub>u,v</sub> w[u, v] · x[${st.a}·${st.stride} − ${P} + u, ${st.b}·${st.stride} − ${P} + v] = ${terms.length ? terms.join(" + ") : "0 (sıfır olmayan çarpım yok)"} = <b>${f(R.Y[st.a][st.b])}</b></p>
        <p class="small">${st.flip ? "Konvolüsyon: çekirdek yatay ve dikey çevrilerek uygulanıyor (yukarıdaki 'uygulanan çekirdek' çevrilmiş hâli). Simetrik çekirdeklerde (kutu, Gauss, keskinleştirme) korelasyonla aynı sonuç; Sobel'de işaret, kaydırma çekirdeğinde yön değişir." : "PyTorch'un <code>nn.Conv2d</code>'si (ve neredeyse tüm derin öğrenme kütüphaneleri) aslında çapraz korelasyon hesaplar: çekirdek çevrilmez. Ağırlıklar öğrenildiği için bu fark önemsizdir — öğrenilen çekirdek çevrilmiş hâliyle aynı işi görür."}${st.pad === 0 ? " Dolgu yok: çıktı her kenardan (K − 1)/2 hücre küçülür." : st.pad === 1 ? " Sıfır dolgu kenarlarda yapay bir 'karanlık çerçeve' ekler (Sobel'de kenar yanıtı oluşur)." : ""}</p>`;
      for (const b of el.querySelectorAll(".cv-out .cv-c")) b.addEventListener("click", () => { st.a = +b.dataset.a; st.b = +b.dataset.b; draw(); el.querySelector(`.cv-out .cv-c[data-a="${st.a}"][data-b="${st.b}"]`)?.focus(); });
    }
    for (const k of ["img", "ker", "pad", "stride", "flip"]) $(`[data-k=${k}]`).addEventListener("change", (e) => { st[k] = +e.target.value; draw(); });
    const set = (p) => { for (const k of Object.keys(st)) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
