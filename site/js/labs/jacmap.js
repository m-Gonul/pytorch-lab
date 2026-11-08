// M0.3 — Jacobian = yerel doğrusal yaklaşım. Bir p noktasının etrafındaki küçük kare, doğrusal olmayan g altında eğri bir
// şekle gider; J(p) ile çarpılmış kare (paralelkenar) ise onun doğrusal yaklaşımıdır. ε küçüldükçe ikisi çakışır.
const n3 = (v) => (Math.abs(v) < 5e-4 ? "0" : Number(v.toPrecision(3)).toString().replace("-", "−"));

export const MAPS = [
  { name: "g(x, y) = (xy, x + y²)", g: ([x, y]) => [x * y, x + y * y], J: ([x, y]) => [[y, x], [1, 2 * y]], inR: [[-2.5, 2.5], [-2.5, 2.5]], outR: [[-6, 6], [-4, 8]], p: [1, 1], ax: ["x", "y"] },
  { name: "kutupsal → kartezyen (r, θ) ↦ (r cos θ, r sin θ)", g: ([r, t]) => [r * Math.cos(t), r * Math.sin(t)], J: ([r, t]) => [[Math.cos(t), -r * Math.sin(t)], [Math.sin(t), r * Math.cos(t)]], inR: [[0, 3], [-3.1416, 3.1416]], outR: [[-3.2, 3.2], [-3.2, 3.2]], p: [2, 0.8], ax: ["r", "θ"] },
  { name: "girdap: yarıçapla artan döndürme", g: ([x, y]) => { const a = 0.6 * Math.hypot(x, y), c = Math.cos(a), s = Math.sin(a); return [c * x - s * y, s * x + c * y]; },
    J: (p) => { const h = 1e-6, g = MAPS[2].g; const dx = [g([p[0] + h, p[1]]), g([p[0] - h, p[1]])], dy = [g([p[0], p[1] + h]), g([p[0], p[1] - h])]; return [[(dx[0][0] - dx[1][0]) / (2 * h), (dy[0][0] - dy[1][0]) / (2 * h)], [(dx[0][1] - dx[1][1]) / (2 * h), (dy[0][1] - dy[1][1]) / (2 * h)]]; },
    inR: [[-2.5, 2.5], [-2.5, 2.5]], outR: [[-2.8, 2.8], [-2.8, 2.8]], p: [1.5, 0.5], ax: ["x", "y"] },
  { name: "doğrusal: (2x + y, x + y)", g: ([x, y]) => [2 * x + y, x + y], J: () => [[2, 1], [1, 1]], inR: [[-2.5, 2.5], [-2.5, 2.5]], outR: [[-7, 7], [-5, 5]], p: [0.5, 0.5], ax: ["x", "y"] },
];
export function numJ(M, p, h = 1e-5) {
  const a = M.g([p[0] + h, p[1]]), b = M.g([p[0] - h, p[1]]), c = M.g([p[0], p[1] + h]), d = M.g([p[0], p[1] - h]);
  return [[(a[0] - b[0]) / (2 * h), (c[0] - d[0]) / (2 * h)], [(a[1] - b[1]) / (2 * h), (c[1] - d[1]) / (2 * h)]];
}
// küçük karenin (p ± ε) g altındaki görüntüsü ile doğrusal yaklaşım arasındaki en büyük göreli fark, alan oranı
export function linErr(M, p, eps) {
  const J = M.J(p), g0 = M.g(p);
  let worst = 0, scale = 0;
  for (let k = 0; k < 32; k++) {
    const t = (2 * Math.PI * k) / 32, d = [eps * Math.cos(t), eps * Math.sin(t)], gv = M.g([p[0] + d[0], p[1] + d[1]]), lin = [J[0][0] * d[0] + J[0][1] * d[1], J[1][0] * d[0] + J[1][1] * d[1]];
    worst = Math.max(worst, Math.hypot(gv[0] - g0[0] - lin[0], gv[1] - g0[1] - lin[1])); scale = Math.max(scale, Math.hypot(...lin));
  }
  return scale > 0 ? worst / scale : 0;
}
function area(pts) { let s = 0; for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; }
export function imageArea(M, p, eps, n = 60) {
  const pts = [];
  const side = (a, b) => { for (let i = 0; i < n; i++) { const t = i / n; pts.push(M.g([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])); } };
  const c = [[p[0] - eps, p[1] - eps], [p[0] + eps, p[1] - eps], [p[0] + eps, p[1] + eps], [p[0] - eps, p[1] + eps]];
  for (let i = 0; i < 4; i++) side(c[i], c[(i + 1) % 4]);
  return area(pts) / (4 * eps * eps);
}

const SZ = 300;
export default {
  custom: true,
  keys: ["map", "px", "py", "eps", "zoom"],
  selfTest(assert) {
    const M = MAPS[0], J = M.J([2, 3]), N = numJ(M, [2, 3]);
    assert.deepEqual(J, [[3, 2], [1, 6]], "elle hesaplanan Jacobian (modül alıştırması)");
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) assert.ok(Math.abs(J[i][j] - N[i][j]) < 1e-6);
    const P = MAPS[1], Jp = P.J([2, 0.7]); assert.ok(Math.abs(Jp[0][0] * Jp[1][1] - Jp[0][1] * Jp[1][0] - 2) < 1e-12, "kutupsal: det J = r");
    assert.ok(linErr(MAPS[3], [0.3, -1], 1) < 1e-12, "doğrusal dönüşümde yaklaşım hatasız");
    assert.ok(linErr(M, [1, 1], 0.01) < linErr(M, [1, 1], 0.1) / 5, "ε küçüldükçe hata orantılı azalır");
    assert.ok(Math.abs(imageArea(M, [1, 1], 0.01) - (1 * 2 - 1 * 1)) < 1e-3, "küçük karenin alan oranı ≈ det J");
  },
  mount(el, ctx) {
    const st = { m: 0, p: [...MAPS[0].p], eps: 0.4, zoom: false };
    const set = (q) => {
      if (q.map != null && q.map !== st.m) { st.m = q.map; st.p = [...MAPS[st.m].p]; }
      if (q.px != null) st.p[0] = q.px; if (q.py != null) st.p[1] = q.py; if (q.eps != null) st.eps = q.eps; if (q.zoom != null) st.zoom = !!q.zoom;
    };
    set(ctx.init);
    el.innerHTML = `<div class="lab-bar"><label>Dönüşüm <select data-k="map">${MAPS.map((M, i) => `<option value="${i}">${M.name}</option>`).join("")}</select></label>
      <label>Kare yarı-kenarı ε <input type="range" data-k="eps" min="-2" max="0" step="0.05" aria-label="ε, log ölçek"> <output data-k="epsv"></output></label>
      <label><input type="checkbox" data-k="zoom"> Çıktıyı g(p) etrafında yakınlaştır</label></div>
      <div class="jm-wrap"><figure class="jm-pane"><figcaption>Girdi düzlemi (p'yi sürükle)</figcaption><svg class="jm" data-k="in" viewBox="0 0 ${SZ} ${SZ}" role="application" aria-label="Girdi düzlemi; p noktasına odaklanıp ok tuşlarıyla taşı"></svg></figure>
      <figure class="jm-pane"><figcaption>Çıktı düzlemi: g ile görüntü (mavi) ve J(p) ile doğrusal yaklaşım (turuncu kesikli)</figcaption><svg class="jm" data-k="out" viewBox="0 0 ${SZ} ${SZ}" role="img" aria-label="Çıktı düzlemi"></svg></figure></div>
      <div class="jm-read" aria-live="polite"></div>`;
    const sIn = el.querySelector("[data-k=in]"), sOut = el.querySelector("[data-k=out]"), read = el.querySelector(".jm-read");
    const epsIn = el.querySelector("[data-k=eps]");
    const T = (R) => ({ X: (x) => ((x - R[0][0]) / (R[0][1] - R[0][0])) * SZ, Y: (y) => SZ - ((y - R[1][0]) / (R[1][1] - R[1][0])) * SZ });
    const poly = (pts, tr, cls) => `<polyline points="${pts.map((v) => tr.X(v[0]).toFixed(1) + "," + tr.Y(v[1]).toFixed(1)).join(" ")}" class="${cls}"/>`;
    function draw() {
      const M = MAPS[st.m], p = st.p, e = st.eps, J = M.J(p), g0 = M.g(p), D = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      const ti = T(M.inR);
      let a = "";
      const [[x0, x1], [y0, y1]] = M.inR;
      for (let k = 0; k <= 10; k++) { const x = x0 + ((x1 - x0) * k) / 10, y = y0 + ((y1 - y0) * k) / 10; a += `<line x1="${ti.X(x)}" y1="0" x2="${ti.X(x)}" y2="${SZ}" class="dd-gl"/><line x1="0" y1="${ti.Y(y)}" x2="${SZ}" y2="${ti.Y(y)}" class="dd-gl"/>`; }
      a += `<text x="${SZ - 6}" y="${SZ - 6}" text-anchor="end" class="jm-ax">${M.ax[0]} →</text><text x="6" y="14" class="jm-ax">↑ ${M.ax[1]}</text>`;
      const sq = (k, axis) => { const pts = []; for (let i = 0; i <= 24; i++) { const t = -e + (2 * e * i) / 24, c = -e + (2 * e * k) / 4; pts.push(axis ? [p[0] + c, p[1] + t] : [p[0] + t, p[1] + c]); } return pts; };
      for (let k = 0; k <= 4; k++) a += poly(sq(k, 0), ti, "jm-sq") + poly(sq(k, 1), ti, "jm-sq");
      a += `<circle class="dd-h dd-ha" data-h="p" r="8" cx="${ti.X(p[0])}" cy="${ti.Y(p[1])}" tabindex="0" role="slider" aria-label="p noktası" aria-valuetext="(${n3(p[0])}, ${n3(p[1])})"/>`;
      sIn.innerHTML = a;
      // çıktı
      let R = M.outR;
      if (st.zoom) { const s = Math.max(e * 2.4 * Math.max(Math.hypot(J[0][0], J[1][0]), Math.hypot(J[0][1], J[1][1]), 0.05), 1e-6); R = [[g0[0] - s, g0[0] + s], [g0[1] - s, g0[1] + s]]; }
      const to = T(R);
      let b = "";
      if (!st.zoom) for (let k = 0; k <= 10; k++) {
        const c = x0 + ((x1 - x0) * k) / 10, d = y0 + ((y1 - y0) * k) / 10, L1 = [], L2 = [];
        for (let i = 0; i <= 40; i++) { L1.push(M.g([c, y0 + ((y1 - y0) * i) / 40])); L2.push(M.g([x0 + ((x1 - x0) * i) / 40, d])); }
        b += poly(L1, to, "jm-bg") + poly(L2, to, "jm-bg");
      } else b += `<line x1="${SZ / 2}" y1="0" x2="${SZ / 2}" y2="${SZ}" class="dd-gl"/><line x1="0" y1="${SZ / 2}" x2="${SZ}" y2="${SZ / 2}" class="dd-gl"/>`;
      for (let k = 0; k <= 4; k++) b += poly(sq(k, 0).map(M.g), to, "jm-img") + poly(sq(k, 1).map(M.g), to, "jm-img");
      const lin = (d) => [g0[0] + J[0][0] * d[0] + J[0][1] * d[1], g0[1] + J[1][0] * d[0] + J[1][1] * d[1]];
      b += poly([[-e, -e], [e, -e], [e, e], [-e, e], [-e, -e]].map(lin), to, "jm-lin");
      b += `<circle cx="${to.X(g0[0])}" cy="${to.Y(g0[1])}" r="4.5" class="jm-gp"/><text x="${to.X(g0[0]) + 8}" y="${to.Y(g0[1]) - 8}" class="dd-lbl">g(p)</text>`;
      sOut.innerHTML = b;
      const N = numJ(M, p), er = linErr(M, p, e), ar = imageArea(M, p, e);
      read.innerHTML = `<table class="lab-tbl"><tbody>
        <tr><td>p = (${M.ax[0]}, ${M.ax[1]})</td><td>(${n3(p[0])}, ${n3(p[1])})</td><td>g(p)</td><td>(${n3(g0[0])}, ${n3(g0[1])})</td></tr>
        <tr><td>J(p) (formül)</td><td><span class="mg-m"><span>${n3(J[0][0])}</span><span>${n3(J[0][1])}</span><span>${n3(J[1][0])}</span><span>${n3(J[1][1])}</span></span></td><td>J(p) (sayısal, merkezi fark)</td><td><span class="mg-m"><span>${n3(N[0][0])}</span><span>${n3(N[0][1])}</span><span>${n3(N[1][0])}</span><span>${n3(N[1][1])}</span></span></td></tr>
        <tr><td>det J(p)</td><td>${n3(D)}</td><td>görüntü alanı / kare alanı</td><td>${n3(ar)}</td></tr>
        <tr><td>doğrusal yaklaşımın göreli hatası</td><td colspan="3">${(er * 100).toPrecision(2)} % &nbsp;<span class="muted small">(ε'yi 10 kat küçült: hata ≈ 10 kat azalır — birinci dereceden yaklaşımın kalanı O(ε²), göreli olarak O(ε))</span></td></tr></tbody></table>
        <p class="small">J'nin <b>sütunları</b>, girdide ${M.ax[0]} ve ${M.ax[1]} yönündeki küçük adımların çıktıda nereye gittiğidir (M0.2'deki matris = dönüşüm bakışı). ${Math.abs(D) < 1e-6 ? "<b>det J = 0</b>: bu noktada kare bir doğruya ezilir (yerel olarak tersinmez)." : D < 0 ? "det J < 0: yönelim burada ters dönüyor." : ""}${st.m === 3 ? " Doğrusal dönüşümde J her yerde aynı ve yaklaşım tam: hata 0." : ""}</p>`;
      el.querySelector("[data-k=epsv]").textContent = n3(e);
      bindHandle();
    }
    const toWorld = (ev) => { const r = sIn.getBoundingClientRect(), M = MAPS[st.m], [[x0, x1], [y0, y1]] = M.inR; return [x0 + ((ev.clientX - r.left) / r.width) * (x1 - x0), y1 - ((ev.clientY - r.top) / r.height) * (y1 - y0)]; };
    const clampP = (v) => { const [[x0, x1], [y0, y1]] = MAPS[st.m].inR; return [Math.max(x0, Math.min(x1, v[0])), Math.max(y0, Math.min(y1, v[1]))]; };
    let dragging = false;
    sIn.addEventListener("pointerdown", (e) => { dragging = true; sIn.setPointerCapture(e.pointerId); st.p = clampP(toWorld(e)); draw(); });
    sIn.addEventListener("pointermove", (e) => { if (dragging) { st.p = clampP(toWorld(e)); draw(); } });
    sIn.addEventListener("pointerup", () => { dragging = false; el.querySelector("[data-h=p]")?.focus({ preventScroll: true }); });
    function bindHandle() {
      const h = el.querySelector("[data-h=p]");
      h.addEventListener("keydown", (e) => { const R = MAPS[st.m].inR, d = ((R[0][1] - R[0][0]) / 100) * (e.shiftKey ? 5 : 1), m = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[e.key]; if (m) { e.preventDefault(); st.p = clampP([st.p[0] + m[0], st.p[1] + m[1]]); draw(); el.querySelector("[data-h=p]").focus({ preventScroll: true }); } });
    }
    const sync = () => { el.querySelector("[data-k=map]").value = st.m; epsIn.value = Math.log10(st.eps); el.querySelector("[data-k=zoom]").checked = st.zoom; };
    el.querySelector("[data-k=map]").addEventListener("change", (e) => { set({ map: Number(e.target.value) }); draw(); });
    epsIn.addEventListener("input", (e) => { st.eps = 10 ** Number(e.target.value); draw(); });
    el.querySelector("[data-k=zoom]").addEventListener("change", (e) => { st.zoom = e.target.checked; draw(); });
    sync(); draw();
    return { set(q) { set(q); sync(); draw(); } };
  },
};
