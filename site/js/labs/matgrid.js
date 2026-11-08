// M0.2 — Matris = doğrusal dönüşüm: sütunları (î ve ĵ'nin görüntüleri) sürükle, ızgara ve birim kare dönüşsün;
// x vektörünü sürükle, Ax'in sütun görünümü (x₁·sütun₁ + x₂·sütun₂) ve satır görünümü (iç çarpımlar) canlı.
const S = 36, H = 5.5;
const X = (x) => 200 + x * S, Y = (y) => 200 - y * S;
const f2 = (v) => (Math.abs(v) < 5e-3 ? "0" : v.toFixed(2).replace("-", "−"));

export function apply(M, v) { return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]; }
export const det = (M) => M[0][0] * M[1][1] - M[0][1] * M[1][0];

export default {
  custom: true,
  keys: ["a", "b", "c", "d", "x1", "x2"],
  selfTest(assert) {
    const M = [[2, 1], [0, 1]];
    assert.deepEqual(apply(M, [1, 0]), [2, 0], "Aê₁ = birinci sütun");
    assert.deepEqual(apply(M, [0, 1]), [1, 1], "Aê₂ = ikinci sütun");
    assert.equal(det(M), 2);
    const R = [[0, -1], [1, 0]], v = apply(M, [3, -2]), w = apply(R, v), RM = [[R[0][0] * M[0][0] + R[0][1] * M[1][0], R[0][0] * M[0][1] + R[0][1] * M[1][1]], [R[1][0] * M[0][0] + R[1][1] * M[1][0], R[1][0] * M[0][1] + R[1][1] * M[1][1]]];
    assert.deepEqual(apply(RM, [3, -2]), w, "bileşke = çarpım");
    assert.equal(det(RM), det(R) * det(M), "det(RM) = det R · det M");
  },
  mount(el, ctx) {
    const st = { M: [[1, 1], [0, 1]], x: [2, 1] };
    const set = (p) => {
      if (p.a != null) st.M[0][0] = p.a; if (p.b != null) st.M[0][1] = p.b; if (p.c != null) st.M[1][0] = p.c; if (p.d != null) st.M[1][1] = p.d;
      if (p.x1 != null) st.x = [p.x1, p.x2 ?? st.x[1]];
    };
    set(ctx.init);
    const init = JSON.parse(JSON.stringify(st));
    el.innerHTML = `<div class="lab-bar"><button class="btn ghost small" type="button" data-k="reset">Başlangıca dön</button><span class="muted small">Mavi ve turuncu uçlar matrisin sütunlarıdır; yeşil uç x vektörüdür.</span></div>
      <div class="dd-wrap"><svg class="dd" viewBox="0 0 400 400" role="application" aria-label="Doğrusal dönüşüm ızgarası; sütun uçlarına ve x'e odaklanıp ok tuşlarıyla taşıyabilirsin">
        <defs>${["a", "b", "p", "q"].map((k) => `<marker id="mg-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="mg-${k}f"/></marker>`).join("")}</defs>
        <g class="mg-base"></g><g class="mg-layer"></g>
        <circle class="dd-h dd-ha" data-h="c1" r="10" tabindex="0" role="slider" aria-label="birinci sütun (î'nin görüntüsü)"/>
        <circle class="dd-h dd-hb" data-h="c2" r="10" tabindex="0" role="slider" aria-label="ikinci sütun (ĵ'nin görüntüsü)"/>
        <circle class="dd-h mg-hx" data-h="x" r="9" tabindex="0" role="slider" aria-label="x vektörü"/>
      </svg><div class="dd-read" aria-live="polite"></div></div>`;
    const svg = el.querySelector("svg");
    let base = "";
    for (let i = -5; i <= 5; i++) base += `<line x1="${X(i)}" y1="${Y(-H)}" x2="${X(i)}" y2="${Y(H)}" class="dd-gl"/><line x1="${X(-H)}" y1="${Y(i)}" x2="${X(H)}" y2="${Y(i)}" class="dd-gl"/>`;
    base += `<line x1="${X(-H)}" y1="${Y(0)}" x2="${X(H)}" y2="${Y(0)}" class="dd-ax"/><line x1="${X(0)}" y1="${Y(-H)}" x2="${X(0)}" y2="${Y(H)}" class="dd-ax"/>`;
    el.querySelector(".mg-base").innerHTML = base;
    function draw() {
      const M = st.M, c1 = [M[0][0], M[1][0]], c2 = [M[0][1], M[1][1]], D = det(M), y = apply(M, st.x);
      let L = "";
      for (let i = -6; i <= 6; i++) {
        const p = apply(M, [i, -6]), q = apply(M, [i, 6]), r = apply(M, [-6, i]), s = apply(M, [6, i]);
        L += `<line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(q[0])}" y2="${Y(q[1])}" class="mg-g${i ? "" : " mg-g0"}"/><line x1="${X(r[0])}" y1="${Y(r[1])}" x2="${X(s[0])}" y2="${Y(s[1])}" class="mg-g${i ? "" : " mg-g0"}"/>`;
      }
      const sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map((v) => apply(M, v));
      L += `<polygon points="${sq.map((v) => X(v[0]) + "," + Y(v[1])).join(" ")}" class="${D >= 0 ? "mg-sq" : "mg-sqn"}"/>`;
      // sütun görünümü: x1·c1 sonra x2·c2 uç uca
      const p1 = [st.x[0] * c1[0], st.x[0] * c1[1]];
      L += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(p1[0])}" y2="${Y(p1[1])}" class="mg-path"/><line x1="${X(p1[0])}" y1="${Y(p1[1])}" x2="${X(y[0])}" y2="${Y(y[1])}" class="mg-path2"/>`;
      L += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(c1[0])}" y2="${Y(c1[1])}" class="dd-a" marker-end="url(#mg-a)"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(c2[0])}" y2="${Y(c2[1])}" class="dd-b" marker-end="url(#mg-b)"/>`;
      L += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(st.x[0])}" y2="${Y(st.x[1])}" class="mg-x" marker-end="url(#mg-q)"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(y[0])}" y2="${Y(y[1])}" class="mg-y" marker-end="url(#mg-p)"/>`;
      L += `<text class="dd-lbl dd-at" x="${X(c1[0]) + 9}" y="${Y(c1[1]) + 16}">Aê₁</text><text class="dd-lbl dd-bt" x="${X(c2[0]) + 9}" y="${Y(c2[1]) - 8}">Aê₂</text><text class="dd-lbl mg-xt" x="${X(st.x[0]) + 9}" y="${Y(st.x[1]) - 8}">x</text><text class="dd-lbl mg-yt" x="${X(y[0]) + 9}" y="${Y(y[1]) - 8}">Ax</text>`;
      el.querySelector(".mg-layer").innerHTML = L;
      for (const [h, v] of [["c1", c1], ["c2", c2], ["x", st.x]]) { const c = el.querySelector(`[data-h=${h}]`); c.setAttribute("cx", X(v[0])); c.setAttribute("cy", Y(v[1])); c.setAttribute("aria-valuetext", `(${f2(v[0])}, ${f2(v[1])})`); }
      const kind = Math.abs(D) < 1e-6 ? "<b>tekil</b>: düzlem bir doğruya (ya da noktaya) çöktü; bilgi kaybı var, ters dönüşüm yok" : D < 0 ? "<b>negatif</b>: yönelim ters döndü (ayna görüntüsü), alan |det| katına çıktı" : `alanlar <b>${f2(D)}</b> katına çıkar`;
      el.querySelector(".dd-read").innerHTML = `<p class="mg-mat">A = <span class="mg-m"><span>${f2(M[0][0])}</span><span>${f2(M[0][1])}</span><span>${f2(M[1][0])}</span><span>${f2(M[1][1])}</span></span> &nbsp; x = (${f2(st.x[0])}, ${f2(st.x[1])})</p>
        <table class="lab-tbl">
        <tr><td>Satır görünümü</td><td>Ax = (${f2(M[0][0])}·${f2(st.x[0])} + ${f2(M[0][1])}·${f2(st.x[1])},&nbsp; ${f2(M[1][0])}·${f2(st.x[0])} + ${f2(M[1][1])}·${f2(st.x[1])})</td></tr>
        <tr><td>Sütun görünümü</td><td>Ax = ${f2(st.x[0])}·(${f2(c1[0])}, ${f2(c1[1])}) + ${f2(st.x[1])}·(${f2(c2[0])}, ${f2(c2[1])})</td></tr>
        <tr><td>Sonuç</td><td><b>(${f2(y[0])}, ${f2(y[1])})</b></td></tr>
        <tr><td>det A = ad − bc</td><td>${f2(D)}</td></tr></table>
        <p class="small">Determinantın işareti/büyüklüğü: ${kind}. Gri gölge birim karenin görüntüsü.</p>`;
    }
    const toWorld = (ev) => { const r = svg.getBoundingClientRect(); return [(((ev.clientX - r.left) / r.width) * 400 - 200) / S, (200 - ((ev.clientY - r.top) / r.height) * 400) / S]; };
    const rd = (v) => Math.max(-5, Math.min(5, Math.round(v * 4) / 4));
    const put = (h, v) => { v = v.map(rd); if (h === "c1") { st.M[0][0] = v[0]; st.M[1][0] = v[1]; } else if (h === "c2") { st.M[0][1] = v[0]; st.M[1][1] = v[1]; } else st.x = v; draw(); };
    const cur = (h) => (h === "c1" ? [st.M[0][0], st.M[1][0]] : h === "c2" ? [st.M[0][1], st.M[1][1]] : st.x);
    for (const c of el.querySelectorAll(".dd-h")) {
      const h = c.dataset.h;
      c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); c.classList.add("drag"); });
      c.addEventListener("pointermove", (e) => { if (c.hasPointerCapture(e.pointerId)) put(h, toWorld(e)); });
      c.addEventListener("pointerup", (e) => { c.releasePointerCapture(e.pointerId); c.classList.remove("drag"); });
      c.addEventListener("keydown", (e) => { const d = e.shiftKey ? 1 : 0.25, m = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[e.key]; if (m) { e.preventDefault(); const v = cur(h); put(h, [v[0] + m[0], v[1] + m[1]]); } });
    }
    el.querySelector("[data-k=reset]").addEventListener("click", () => { st.M = JSON.parse(JSON.stringify(init.M)); st.x = [...init.x]; draw(); });
    draw();
    return { set(p) { set(p); draw(); } };
  },
};
