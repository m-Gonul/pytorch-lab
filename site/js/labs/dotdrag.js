// M0.1 — Sürüklenebilir iki vektör: iç çarpım, açı, izdüşüm, işaret bölgeleri, birim çember.
// Uçları fareyle/dokunarak ya da odaklanıp ok tuşlarıyla (Shift: büyük adım) taşı.
const S = 40, H = 5; // ölçek (piksel/birim), görünür yarı-genişlik
const MODES = ["İzdüşüm (a'nın b üzerine)", "İşaret bölgeleri (b normal vektör)", "Birim vektörler (yalnız açı)"];

export function geom(a, b) {
  const dot = a[0] * b[0] + a[1] * b[1];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const cos = na && nb ? Math.max(-1, Math.min(1, dot / (na * nb))) : NaN;
  const sp = nb ? dot / nb : NaN; // skaler izdüşüm
  const pv = nb ? [(dot / (nb * nb)) * b[0], (dot / (nb * nb)) * b[1]] : [NaN, NaN]; // vektör izdüşüm
  const res = [a[0] - pv[0], a[1] - pv[1]];
  return { dot, na, nb, cos, deg: (Math.acos(cos) * 180) / Math.PI, sp, pv, res };
}

const f2 = (v) => (Number.isFinite(v) ? (Math.abs(v) < 5e-3 ? "0" : v.toFixed(2).replace("-", "−")) : "tanımsız");
const X = (x) => 200 + x * S, Y = (y) => 200 - y * S;

export default {
  custom: true,
  keys: ["ax", "ay", "bx", "by", "mode", "snap"],
  selfTest(assert) {
    const g = geom([3, 1], [1, 2]);
    assert.equal(g.dot, 5);
    assert.ok(Math.abs(g.na * g.nb * g.cos - g.dot) < 1e-12, "‖a‖‖b‖cosθ = a·b");
    assert.ok(Math.abs(g.res[0] * 1 + g.res[1] * 2) < 1e-12, "artık b'ye dik");
    assert.ok(Math.abs(geom([1, 0], [0, 4]).deg - 90) < 1e-9, "dik vektörler 90°");
    assert.ok(Number.isNaN(geom([1, 1], [0, 0]).cos), "sıfır vektörde açı tanımsız");
    for (let k = 0; k < 200; k++) { const r = () => Math.random() * 10 - 5; const q = geom([r(), r()], [r(), r()]); assert.ok(q.cos >= -1 && q.cos <= 1); }
  },
  mount(el, ctx) {
    const st = { a: [3, 1], b: [1, 2], mode: 0, snap: false };
    const apply = (p) => { if (p.ax != null) st.a = [p.ax, p.ay]; if (p.bx != null) st.b = [p.bx, p.by]; if (p.mode != null) st.mode = p.mode; if (p.snap != null) st.snap = !!p.snap; };
    apply(ctx.init);
    const init = JSON.parse(JSON.stringify(st));
    el.innerHTML = `
      <div class="lab-bar">
        <label>Görünüm <select data-k="mode">${MODES.map((m, i) => `<option value="${i}">${m}</option>`).join("")}</select></label>
        <label><input type="checkbox" data-k="snap"> Tam sayılara yapıştır (elle hesapla karşılaştırmak için)</label>
        <button class="btn ghost small" type="button" data-k="reset">Başlangıca dön</button>
      </div>
      <div class="dd-wrap">
        <svg class="dd" viewBox="0 0 400 400" role="application" aria-label="İki vektörlü düzlem. a ve b uçlarına odaklanıp ok tuşlarıyla taşıyabilirsin.">
          <defs>
            <marker id="dd-ha" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dd-a-fill"/></marker>
            <marker id="dd-hb" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dd-b-fill"/></marker>
            <marker id="dd-hp" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dd-p-fill"/></marker>
          </defs>
          <g class="dd-grid"></g><g class="dd-layer"></g>
          <circle class="dd-h dd-ha" r="11" tabindex="0" role="slider" aria-label="a vektörünün ucu"/>
          <circle class="dd-h dd-hb" r="11" tabindex="0" role="slider" aria-label="b vektörünün ucu"/>
        </svg>
        <div class="dd-read" aria-live="polite"></div>
      </div>`;
    const svg = el.querySelector("svg"), layer = el.querySelector(".dd-layer");
    let g = "";
    for (let i = -H; i <= H; i++) g += `<line x1="${X(i)}" y1="${Y(-H)}" x2="${X(i)}" y2="${Y(H)}" class="${i ? "dd-gl" : "dd-ax"}"/><line x1="${X(-H)}" y1="${Y(i)}" x2="${X(H)}" y2="${Y(i)}" class="${i ? "dd-gl" : "dd-ax"}"/>${i && i % 2 === 0 ? `<text x="${X(i)}" y="${Y(0) + 13}" class="dd-tk">${i}</text><text x="${X(0) - 6}" y="${Y(i) + 4}" class="dd-tk" text-anchor="end">${i}</text>` : ""}`;
    el.querySelector(".dd-grid").innerHTML = g;
    const sel = el.querySelector("[data-k=mode]"), snap = el.querySelector("[data-k=snap]");

    function draw() {
      let a = st.a, b = st.b;
      if (st.mode === 2) { const na = Math.hypot(...a) || 1, nb = Math.hypot(...b) || 1; a = [a[0] / na, a[1] / na]; b = [b[0] / nb, b[1] / nb]; }
      const q = geom(a, b);
      let L = "";
      if (st.mode === 1 && q.nb > 1e-9) {
        // b'ye dik doğru: x·b = 0; b tarafı (x·b > 0) gölgeli
        const n = [b[0] / q.nb, b[1] / q.nb], t = [-n[1], n[0]], R = 12;
        const p1 = [t[0] * R, t[1] * R], p2 = [-t[0] * R, -t[1] * R];
        L += `<polygon class="dd-pos" points="${[p1, [p1[0] + n[0] * R, p1[1] + n[1] * R], [p2[0] + n[0] * R, p2[1] + n[1] * R], p2].map((p) => X(p[0]) + "," + Y(p[1])).join(" ")}"/>`;
        L += `<line class="dd-hyp" x1="${X(p1[0])}" y1="${Y(p1[1])}" x2="${X(p2[0])}" y2="${Y(p2[1])}"/>`;
        L += `<text class="dd-lbl" x="${X(n[0] * 3.3 + t[0] * 2.2)}" y="${Y(n[1] * 3.3 + t[1] * 2.2)}">x·b > 0</text><text class="dd-lbl" x="${X(-n[0] * 3.3 + t[0] * 2.2)}" y="${Y(-n[1] * 3.3 + t[1] * 2.2)}">x·b &lt; 0</text>`;
      }
      if (st.mode === 2) L += `<circle cx="${X(0)}" cy="${Y(0)}" r="${S}" class="dd-unit"/>`;
      if (st.mode === 0 && q.nb > 1e-9) {
        L += `<line class="dd-perp" x1="${X(a[0])}" y1="${Y(a[1])}" x2="${X(q.pv[0])}" y2="${Y(q.pv[1])}"/>`;
        L += `<line class="dd-p" x1="${X(0)}" y1="${Y(0)}" x2="${X(q.pv[0])}" y2="${Y(q.pv[1])}" marker-end="url(#dd-hp)"/>`;
        // dik açı işareti
        const u = [b[0] / q.nb, b[1] / q.nb], s = 0.25 * Math.sign(q.dot || 1), r = q.res, nr = Math.hypot(...r);
        if (nr > 0.3) { const v = [r[0] / nr * 0.25, r[1] / nr * 0.25], c = [q.pv[0] - u[0] * s, q.pv[1] - u[1] * s]; L += `<polyline class="dd-ra" points="${X(c[0])},${Y(c[1])} ${X(c[0] + v[0])},${Y(c[1] + v[1])} ${X(q.pv[0] + v[0])},${Y(q.pv[1] + v[1])}"/>`; }
      }
      // açı yayı
      if (Number.isFinite(q.cos)) {
        const t1 = Math.atan2(a[1], a[0]), t2 = Math.atan2(b[1], b[0]);
        let d = t2 - t1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
        const r = 0.9, steps = 24, pts = [];
        for (let k = 0; k <= steps; k++) { const t = t1 + (d * k) / steps; pts.push(`${X(r * Math.cos(t))},${Y(r * Math.sin(t))}`); }
        L += `<polyline class="dd-arc" points="${pts.join(" ")}"/><text class="dd-lbl" x="${X(1.25 * Math.cos(t1 + d / 2))}" y="${Y(1.25 * Math.sin(t1 + d / 2)) + 4}" text-anchor="middle">θ</text>`;
      }
      L += `<line class="dd-a" x1="${X(0)}" y1="${Y(0)}" x2="${X(a[0])}" y2="${Y(a[1])}" marker-end="url(#dd-ha)"/><text class="dd-lbl dd-at" x="${X(a[0]) + 10}" y="${Y(a[1]) - 8}">a</text>`;
      L += `<line class="dd-b" x1="${X(0)}" y1="${Y(0)}" x2="${X(b[0])}" y2="${Y(b[1])}" marker-end="url(#dd-hb)"/><text class="dd-lbl dd-bt" x="${X(b[0]) + 10}" y="${Y(b[1]) - 8}">b</text>`;
      layer.innerHTML = L;
      for (const [cls, v] of [[".dd-ha", st.a], [".dd-hb", st.b]]) { const c = el.querySelector(cls); c.setAttribute("cx", X(v[0])); c.setAttribute("cy", Y(v[1])); c.setAttribute("aria-valuetext", `(${f2(v[0])}, ${f2(v[1])})`); }
      const sign = q.dot > 1e-9 ? "pozitif: açı 90°'den küçük, a b'nin 'önünde'" : q.dot < -1e-9 ? "negatif: açı 90°'den büyük, a b'nin 'arkasında'" : "sıfır: vektörler dik";
      el.querySelector(".dd-read").innerHTML = `<table class="lab-tbl">
        <tr><td>a</td><td>(${f2(a[0])}, ${f2(a[1])})</td></tr><tr><td>b</td><td>(${f2(b[0])}, ${f2(b[1])})</td></tr>
        <tr><td>a·b = a₁b₁ + a₂b₂</td><td>${f2(a[0])}·${f2(b[0])} + ${f2(a[1])}·${f2(b[1])} = <b>${f2(q.dot)}</b></td></tr>
        <tr><td>‖a‖, ‖b‖</td><td>${f2(q.na)}, ${f2(q.nb)}</td></tr>
        <tr><td>cos θ = a·b / (‖a‖‖b‖)</td><td>${Number.isFinite(q.cos) ? q.cos.toFixed(3).replace("-", "−") : "tanımsız (sıfır vektör)"}</td></tr>
        <tr><td>θ</td><td>${Number.isFinite(q.deg) ? q.deg.toFixed(1) + "°" : "—"}</td></tr>
        <tr><td>‖a‖‖b‖cos θ</td><td>${f2(q.na * q.nb * q.cos)} <span class="muted">(a·b ile aynı)</span></td></tr>
        <tr><td>a'nın b üzerine skaler izdüşümü a·b/‖b‖</td><td>${f2(q.sp)}</td></tr>
      </table><p class="small">İç çarpımın işareti <b>${sign}</b>.${st.mode === 2 ? " Birim vektörlerde a·b doğrudan cos θ'dır." : ""}</p>`;
      sel.value = st.mode; snap.checked = st.snap;
    }
    const toWorld = (ev) => { const r = svg.getBoundingClientRect(); const x = ((ev.clientX - r.left) / r.width) * 400, y = ((ev.clientY - r.top) / r.height) * 400; return [(x - 200) / S, (200 - y) / S]; };
    const clamp = (v) => Math.max(-H, Math.min(H, v));
    const put = (k, p) => { st[k] = p.map((v) => clamp(st.snap ? Math.round(v) : Math.round(v * 100) / 100)); draw(); };
    for (const k of ["a", "b"]) {
      const h = el.querySelector(k === "a" ? ".dd-ha" : ".dd-hb");
      h.addEventListener("pointerdown", (e) => { h.setPointerCapture(e.pointerId); h.classList.add("drag"); });
      h.addEventListener("pointermove", (e) => { if (h.hasPointerCapture(e.pointerId)) put(k, toWorld(e)); });
      h.addEventListener("pointerup", (e) => { h.releasePointerCapture(e.pointerId); h.classList.remove("drag"); });
      h.addEventListener("keydown", (e) => {
        const d = e.shiftKey ? 1 : st.snap ? 1 : 0.25, m = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[e.key];
        if (m) { e.preventDefault(); put(k, [st[k][0] + m[0], st[k][1] + m[1]]); }
      });
    }
    sel.addEventListener("change", () => { st.mode = Number(sel.value); draw(); });
    snap.addEventListener("change", () => { st.snap = snap.checked; if (st.snap) { st.a = st.a.map(Math.round); st.b = st.b.map(Math.round); } draw(); });
    el.querySelector("[data-k=reset]").addEventListener("click", () => { Object.assign(st, JSON.parse(JSON.stringify(init))); draw(); });
    draw();
    return { set(p) { apply(p); draw(); } };
  },
};
