// M1.2 — Sürüklenebilir başlangıç: parabol f(x) = a·x²/2 üzerinde elle adım at.
// Custom şekil: kendi çizimi ve olayları; çekirdek hesap modül düzeyinde, selfTest doğrular.
import { fmt } from "./plot.js";

// çekirdek hesap: modül düzeyinde, selfTest burayı doğrular
export function step1(x, a, lr) { return (1 - lr * a) * x; }
export function regime(a, lr) {
  const r = 1 - lr * a;
  if (Math.abs(r) < 1e-12) return { r, name: "tek adımda çözüm (η = 1/a, Newton adımı)" };
  if (Math.abs(r) > 1) return { r, name: "ıraksama (ηa > 2): her adım |x|'i büyütür" };
  if (r < 0) return { r, name: "salınımlı yakınsama (1 < ηa < 2): işaret değiştirerek iner" };
  return { r, name: "tekdüze yakınsama (ηa < 1): işaret değiştirmeden iner" };
}
// merdiven: x_t → x_{t+1}; yatay parça f(x_t) seviyesinde, dikey parça eğriye iniş
export function ladder(x0, a, lr, T) {
  const xs = [x0], segs = [];
  for (let t = 0; t < T; t++) {
    const nx = step1(xs[t], a, lr);
    segs.push({ x1: xs[t], x2: nx, y: (a * xs[t] * xs[t]) / 2 });
    xs.push(nx);
    if (!Number.isFinite(nx) || Math.abs(nx) > 1e4) break;
  }
  return { xs, segs };
}

const XL = [-6, 6], W = 560, Hh = 380, L = 40, R = 16, Tm = 14, B = 34;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default {
  custom: true,
  keys: ["lr", "a", "x0"],
  selfTest(assert) {
    assert.equal(step1(4, 2, 0.3), 1.6, "çarpan 1 − ηa = 0.4");
    assert.equal(step1(0, 5, 0.2), 0, "minimumda gradyan sıfır");
    assert.equal(step1(3, 2, 0.5), 0, "η = 1/a tek adımda minimum");
    const lg = ladder(4, 2, 0.3, 3);
    assert.ok(lg.xs.every((x, i) => Math.abs(x - [4, 1.6, 0.64, 0.256][i]) < 1e-12), "merdiven üç adımda 4·0.4³");
    assert.equal(lg.segs.length, 3, "her adım bir yatay-dikey çifti");
    assert.equal(lg.segs[0].y, 16, "yatay parça f(x₀) = 2·4²/2 yüksekliğinde");
    const dv = ladder(1, 2, 1.2, 8);
    assert.ok(Math.abs(dv.xs[8]) > 10, "ηa = 2.4 > 2: ıraksama, |x| = 1.4⁸ ≈ 14.8");
    assert.ok(regime(2, 0.3).name.startsWith("tekdüze"), "ηa = 0.6 < 1: tekdüze");
    assert.ok(regime(2, 0.9).name.startsWith("salınımlı"), "ηa = 1.8: salınımlı");
    assert.ok(regime(2, 1.2).name.startsWith("ıraksama"), "ηa = 2.4: ıraksama");
    assert.ok(regime(2, 0.5).name.startsWith("tek adımda"), "ηa = 1: Newton");
  },
  mount(el, ctx) {
    const st = { a: 2, lr: 0.3, xs: [4], segs: [] };
    const apply = (p) => {
      if (p.a != null) st.a = clamp(p.a, 0.1, 10);
      if (p.lr != null) st.lr = clamp(p.lr, 0.01, 1.5);
      if (p.x0 != null) { st.xs = [clamp(p.x0, XL[0] + 0.2, XL[1] - 0.2)]; st.segs = []; }
    };
    apply(ctx.init || {});
    const init = JSON.parse(JSON.stringify({ a: st.a, lr: st.lr, x0: st.xs[0] }));
    el.innerHTML = `
      <div class="lab-bar">
        <label>η <input type="range" data-k="lr" min="0.01" max="1.5" step="0.01" value="${st.lr}" style="width:8em"> <b data-o="lr">${st.lr}</b></label>
        <label>a <input type="range" data-k="a" min="0.1" max="10" step="0.1" value="${st.a}" style="width:8em"> <b data-o="a">${st.a}</b></label>
        <button class="btn small" type="button" data-k="step">Adım</button>
        <button class="btn small" type="button" data-k="ten">10 adım</button>
        <button class="btn ghost small" type="button" data-k="reset">Başlangıca dön</button>
        <span class="muted small">Noktayı sürükleyerek yeni bir x₀ seç.</span>
      </div>
      <svg class="df" viewBox="0 0 ${W} ${Hh}" role="application" aria-label="f(x) = a·x²/2 parabolü üzerinde gradyan inişi merdiveni. Noktayı sürükleyip ok tuşlarıyla da taşıyabilirsin." tabindex="0">
        <g class="df-grid"></g><g class="gd-layer"></g>
      </svg>
      <div class="df-read" aria-live="polite"></div>`;
    const svg = el.querySelector("svg"), layer = el.querySelector(".gd-layer"), grid = el.querySelector(".df-grid");
    let yMax = 1;
    const sx = (v) => L + ((v - XL[0]) / (XL[1] - XL[0])) * (W - L - R);
    const sy = (v) => Tm + (Hh - Tm - B) * (1 - v / yMax);
    const invx = (px) => XL[0] + ((px - L) / (W - L - R)) * (XL[1] - XL[0]);
    const px = (e) => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
    function paint() {
      const f = (x) => (st.a * x * x) / 2;
      yMax = Math.max(2, f(6) * 1.06);
      let gr = "";
      for (let i = Math.ceil(XL[0]); i <= XL[1]; i += 2) gr += `<line x1="${sx(i)}" y1="${Tm}" x2="${sx(i)}" y2="${Hh - B}" class="df-gl"/><text x="${sx(i)}" y="${Hh - 16}" class="df-tk" text-anchor="middle">${i}</text>`;
      for (let j = 0; j <= 4; j++) { const yv = (yMax * j) / 4; gr += `<line x1="${L}" y1="${sy(yv)}" x2="${W - R}" y2="${sy(yv)}" class="df-gl"/><text x="${L - 6}" y="${sy(yv) + 4}" class="df-tk" text-anchor="end">${fmt(yv, 3)}</text>`; }
      grid.innerHTML = gr;
      let path = "";
      for (let i = 0; i <= 120; i++) { const x = XL[0] + ((XL[1] - XL[0]) * i) / 120, y = f(x); path += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(y).toFixed(1); }
      let s = `<path d="${path}" fill="none" class="df-ln" style="stroke-width:2"/>`;
      s += `<line x1="${sx(XL[0])}" y1="${sy(0)}" x2="${sx(XL[1])}" y2="${sy(0)}" class="df-fr"/>`;
      for (const g of st.segs.slice(-40)) {
        if (Number.isFinite(g.x2) && Math.abs(g.x2) <= 6.4) {
          s += `<line x1="${sx(g.x1)}" y1="${sy(g.y)}" x2="${sx(g.x2)}" y2="${sy(g.y)}" class="df-res"/>`;
          s += `<line x1="${sx(g.x2)}" y1="${sy(g.y)}" x2="${sx(g.x2)}" y2="${sy(f(g.x2))}" class="df-ln" style="stroke-width:1.4"/>`;
        }
      }
      // sıradaki adımın gölgesi
      const xc = st.xs.at(-1), nx = step1(xc, st.a, st.lr);
      if (Number.isFinite(nx) && Math.abs(nx) <= 6.4) {
        s += `<line x1="${sx(xc)}" y1="${sy(f(xc))}" x2="${sx(nx)}" y2="${sy(f(xc))}" class="gd-next"/>`;
        s += `<line x1="${sx(nx)}" y1="${sy(f(xc))}" x2="${sx(nx)}" y2="${sy(f(nx))}" class="df-res"/>`;
      }
      // eğim oku: yatay bileşen −η·f'(x)
      const gp = st.a * xc, arr = clamp(xc - 2.2 * st.lr * gp, XL[0], XL[1]);
      s += `<line x1="${sx(xc)}" y1="${sy(f(xc))}" x2="${sx(arr)}" y2="${sy(f(xc))}" class="df-mean"/>`;
      s += `<text x="${sx(xc) + 8}" y="${sy(f(xc)) - 10}" class="df-lb">x_${st.xs.length - 1} = ${fmt(xc, 3)}</text>`;
      s += `<circle cx="${sx(xc)}" cy="${sy(f(xc))}" r="8" class="df-pt" tabindex="0" role="slider" aria-label="Geçerli nokta x_${st.xs.length - 1} = ${fmt(xc, 3)}. Sürükleyerek ya da ok tuşlarıyla taşı."/>`;
      layer.innerHTML = s;
      const rg = regime(st.a, st.lr);
      el.querySelector(".df-read").innerHTML = `<table class="lab-tbl"><tbody>
        <tr><td>adım t</td><td><b>${st.xs.length - 1}</b></td></tr>
        <tr><td>x_t, f(x_t)</td><td>${fmt(xc, 4)}, ${fmt(f(xc), 4)}</td></tr>
        <tr><td>gradyan f'(x_t) = a·x_t</td><td>${fmt(gp, 4)}</td></tr>
        <tr><td>çarpan 1 − ηa</td><td>${fmt(rg.r, 4)}</td></tr>
        <tr><td>rejim</td><td>${rg.name}</td></tr></tbody></table>`;
      el.querySelector('[data-o="lr"]').textContent = st.lr;
      el.querySelector('[data-o="a"]').textContent = st.a;
    }
    let drag = false;
    svg.addEventListener("pointerdown", (e) => { drag = true; svg.setPointerCapture(e.pointerId); st.xs = [clamp(invx(px(e)), XL[0] + 0.2, XL[1] - 0.2)]; st.segs = []; e.preventDefault(); paint(); });
    svg.addEventListener("pointermove", (e) => { if (drag) { st.xs = [clamp(invx(px(e)), XL[0] + 0.2, XL[1] - 0.2)]; st.segs = []; paint(); } });
    svg.addEventListener("pointerup", () => { drag = false; });
    svg.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: -0.1, ArrowRight: 0.1 }[e.key];
      if (d == null) return;
      e.preventDefault();
      st.xs = [clamp(st.xs.at(-1) + (e.shiftKey ? 5 * d : d), XL[0] + 0.2, XL[1] - 0.2)];
      st.segs = [];
      paint();
    });
    el.addEventListener("input", (e) => {
      const k = e.target.dataset?.k;
      if (k === "lr") st.lr = Number(e.target.value);
      if (k === "a") st.a = Number(e.target.value);
      if (k) paint();
    });
    el.addEventListener("click", (e) => {
      const k = e.target.dataset?.k;
      if (k === "step" || k === "ten") {
        const n = k === "step" ? 1 : 10;
        for (let i = 0; i < n; i++) {
          const x = st.xs.at(-1), nx = step1(x, st.a, st.lr);
          if (!Number.isFinite(nx) || Math.abs(nx) > 1e6) break;
          st.segs.push({ x1: x, x2: nx, y: (st.a * x * x) / 2 });
          st.xs.push(nx);
        }
        paint();
      }
      if (k === "reset") { st.a = init.a; st.lr = init.lr; st.xs = [init.x0]; st.segs = []; el.querySelector('[data-k="lr"]').value = init.lr; el.querySelector('[data-k="a"]').value = init.a; paint(); }
    });
    paint();
    return { set: (p) => { apply(p); const r1 = el.querySelector('[data-k="lr"]'), r2 = el.querySelector('[data-k="a"]'); if (r1) r1.value = st.lr; if (r2) r2.value = st.a; paint(); } };
  },
};
