// M1.1 — Sürüklenebilir veri noktaları: en küçük kareler veriye nasıl tepki verir?
// Kapalı form doğrusu, ağırlık merkezi (x̄, ȳ) ve artıklar canlı güncellenir. Hesap gerçektir (kapalı form).
import { fmt, rng } from "./plot.js";

// çekirdek hesap: modül düzeyinde, selfTest burayı doğrular
export function ols(pts) {
  const n = pts.length;
  if (n < 2) return null;
  const mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
  let sxy = 0, sxx = 0;
  for (const p of pts) { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; }
  if (sxx < 1e-12) return null; // bütün x'ler aynı: eğim tanımsız (P6)
  const w = sxy / sxx, b = my - w * mx;
  const mse = pts.reduce((a, p) => a + (w * p[0] + b - p[1]) ** 2, 0) / n;
  const syy = pts.reduce((a, p) => a + (p[1] - my) ** 2, 0) / n;
  const r2 = syy > 1e-12 ? 1 - mse / syy : 0;
  return { w, b, mx, my, mse, r2 };
}

const XL = [-4, 4], YL = [-4, 5.5];
const W = 560, Hh = 380, L = 44, R = 16, T = 14, B = 34;
const sx = (v) => L + ((v - XL[0]) / (XL[1] - XL[0])) * (W - L - R);
const sy = (v) => T + (Hh - T - B) * (1 - (v - YL[0]) / (YL[1] - YL[0]));
const invx = (px) => XL[0] + ((px - L) / (W - L - R)) * (XL[1] - XL[0]);
const invy = (py) => YL[0] + (1 - (py - T) / (Hh - T - B)) * (YL[1] - YL[0]);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export default {
  custom: true,
  keys: ["c", "out", "n", "seed"],
  selfTest(assert) {
    const p = ols([[1, 2], [2, 4], [3, 5]]);
    assert.ok(Math.abs(p.w - 1.5) < 1e-12 && Math.abs(p.b - 0.6666666666666666) < 1e-12, "kapalı form üç noktada (1.5, 0.667)");
    // doğrunun ağırlık merkezinden geçtiği (normal denklem ⇒ b = ȳ − w x̄)
    assert.ok(Math.abs(p.w * p.mx + p.b - p.my) < 1e-12, "doğru (x̄, ȳ)'den geçer");
    // Pisagor: R² = açıklanan varyans oranı, artıklar x'e dik
    assert.ok(Math.abs(p.r2 - (1 - (1 / 18) / (42 / 27))) < 1e-6, "R² = 1 − MSE/Var(y)");  // 27/28 ≈ 0.9643
    assert.equal(ols([[2, 1], [2, 2], [2, 3]]), null, "bütün x'ler aynıysa eğim tanımsız (P6)");
    const q = ols([[1, 0], [2, 0], [3, 0], [4, 100]]);
    assert.ok(Number.isFinite(q.w) && Number.isFinite(q.b), "tek aykırı değerle bile sonlu çözüm");
    assert.ok(Math.abs(q.b + q.w * (10 / 4) - 100 / 4) < 1e-12, "aykırıda bile doğru (x̄, ȳ)'den geçer");
  },
  mount(el, ctx) {
    const st = { pts: [], seed: 1 };
    const reseed = (seed, c = 0, out = 0, n = 9) => {
      const r = rng(seed);
      const pts = [];
      for (let i = 0; i < n; i++) {
        const x = -3 + 6 * ((i + 0.5) / n) + (r() - 0.5) * 0.5 + c;
        const y = 1.2 * x + 0.4 + 0.7 * r.normal();
        pts.push([clamp(x, XL[0] + 0.2, XL[1] - 0.2), clamp(y, YL[0] + 0.2, YL[1] - 0.2)]);
      }
      if (out) { const i = Math.floor(r() * n); pts[i][1] = clamp(pts[i][1] + 3.4 * out, YL[1] - 0.2, YL[1] + 0); }
      return pts;
    };
    const apply = (p) => {
      if (p.c != null || p.out != null || p.n != null || p.seed != null) {
        const c = p.c ?? 0, out = p.out ?? 0, n = p.n ?? 9, seed = p.seed ?? st.seed;
        st.seed = seed; st.pts = reseed(seed, c, out, n);
      } else if (Array.isArray(p.pts)) st.pts = p.pts.map((q) => [clamp(q[0], XL[0] + 0.2, XL[1] - 0.2), clamp(q[1], YL[0] + 0.2, YL[1] - 0.2)]);
      if (p.seed != null) st.seed = p.seed;
    };
    apply(ctx.init);
    const init = JSON.parse(JSON.stringify({ pts: st.pts, seed: st.seed }));
    el.innerHTML = `
      <div class="lab-bar">
        <button class="btn small" type="button" data-k="rand">Rastgele veri</button>
        <button class="btn small" type="button" data-k="add">Nokta ekle</button>
        <button class="btn ghost small" type="button" data-k="del">Son noktayı sil</button>
        <button class="btn ghost small" type="button" data-k="reset">Başlangıca dön</button>
        <span class="muted small">Noktaları fareyle ya da dokunarak sürükle.</span>
      </div>
      <svg class="df" viewBox="0 0 ${W} ${Hh}" role="application" aria-label="Sürüklenebilir noktalar ve en küçük kareler doğrusu. Noktaları odaklanıp ok tuşlarıyla da taşıyabilirsin." tabindex="0">
        <g class="df-grid"></g><g class="df-layer"></g>
        <circle class="df-focus" r="12" opacity="0"/>
      </svg>
      <div class="df-read" aria-live="polite"></div>`;
    const svg = el.querySelector("svg"), layer = el.querySelector(".df-layer"), focus = el.querySelector(".df-focus");
    // ızgara
    let gr = "";
    for (let i = Math.ceil(XL[0]); i <= XL[1]; i++) gr += `<line x1="${sx(i)}" y1="${T}" x2="${sx(i)}" y2="${Hh - B}" class="df-gl"/><text x="${sx(i)}" y="${Hh - 16}" class="df-tk" text-anchor="middle">${i}</text>`;
    for (let j = Math.ceil(YL[0]); j <= YL[1]; j++) gr += `<line x1="${L}" y1="${sy(j)}" x2="${W - R}" y2="${sy(j)}" class="df-gl"/><text x="${L - 6}" y="${sy(j) + 4}" class="df-tk" text-anchor="end">${j}</text>`;
    el.querySelector(".df-grid").innerHTML = gr;
    let drag = -1;
    const read = el.querySelector(".df-read");
    function paint() {
      const p = ols(st.pts);
      let s = `<rect x="${L}" y="${T}" width="${W - L - R}" height="${Hh - T - B}" class="df-fr"/>`;
      // artıklar: noktadan doğruya dikey uzantı
      if (p) for (let i = 0; i < st.pts.length; i++) {
        const [x, y] = st.pts[i], yh = p.w * x + p.b;
        s += `<line x1="${sx(x)}" y1="${sy(y)}" x2="${sx(x)}" y2="${sy(yh)}" class="df-res"/>`;
      }
      if (p) {
        const xl = XL, y1 = p.w * xl[0] + p.b, y2 = p.w * xl[1] + p.b;
        s += `<line x1="${sx(xl[0])}" y1="${sy(y1)}" x2="${sx(xl[1])}" y2="${sy(y2)}" class="df-ln"/>`;
        s += `<circle cx="${sx(p.mx)}" cy="${sy(p.my)}" r="4.5" class="df-mean"/>`;
        s += `<line x1="${sx(p.mx)}" y1="${sy(p.my) - 9}" x2="${sx(p.mx)}" y2="${sy(p.my) - 3}" class="df-mean"/><text x="${sx(p.mx) + 8}" y="${sy(p.my) - 10}" class="df-lb">(x̄, ȳ)</text>`;
      }
      for (let i = 0; i < st.pts.length; i++) {
        const [x, y] = st.pts[i];
        s += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="${i === drag ? 9 : 6.5}" class="df-pt${i === drag ? " hot" : ""}" data-i="${i}" tabindex="0" role="slider" aria-label="Nokta ${i + 1}: x ${fmt(x)}, y ${fmt(y)}"/>`;
      }
      layer.innerHTML = s;
      read.innerHTML = p
        ? `<table class="lab-tbl"><tbody>
        <tr><td>kapalı form w, b</td><td><b>${fmt(p.w)}, ${fmt(p.b)}</b></td></tr>
        <tr><td>ağırlık merkezi (x̄, ȳ)</td><td>${fmt(p.mx)}, ${fmt(p.my)}</td></tr>
        <tr><td>MSE (uydurulan)</td><td>${fmt(p.mse)}</td></tr>
        <tr><td>R²</td><td>${fmt(p.r2)}</td></tr>
        <tr><td>n</td><td>${st.pts.length}</td></tr></tbody></table>`
        : `<p class="muted">Çözüm için en az 2 farklı x değerli nokta gerekiyor (P6: bütün x'ler aynıysa eğim tanımsız).</p>`;
    }
    const px = (e) => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
    const py = (e) => { const r = svg.getBoundingClientRect(); return (e.clientY - r.top) * Hh / r.height; };
    const move = (e, i) => {
      st.pts[i] = [clamp(invx(px(e)), XL[0] + 0.2, XL[1] - 0.2), clamp(invy(py(e)), YL[0] + 0.2, YL[1] - 0.2)];
      paint();
    };
    svg.addEventListener("pointerdown", (e) => {
      const c = e.target.closest(".df-pt"); if (!c) return;
      drag = Number(c.dataset.i); svg.setPointerCapture(e.pointerId); e.preventDefault(); paint();
    });
    svg.addEventListener("pointermove", (e) => { if (drag >= 0) move(e, drag); });
    svg.addEventListener("pointerup", () => { drag = -1; paint(); });
    svg.addEventListener("dblclick", (e) => {
      const x = clamp(invx(px(e)), XL[0] + 0.2, XL[1] - 0.2), y = clamp(invy(py(e)), YL[0] + 0.2, YL[1] - 0.2);
      st.pts.push([x, y]); paint();
    });
    svg.addEventListener("keydown", (e) => {
      const c = e.target.closest?.(".df-pt"); if (!c) return;
      const i = Number(c.dataset.i), s = e.shiftKey ? 0.5 : 0.1, m = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, s], ArrowDown: [0, -s] }[e.key];
      if (!m) return;
      e.preventDefault();
      st.pts[i] = [clamp(st.pts[i][0] + m[0], XL[0] + 0.2, XL[1] - 0.2), clamp(st.pts[i][1] + m[1], YL[0] + 0.2, YL[1] - 0.2)];
      paint();
    });
    el.addEventListener("click", (e) => {
      const k = e.target.dataset?.k;
      if (k === "rand") { st.seed = 1 + Math.floor(Math.random() * 99998); st.pts = reseed(st.seed); paint(); }
      if (k === "add") { const x = (XL[0] + XL[1]) / 2, y = (YL[0] + YL[1]) / 2; st.pts.push([x + (st.pts.length % 5) * 0.3 - 0.6, y]); paint(); }
      if (k === "del" && st.pts.length > 1) { st.pts.pop(); paint(); }
      if (k === "reset") { st.pts = JSON.parse(JSON.stringify(init.pts)); st.seed = init.seed; paint(); }
    });
    paint();
    return { set: (p) => { apply(p); paint(); } };
  },
};
