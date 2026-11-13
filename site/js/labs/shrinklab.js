// M1.3 â€” Tek koordinatta kÃ¼Ã§Ã¼ltme: lasso S(z,Î»), ridge z/(1+Î»), elastic net S(z,Î±Î»)/(1+(1âˆ’Î±)Î»).
// z'yi fareyle/dokunarak ya da odaklanÄ±p ok tuÅŸlarÄ±yla (Shift: bÃ¼yÃ¼k adÄ±m) taÅŸÄ±; Î» ve Î± kaydÄ±rÄ±cÄ±larla cezayÄ± ayarla.
const R = 4; // gÃ¶rÃ¼nÃ¼r aralÄ±k âˆ’R..R
const W = 440, H = 330, Lm = 46, Rm = 16, Tm = 14, Bm = 40;
const sx = (v) => Lm + ((v + R) / (2 * R)) * (W - Lm - Rm);
const sy = (v) => Tm + (H - Tm - Bm) * (1 - (v + R) / (2 * R));

export function soft(z, t) { const v = Math.sign(z) * Math.max(Math.abs(z) - t, 0); return v || 0; }
export const ridgeShrink = (z, lam) => z / (1 + lam);
// elastic net: amaÃ§ Â½(wâˆ’z)Â² + Î»â‚|w| + Î»â‚‚/2 wÂ², Î»â‚ = Î±Î», Î»â‚‚ = (1âˆ’Î±)Î»
export function enetShrink(z, lam, alpha) { return soft(z, alpha * lam) / (1 + (1 - alpha) * lam); }

export default {
  custom: true,
  keys: ["z", "lam", "alpha"],
  selfTest(assert) {
    assert.equal(soft(1.5, 0.5), 1);
    assert.equal(soft(-0.3, 0.5), 0);
    assert.equal(soft(-2, 0.5), -1.5);
    assert.ok(Math.abs(ridgeShrink(2, 2) - 2 / 3) < 1e-12, "ridge: z/(1+Î»)");
    for (const lam of [0.3, 1, 2.7]) for (const z of [-3, -0.7, 0, 0.2, 2.5]) {
      assert.ok(Math.abs(enetShrink(z, lam, 1) - soft(z, lam)) < 1e-12, "Î±=1 â†’ saf lasso");
      assert.ok(Math.abs(enetShrink(z, lam, 0) - ridgeShrink(z, lam)) < 1e-12, "Î±=0 â†’ saf ridge");
      assert.ok(Math.abs(enetShrink(-z, lam, 0.5) + enetShrink(z, lam, 0.5)) < 1e-12, "iÅŸaret simetrisi");
    }
    assert.ok(Math.abs(enetShrink(1.5, 2, 0.5) - 0.25) < 1e-12, "S(1.5,1)/(1+1) = 0.25");
    assert.ok(enetShrink(0.4, 1, 0.5) === 0, "|z| â‰¤ Î»â‚ â†’ tam sÄ±fÄ±r");
  },
  mount(el, ctx) {
    const init = { z: 1.5, lam: 1, alpha: 0.5 };
    const st = { ...init };
    const apply = (p) => {
      if (p.z != null) st.z = Math.max(-R, Math.min(R, p.z));
      if (p.lam != null) st.lam = Math.max(0, Math.min(3, p.lam));
      if (p.alpha != null) st.alpha = Math.max(0, Math.min(1, p.alpha));
    };
    apply(ctx.init);
    el.innerHTML = `
      <div class="lab-bar">
        <label>Î» (ceza gÃ¼cÃ¼) <input type="range" min="0" max="3" step="0.05" value="${st.lam}" data-k="lam"></label>
        <label>Î± (L1 payÄ±) <input type="range" min="0" max="1" step="0.05" value="${st.alpha}" data-k="alpha"></label>
        <button class="btn ghost small" type="button" data-k="reset">BaÅŸlangÄ±ca dÃ¶n</button>
      </div>
      <figure class="lab-plot"><svg viewBox="0 0 ${W} ${H}" role="application" aria-label="KÃ¼Ã§Ã¼ltme eÄŸrileri. z iÅŸaretÃ§isini odaklanÄ±p ok tuÅŸlarÄ±yla veya sÃ¼rÃ¼kleyerek taÅŸÄ±yabilirsin.">
        <g class="sl-grid"></g><g class="sl-layer"></g>
        <line class="sl-zh" x1="0" x2="0" y1="${sy(R)}" y2="${sy(-R)}"/>
        <circle class="sl-h" r="9" tabindex="0" role="slider" aria-label="z (cezasÄ±z Ã§Ã¶zÃ¼m)" aria-valuemin="${-R}" aria-valuemax="${R}"/>
      </svg><figcaption id="sl-cap"></figcaption></figure>
      <div class="sl-read" aria-live="polite"></div>`;
    const svg = el.querySelector("svg"), layer = el.querySelector(".sl-layer"), grid = el.querySelector(".sl-grid"), handle = el.querySelector(".sl-h"), zh = el.querySelector(".sl-zh");
    let g = "";
    for (let i = -4; i <= 4; i++) {
      g += `<line class="grid" x1="${sx(i)}" x2="${sx(i)}" y1="${sy(R)}" y2="${sy(-R)}"/><line class="grid" x1="${sx(-R)}" x2="${sx(R)}" y1="${sy(i)}" y2="${sy(i)}"/>`;
      if (i) g += `<text class="tk" x="${sx(i)}" y="${H - 24}" text-anchor="middle">${i}</text><text class="tk" x="${sx(-R) - 6}" y="${sy(i) + 4}" text-anchor="end">${i}</text>`;
    }
    g += `<line class="mark" x1="${sx(-R)}" x2="${sx(R)}" y1="${sy(0)}" y2="${sy(0)}"/><line class="mark" x1="${sx(0)}" x2="${sx(0)}" y1="${sy(R)}" y2="${sy(-R)}"/>`;
    grid.innerHTML = g;
    const xs = [], zs = [];
    for (let z = -R; z <= R + 1e-9; z += 0.05) { zs.push(z); xs.push(soft(z, 0)); }
    const path = (ys) => zs.map((z, i) => `${i ? "L" : "M"}${sx(z).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join("");
    const identity = `<path class="s4 ln dash" d="${path(xs)}"/>`;
    const f1 = (v, d = 2) => (Math.abs(v) < 1e-9 ? "0" : Number(v.toPrecision(d)).toString().replace("-", "âˆ’"));

    function draw() {
      const lam1 = st.alpha * st.lam, lam2 = (1 - st.alpha) * st.lam;
      const wl = soft(st.z, st.lam), wr = ridgeShrink(st.z, st.lam), we = enetShrink(st.z, st.lam, st.alpha);
      let L = identity
        + `<path class="s1 ln" d="${path(zs.map((z) => soft(z, st.lam)))}"/>`
        + `<path class="s2 ln" d="${path(zs.map((z) => ridgeShrink(z, st.lam)))}"/>`
        + `<path class="s0 ln" d="${path(zs.map((z) => enetShrink(z, st.lam, st.alpha)))}"/>`;
      if (lam1 > 1e-9) L += `<rect class="sl-zero" x="${sx(-lam1)}" y="${sy(R)}" width="${sx(lam1) - sx(-lam1)}" height="${sy(-R) - sy(R)}"/>`;
      L += `<text class="tk" x="${sx(Math.min(lam1, R - 0.4)) + 4}" y="${sy(R) + 12}" text-anchor="start">|z| â‰¤ Î»â‚ = ${f1(lam1)} â†’ 0</text>`;
      layer.innerHTML = L;
      zh.setAttribute("x1", sx(st.z)); zh.setAttribute("x2", sx(st.z));
      handle.setAttribute("cx", sx(st.z)); handle.setAttribute("cy", sy(we));
      handle.setAttribute("aria-valuenow", st.z);
      el.querySelector("#sl-cap").innerHTML = `<span class="lg"><i class="s1"></i>lasso <code>${f1(wl, 3)}</span></span><span class="lg"><i class="s2"></i>ridge <code>${f1(wr, 3)}</span></span><span class="lg"><i class="s0"></i>elastic net <code>${f1(we, 3)}</span></span>`;
      el.querySelector(".sl-read").innerHTML = `z = <b class="mono">${f1(st.z, 3)}</b>, Î» = ${f1(st.lam)}, Î± = ${f1(st.alpha)} â†’ Î»â‚ = ${f1(lam1)}, Î»â‚‚ = ${f1(lam2)}. Elastic net: Åµ = S(z, Î»â‚)/(1+Î»â‚‚) = <b class="mono">${f1(we, 3)}</b>${we === 0 ? " â€” <b>tam sÄ±fÄ±r</b>: Ã¶znitelik seÃ§ildi dÄ±ÅŸarÄ±." : ""}`;
    }
    const sync = () => {
      for (const k of ["lam", "alpha"]) el.querySelector(`[data-k="${k}"]`).value = st[k];
      draw();
    };
    el.querySelector(".lab-bar").addEventListener("input", (e) => { const k = e.target.dataset.k; if (k) { st[k] = Number(e.target.value); draw(); } });
    el.querySelector("[data-k=reset]").addEventListener("click", () => { Object.assign(st, init); apply(ctx.init); sync(); });
    const setZ = (v) => { st.z = Math.max(-R, Math.min(R, Math.round(v * 100) / 100)); draw(); };
    const zx = (e) => { const r = svg.getBoundingClientRect(); return -R + ((e.clientX - r.left) / r.width * W - Lm) / (W - Lm - Rm) * 2 * R; };
    svg.addEventListener("pointerdown", (e) => { svg.setPointerCapture(e.pointerId); setZ(zx(e)); });
    svg.addEventListener("pointermove", (e) => { if (e.buttons) setZ(zx(e)); });
    handle.addEventListener("keydown", (e) => {
      const d = e.shiftKey ? 0.25 : 0.05;
      if (e.key === "ArrowLeft") { setZ(st.z - d); e.preventDefault(); }
      if (e.key === "ArrowRight") { setZ(st.z + d); e.preventDefault(); }
    });
    sync();
    return { set: (p) => { apply(p); sync(); } };
  },
};
