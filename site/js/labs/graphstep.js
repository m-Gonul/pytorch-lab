// M0.3 — Hesap grafiğinde adım adım ileri ve geri geçiş. Her geri adımda: (yukarıdan gelen gradyan) × (yerel türev),
// birden çok yoldan gelen katkılar toplanır. Sonda sayısal türevle karşılaştırma.
const sig = (z) => 1 / (1 + Math.exp(-z));
const n3 = (v) => { if (!Number.isFinite(v)) return "—"; const r = Math.abs(v) < 5e-5 ? 0 : Number(v.toPrecision(4)); return String(r).replace("-", "−"); };

// op: ileri değer, yerel türevler (her girdiye göre), metinler
const OPS = {
  add: { f: (a, b) => a + b, d: () => [1, 1], txt: (A, B) => `${A} + ${B}`, dt: () => ["1", "1"] },
  sub: { f: (a, b) => a - b, d: () => [1, -1], txt: (A, B) => `${A} − ${B}`, dt: () => ["1", "−1"] },
  mul: { f: (a, b) => a * b, d: (a, b) => [b, a], txt: (A, B) => `${A} · ${B}`, dt: (A, B, a, b) => [`${B} (= ${n3(b)})`, `${A} (= ${n3(a)})`] },
  fma: { f: (w, x, b) => w * x + b, d: (w, x) => [x, w, 1], txt: (W, X, B) => `${W} · ${X} + ${B}`, dt: (W, X, B, w, x) => [`${X} (= ${n3(x)})`, `${W} (= ${n3(w)})`, "1"] },
  sig: { f: sig, d: (z) => [sig(z) * (1 - sig(z))], txt: (Z) => `σ(${Z})`, dt: (Z, z) => [`σ(${Z})(1 − σ(${Z})) (= ${n3(sig(z) * (1 - sig(z)))})`] },
  sq: { f: (e) => e * e, d: (e) => [2 * e], txt: (E) => `${E}²`, dt: (E, e) => [`2${E} (= ${n3(2 * e)})`] },
  relu: { f: (q) => Math.max(q, 0), d: (q) => [q > 0 ? 1 : 0], txt: (Q) => `max(${Q}, 0)`, dt: (Q, q) => [`[${Q} > 0] (= ${q > 0 ? 1 : 0})`] },
};

export const EXPRS = [
  { name: "f = (x + y) · y", out: "f", vars: { x: 2, y: 3 },
    nodes: [{ id: "x", c: 0, r: 0 }, { id: "y", c: 0, r: 2 }, { id: "a", op: "add", ins: ["x", "y"], c: 2, r: 0.6 }, { id: "f", op: "mul", ins: ["a", "y"], c: 4, r: 1.2 }] },
  { name: "Nöron: L = (σ(w·x + b) − t)²", out: "L", vars: { w: 0.8, x: 1.5, b: -0.5, t: 1 },
    nodes: [{ id: "w", c: 0, r: 0 }, { id: "x", c: 0, r: 1 }, { id: "b", c: 0, r: 2 }, { id: "z", op: "fma", ins: ["w", "x", "b"], c: 1, r: 1 }, { id: "a", op: "sig", ins: ["z"], c: 2, r: 1 }, { id: "t", c: 2, r: 2.4 }, { id: "e", op: "sub", ins: ["a", "t"], c: 3, r: 1.5 }, { id: "L", op: "sq", ins: ["e"], c: 4, r: 1.5 }] },
  { name: "Kapı ve dallanma: f = max(x·y, 0) + x", out: "f", vars: { x: 2, y: -1 },
    nodes: [{ id: "x", c: 0, r: 0 }, { id: "y", c: 0, r: 2 }, { id: "q", op: "mul", ins: ["x", "y"], c: 1, r: 1.5 }, { id: "r", op: "relu", ins: ["q"], c: 2, r: 1.5 }, { id: "f", op: "add", ins: ["r", "x"], c: 4, r: 0.6 }] },
];

// Tam ileri + geri geçiş; steps: her adımın açıklaması ve o adımdan sonraki durum
export function run(E, vars) {
  const val = {}, grad = {}, steps = [], ops = E.nodes.filter((n) => n.op);
  for (const n of E.nodes) if (!n.op) val[n.id] = vars[n.id];
  const snap = (s) => steps.push({ ...s, val: { ...val }, grad: { ...grad } });
  snap({ phase: "başlangıç", msg: `Girdiler: ${E.nodes.filter((n) => !n.op).map((n) => `${n.id} = ${n3(val[n.id])}`).join(", ")}. "İleri" ile düğüm değerlerini soldan sağa hesapla.` });
  for (const n of ops) {
    const O = OPS[n.op], a = n.ins.map((i) => val[i]);
    val[n.id] = O.f(...a);
    snap({ phase: "ileri", node: n.id, msg: `${n.id} = ${O.txt(...n.ins)} = ${O.txt(...a.map(n3))} = <b>${n3(val[n.id])}</b>` });
  }
  grad[E.out] = 1;
  snap({ phase: "tohum", node: E.out, msg: `Geri geçiş çıktıdan başlar: ∂${E.out}/∂${E.out} = <b>1</b>. Ara değerler ileri geçişte saklandı; yerel türevler onlardan hesaplanır.` });
  for (const n of [...ops].reverse()) {
    const O = OPS[n.op], a = n.ins.map((i) => val[i]), loc = O.d(...a), lt = O.dt(...n.ins, ...a), up = grad[n.id] ?? 0, parts = [];
    n.ins.forEach((i, k) => {
      const c = up * loc[k], had = grad[i] != null;
      parts.push(`∂${E.out}/∂${i} ${had ? "+=" : "="} ∂${E.out}/∂${n.id} · ∂${n.id}/∂${i} = ${n3(up)} · ${lt[k]} = <b>${n3(c)}</b>${had ? ` → toplam <b>${n3(grad[i] + c)}</b> (ikinci yol: katkılar <b>toplanır</b>)` : ""}`);
      grad[i] = (grad[i] ?? 0) + c;
    });
    snap({ phase: "geri", node: n.id, edges: n.ins, msg: `<b>${n.id}</b> düğümünden girdilerine (yukarıdan gelen × yerel türev):<br>${parts.join("<br>")}` });
  }
  return { val, grad, steps };
}
export function numeric(E, vars, h = 1e-5) {
  const out = {};
  for (const k of Object.keys(vars)) {
    const p = run(E, { ...vars, [k]: vars[k] + h }).val[E.out], m = run(E, { ...vars, [k]: vars[k] - h }).val[E.out];
    out[k] = (p - m) / (2 * h);
  }
  return out;
}

const BW = 112, BH = 58, CX = (c) => 8 + c * 140, RY = (r) => 10 + r * 78;

export default {
  custom: true,
  keys: ["expr", "x", "y", "w", "b", "t", "step"],
  selfTest(assert) {
    const r0 = run(EXPRS[0], { x: 2, y: 3 });
    assert.equal(r0.val.f, 15); assert.equal(r0.grad.x, 3); assert.equal(r0.grad.y, 8, "iki yoldan gelen gradyan toplanır: 5 + 3");
    const v = { w: 0.8, x: 1.5, b: -0.5, t: 1 }, r1 = run(EXPRS[1], v), nm = numeric(EXPRS[1], v);
    for (const k in v) assert.ok(Math.abs(r1.grad[k] - nm[k]) < 1e-7, `nöron ∂L/∂${k}`);
    const r2 = run(EXPRS[2], { x: 2, y: -1 });
    assert.equal(r2.grad.x, 1, "ReLU kapalı: yalnız doğrudan yol"); assert.equal(r2.grad.y, 0);
    const r3 = run(EXPRS[2], { x: 2, y: 1.5 });
    assert.equal(r3.grad.x, 2.5, "ReLU açık: y + 1"); assert.equal(r3.grad.y, 2);
    assert.equal(r1.steps.length, 1 + 4 + 1 + 4, "başlangıç + ileri + tohum + geri");
  },
  mount(el, ctx) {
    let ei = Number(ctx.init.expr ?? 0), vars = { ...EXPRS[ei].vars }, step = 0;
    for (const k of Object.keys(vars)) if (ctx.init[k] != null) vars[k] = ctx.init[k];
    el.innerHTML = `<div class="lab-bar gs-bar"><label>İfade <select data-k="expr">${EXPRS.map((E, i) => `<option value="${i}">${E.name}</option>`).join("")}</select></label><span class="gs-vars"></span></div>
      <div class="lab-bar gs-nav"><button class="btn ghost small" type="button" data-k="first" aria-label="Başa dön">⏮ Baş</button><button class="btn ghost small" type="button" data-k="prev">◀ Geri</button><button class="btn primary small" type="button" data-k="next">İleri ▶</button><button class="btn ghost small" type="button" data-k="last">Son ⏭</button><span class="gs-step muted small" role="status"></span></div>
      <div class="gs-wrap"><svg class="gs" viewBox="0 0 690 220" role="img" aria-label="Hesap grafiği"></svg></div>
      <div class="gs-msg prose" aria-live="polite"></div><div class="gs-check"></div>
      <p class="small muted gs-key"><span class="gs-kv">yeşil</span> = ileri geçişteki değer, <span class="gs-kg">turuncu</span> = çıktının bu düğüme göre gradyanı.</p>`;
    const svg = el.querySelector("svg.gs"), msg = el.querySelector(".gs-msg"), chk = el.querySelector(".gs-check"), stp = el.querySelector(".gs-step");
    const varsBox = el.querySelector(".gs-vars");
    function buildVars() {
      varsBox.innerHTML = Object.keys(vars).map((k) => `<label class="gs-in">${k} <input type="number" step="0.1" data-v="${k}" value="${vars[k]}" aria-label="girdi ${k}"></label>`).join("");
      for (const inp of varsBox.querySelectorAll("input")) inp.addEventListener("input", (e) => { const v = Number(e.target.value); if (e.target.value !== "" && Number.isFinite(v)) { vars[e.target.dataset.v] = v; e.target.removeAttribute("aria-invalid"); draw(); } else e.target.setAttribute("aria-invalid", "true"); });
    }
    function draw() {
      const E = EXPRS[ei], R = run(E, vars), S = R.steps, s = S[Math.max(0, Math.min(step, S.length - 1))];
      step = S.indexOf(s);
      const pos = Object.fromEntries(E.nodes.map((n) => [n.id, [CX(n.c), RY(n.r)]]));
      const maxY = Math.max(...E.nodes.map((n) => RY(n.r))) + BH + 10;
      svg.setAttribute("viewBox", `0 0 690 ${maxY}`);
      let g = `<defs><marker id="gs-ar" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="gs-arf"/></marker></defs>`;
      for (const n of E.nodes) for (const i of n.ins || []) {
        const [x1, y1] = pos[i], [x2, y2] = pos[n.id], on = s.phase === "geri" && s.node === n.id;
        g += `<line x1="${x1 + BW}" y1="${y1 + BH / 2}" x2="${x2 - 2}" y2="${y2 + BH / 2}" class="gs-e${on ? " on" : ""}" marker-end="url(#gs-ar)"/>`;
      }
      for (const n of E.nodes) {
        const [x, y] = pos[n.id], v = s.val[n.id], gr = s.grad[n.id], cur = s.node === n.id, tgt = s.phase === "geri" && s.edges?.includes(n.id);
        const lbl = n.op ? `${n.id} = ${OPS[n.op].txt(...n.ins)}` : `${n.id} (girdi)`;
        g += `<g class="gs-n${cur ? " cur" : ""}${tgt ? " tgt" : ""}${n.op ? "" : " inp"}"><rect x="${x}" y="${y}" width="${BW}" height="${BH}" rx="7"/>
          <text x="${x + 8}" y="${y + 17}" class="gs-l">${lbl}</text>
          <text x="${x + 8}" y="${y + 34}" class="gs-v">${v != null ? "= " + n3(v) : "?"}</text>
          <text x="${x + 8}" y="${y + 50}" class="gs-g">${gr != null ? `∂${E.out}/∂${n.id} = ${n3(gr)}` : ""}</text></g>`;
      }
      svg.innerHTML = g;
      stp.textContent = `Adım ${step + 1} / ${S.length} — ${s.phase === "ileri" ? "ileri geçiş" : s.phase === "geri" ? "geri geçiş" : s.phase === "tohum" ? "geri geçiş başlıyor" : "başlangıç"}`;
      msg.innerHTML = `<p>${s.msg}</p>`;
      if (step === S.length - 1) {
        const nm = numeric(E, vars);
        chk.innerHTML = `<table class="lab-tbl"><thead><tr><th>girdi</th><th>geri yayılım</th><th>sayısal (merkezi fark, h = 10⁻⁵)</th><th>fark</th></tr></thead><tbody>${Object.keys(vars).map((k) => `<tr><td>∂${E.out}/∂${k}</td><td>${n3(R.grad[k] ?? 0)}</td><td>${n3(nm[k])}</td><td>${Math.abs((R.grad[k] ?? 0) - nm[k]).toExponential(1)}</td></tr>`).join("")}</tbody></table>`;
      } else chk.innerHTML = "";
      el.querySelector("[data-k=prev]").disabled = step === 0; el.querySelector("[data-k=first]").disabled = step === 0;
      el.querySelector("[data-k=next]").disabled = step === S.length - 1; el.querySelector("[data-k=last]").disabled = step === S.length - 1;
    }
    const total = () => run(EXPRS[ei], vars).steps.length;
    el.querySelector("[data-k=expr]").value = ei;
    el.querySelector("[data-k=expr]").addEventListener("change", (e) => { ei = Number(e.target.value); vars = { ...EXPRS[ei].vars }; step = 0; buildVars(); draw(); });
    el.querySelector("[data-k=first]").addEventListener("click", () => { step = 0; draw(); });
    el.querySelector("[data-k=prev]").addEventListener("click", () => { step = Math.max(0, step - 1); draw(); });
    el.querySelector("[data-k=next]").addEventListener("click", () => { step = Math.min(total() - 1, step + 1); draw(); });
    el.querySelector("[data-k=last]").addEventListener("click", () => { step = total() - 1; draw(); });
    el.addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return; if (e.key === "ArrowRight") { step = Math.min(total() - 1, step + 1); draw(); } else if (e.key === "ArrowLeft") { step = Math.max(0, step - 1); draw(); } });
    buildVars(); draw();
    return {
      set(p) {
        if (p.expr != null && p.expr !== ei) { ei = p.expr; vars = { ...EXPRS[ei].vars }; el.querySelector("[data-k=expr]").value = ei; }
        for (const k of Object.keys(vars)) if (p[k] != null) vars[k] = p[k];
        step = p.step == null ? 0 : p.step < 0 ? 1e9 : p.step;
        buildVars(); draw();
      },
    };
  },
};
