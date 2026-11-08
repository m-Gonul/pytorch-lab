// M3.1 — Tarayıcıda micrograd: bir ifade yaz, Value grafiği kurulsun; topolojik sıra, ileri değerler ve geri geçişin gradyanları,
// sayısal türevle karşılaştırma. Desteklenen: + − * / ^ (sabit üs), tanh, exp, log, relu, sigmoid, parantez, değişkenler.
export class Value {
  constructor(data, prev = [], op = "", label = "") { this.data = data; this.grad = 0; this.prev = prev; this.op = op; this.label = label; this._back = () => {}; }
  add(o) { const v = new Value(this.data + o.data, [this, o], "+"); v._back = () => { this.grad += v.grad; o.grad += v.grad; }; return v; }
  mul(o) { const v = new Value(this.data * o.data, [this, o], "·"); v._back = () => { this.grad += o.data * v.grad; o.grad += this.data * v.grad; }; return v; }
  pow(k) { const v = new Value(this.data ** k, [this], `^${k}`); v._back = () => { this.grad += k * this.data ** (k - 1) * v.grad; }; return v; }
  neg() { return this.mul(new Value(-1, [], "", "−1")); }
  sub(o) { return this.add(o.neg()); }
  div(o) { return this.mul(o.pow(-1)); }
  fn(name) {
    const x = this.data, F = { tanh: [Math.tanh(x), (y) => 1 - y * y], exp: [Math.exp(x), (y) => y], log: [Math.log(x), () => 1 / x], relu: [Math.max(0, x), () => (x > 0 ? 1 : 0)], sigmoid: [1 / (1 + Math.exp(-x)), (y) => y * (1 - y)] }[name];
    const v = new Value(F[0], [this], name); v._back = () => { this.grad += F[1](v.data) * v.grad; }; return v;
  }
  topo() { const out = [], seen = new Set(); const go = (v) => { if (seen.has(v)) return; seen.add(v); v.prev.forEach(go); out.push(v); }; go(this); return out; }
  backward() { const T = this.topo(); for (const v of T) v.grad = 0; this.grad = 1; for (let i = T.length - 1; i >= 0; i--) T[i]._back(); return T; }
}
// özyinelemeli iniş ayrıştırıcı
export function parse(src, env) {
  const toks = src.match(/\s*([A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*(?:e-?\d+)?|\*\*|[-+*/^(),])/g)?.map((t) => t.trim()) || [];
  if (toks.join("").replace(/\s/g, "") !== src.replace(/\s/g, "")) throw new Error("Tanınmayan karakter var.");
  let i = 0; const peek = () => toks[i], take = (t) => { if (t && toks[i] !== t) throw new Error(`'${t}' bekleniyordu, '${toks[i] ?? "son"}' bulundu.`); return toks[i++]; };
  const vars = {};
  const expr = () => { let v = term(); while (peek() === "+" || peek() === "-") { const op = take(); const w = term(); v = op === "+" ? v.add(w) : v.sub(w); } return v; };
  const term = () => { let v = unary(); while (peek() === "*" || peek() === "/") { const op = take(); const w = unary(); v = op === "*" ? v.mul(w) : v.div(w); } return v; };
  const unary = () => (peek() === "-" ? (take(), unary().neg()) : power());
  const power = () => { const b = atom(); if (peek() === "^" || peek() === "**") { take(); const neg = peek() === "-" ? (take(), -1) : 1; const t = take(); if (!/^\d/.test(t ?? "")) throw new Error("Üs sabit bir sayı olmalı (ör. x^2)."); return b.pow(neg * Number(t)); } return b; };
  const atom = () => {
    const t = take();
    if (t === undefined) throw new Error("İfade erken bitti.");
    if (t === "(") { const v = expr(); take(")"); return v; }
    if (/^\d/.test(t)) return new Value(Number(t), [], "", t);
    if (["tanh", "exp", "log", "relu", "sigmoid"].includes(t)) { take("("); const v = expr(); take(")"); return v.fn(t); }
    if (!/^[A-Za-z_]/.test(t)) throw new Error(`Beklenmeyen '${t}': burada bir sayı, değişken ya da parantez olmalı.`);
    if (!(t in env)) throw new Error(`'${t}' tanımsız: değişkenleri aşağıda tanımla.`);
    if (!vars[t]) vars[t] = new Value(env[t], [], "", t);
    return vars[t];
  };
  const out = expr(); if (i < toks.length) throw new Error(`Fazladan '${toks[i]}'.`);
  return { out, vars };
}
export function numGrad(src, env, k, h = 1e-6) { const p = parse(src, { ...env, [k]: env[k] + h }).out.data, m = parse(src, { ...env, [k]: env[k] - h }).out.data; return (p - m) / (2 * h); }
const f4 = (v) => (Number.isFinite(v) ? (Math.abs(v) < 1e-12 ? "0" : Number(v.toPrecision(5)).toString().replace("-", "−")) : String(v));

const PRESETS = {
  karpathy: { e: "(a*b + c)*f", v: "a=2, b=-3, c=10, f=-2" },
  noron: { e: "tanh(x1*w1 + x2*w2 + b)", v: "x1=2, x2=0, w1=-3, w2=1, b=6.8813735870195432" },
  tekrar: { e: "x*x + x", v: "x=3" },
  bolme: { e: "(a + b)/(a - b)", v: "a=3, b=1" },
};
export default {
  custom: true,
  keys: ["preset"],
  selfTest(assert) {
    const r = parse("(a*b + c)*f", { a: 2, b: -3, c: 10, f: -2 }); r.out.backward();
    assert.equal(r.out.data, -8); assert.equal(r.vars.a.grad, 6); assert.equal(r.vars.b.grad, -4); assert.equal(r.vars.c.grad, -2); assert.equal(r.vars.f.grad, 4);
    const n = parse("tanh(x1*w1 + x2*w2 + b)", { x1: 2, x2: 0, w1: -3, w2: 1, b: 6.8813735870195432 }); n.out.backward();
    assert.ok(Math.abs(n.out.data - Math.SQRT1_2) < 1e-9); assert.ok(Math.abs(n.vars.w1.grad - 1) < 1e-9, "Karpathy: dL/dw1 = x1·(1 − o²) = 2·0.5");
    const t = parse("x*x + x", { x: 3 }); t.out.backward(); assert.equal(t.vars.x.grad, 7, "aynı değişken iki yerde: gradyanlar toplanır");
    const d = parse("(a + b)/(a - b)", { a: 3, b: 1 }); d.out.backward(); assert.ok(Math.abs(d.vars.a.grad - numGrad("(a + b)/(a - b)", { a: 3, b: 1 }, "a")) < 1e-6);
    assert.throws(() => parse("a + ", { a: 1 }));
  },
  mount(el, ctx) {
    el.innerHTML = `<div class="lab-bar eg-bar"><label class="eg-e">İfade <input type="text" data-k="e" spellcheck="false" autocomplete="off"></label>
      <label class="eg-v">Değişkenler <input type="text" data-k="v" spellcheck="false" autocomplete="off"></label>
      <button class="btn primary small" type="button" data-k="go">Grafiği kur ve geri yay</button></div>
      <p class="small muted">Örnek sözdizimi: <code>tanh(x*w + b)</code>, <code>(a+b)^2/c</code>, <code>exp(-x) + relu(y)</code>. Üs sabit olmalı.</p>
      <div class="eg-msg" aria-live="polite"></div><div class="gs-wrap"><svg class="gs eg-svg" viewBox="0 0 700 200" role="img" aria-label="Hesap grafiği"></svg></div><div class="eg-tbl"></div>`;
    const $ = (s) => el.querySelector(s);
    const run = () => {
      const src = $("[data-k=e]").value, vs = $("[data-k=v]").value, env = {};
      try {
        for (const part of vs.split(",").map((s) => s.trim()).filter(Boolean)) { const m = part.match(/^([A-Za-z_]\w*)\s*=\s*(-?[\d.]+(?:e-?\d+)?)$/); if (!m) throw new Error(`'${part}' anlaşılmadı: ad=sayı biçiminde yaz.`); env[m[1]] = Number(m[2]); }
        const { out, vars } = parse(src, env), T = out.backward();
        if (T.length > 60) throw new Error(`Grafik çok büyük (${T.length} düğüm); en fazla 60.`);
        // katman: girdilerden en uzun yol
        const lev = new Map(); for (const v of T) lev.set(v, v.prev.length ? 1 + Math.max(...v.prev.map((p) => lev.get(p))) : 0);
        const L = Math.max(...lev.values()), cols = Array.from({ length: L + 1 }, () => []); T.forEach((v) => cols[lev.get(v)].push(v));
        const W = Math.max(700, 120 * (L + 1) + 40), H = Math.max(160, 62 * Math.max(...cols.map((c) => c.length)) + 20), pos = new Map();
        cols.forEach((c, x) => c.forEach((v, y) => pos.set(v, [20 + x * 120, 10 + y * 62 + (H - 20 - 62 * c.length) / 2])));
        let g = `<defs><marker id="eg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="gs-arf"/></marker></defs>`;
        for (const v of T) for (const p of v.prev) { const [x1, y1] = pos.get(p), [x2, y2] = pos.get(v); g += `<line x1="${x1 + 96}" y1="${y1 + 24}" x2="${x2 - 2}" y2="${y2 + 24}" class="gs-e" marker-end="url(#eg-ar)"/>`; }
        T.forEach((v, k) => { const [x, y] = pos.get(v), name = v.label || v.op || "?"; g += `<g class="gs-n${v === out ? " cur" : ""}${!v.prev.length ? " inp" : ""}"><rect x="${x}" y="${y}" width="96" height="48" rx="6"/><text x="${x + 6}" y="${y + 15}" class="gs-l">#${k} ${name}</text><text x="${x + 6}" y="${y + 30}" class="gs-v">${f4(v.data)}</text><text x="${x + 6}" y="${y + 44}" class="gs-g">∇ ${f4(v.grad)}</text></g>`; });
        const svg = $(".eg-svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.minWidth = `${Math.min(W, 900)}px`; svg.innerHTML = g;
        $(".eg-tbl").innerHTML = `<table class="lab-tbl"><thead><tr><th>değişken</th><th>değer</th><th>geri yayılım ∂out/∂x</th><th>sayısal (merkezi fark)</th><th>fark</th></tr></thead><tbody>${Object.entries(vars).map(([k, v]) => { const nm = numGrad(src, env, k); return `<tr><td>${k}</td><td>${f4(v.data)}</td><td>${f4(v.grad)}</td><td>${f4(nm)}</td><td>${Math.abs(nm - v.grad).toExponential(1)}</td></tr>`; }).join("")}</tbody></table>`;
        $(".eg-msg").innerHTML = `<p class="small">Çıktı = <b>${f4(out.data)}</b>. ${T.length} düğüm, topolojik sırayla #0…#${T.length - 1} (her düğüm girdilerinden sonra gelir). Geri geçiş bu sıranın <b>tersinde</b> yürür; bir değişken birden çok yerde kullanıldıysa gradyanı her kullanımdan gelen katkıların toplamıdır. Çıkarma ve bölme, micrograd'daki gibi toplama, −1 ile çarpma ve −1. kuvvetle kurulur.</p>`;
      } catch (e) { $(".eg-msg").innerHTML = `<p class="lab-iss error">${e.message}</p>`; $(".eg-svg").innerHTML = ""; $(".eg-tbl").innerHTML = ""; }
    };
    const load = (k) => { const P = PRESETS[k] || PRESETS.karpathy; $("[data-k=e]").value = P.e; $("[data-k=v]").value = P.v; run(); };
    $("[data-k=go]").addEventListener("click", run);
    for (const inp of el.querySelectorAll("input")) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
    load(ctx.init.preset || "karpathy");
    return { set(p) { load(p.preset); } };
  },
};
