// Küçük, bağımlılıksız SVG çizici: çizgi, nokta, basamak, çubuk, ısı haritası (alan), gri görüntü.
// Renkler CSS sınıflarıyla (s0…s5) — açık/koyu temada aynı anlam.

export const fmt = (v, d = 4) => {
  if (v == null || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = v.toExponential(Math.max(0, d - 2)).split("e");
    const sup = String(Number(e)).replace(/[-0-9]/g, (c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]);
    const mm = String(Number(m)).replace("-", "−");
    return mm === "1" ? `10${sup}` : mm === "−1" ? `−10${sup}` : `${mm}·10${sup}`;
  }
  return String(Number(v.toPrecision(d))).replace("-", "−");
};

function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let uid = 0;
export function plot(P, opt = {}) {
  const W = opt.w || 520, H = opt.h || 300, L = 62, R = 14, T = 26, B = 44;
  const iw = W - L - R, ih = H - T - B;
  const all = P.series || [];
  const fin = (v) => Number.isFinite(v);
  let xs = [], ys = [];
  for (const s of all) { xs.push(...s.x.filter(fin)); ys.push(...s.y.filter((v) => fin(v) && (!P.ylog || v > 0))); }
  if (P.field) { xs.push(P.field.x0, P.field.x1); ys.push(P.field.y0, P.field.y1); }
  if (P.img) { xs.push(0, P.img.w); ys.push(0, P.img.h); }
  let [x0, x1] = P.xlim || [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = P.ylim || [Math.min(...ys), Math.max(...ys)];
  if (!fin(x0) || !fin(x1)) [x0, x1] = [0, 1];
  if (!fin(y0) || !fin(y1)) [y0, y1] = [0, 1];
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  if (y1 === y0) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
  if (!P.ylim && !P.ylog && !P.field && !P.img) { const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad; if (P.yzero && y0 > 0) y0 = 0; }
  const ly = (v) => (P.ylog ? Math.log10(v) : v);
  const Y0 = ly(y0), Y1 = ly(y1);
  const sx = (v) => L + ((v - x0) / (x1 - x0)) * iw;
  const sy = (v) => T + ih - ((ly(v) - Y0) / (Y1 - Y0)) * ih;
  const clipId = `clip${++uid}`;
  let g = `<defs><clipPath id="${clipId}"><rect x="${L}" y="${T}" width="${iw}" height="${ih}"/></clipPath></defs>`;
  // ısı haritası (ör. kayıp yüzeyi): log ölçekli renk
  if (P.field) {
    const F = P.field, dx = iw / F.nx, dy = ih / F.ny;
    const zs = F.z.filter(fin), lo = Math.min(...zs), hi = Math.max(...zs);
    const tr = (z) => (F.log ? Math.log(z - lo + 0.02 * (hi - lo) + 1e-12) : z);
    const tlo = tr(lo), thi = tr(hi);
    let r = "";
    for (let j = 0; j < F.ny; j++) for (let i = 0; i < F.nx; i++) {
      const t = (tr(F.z[j * F.nx + i]) - tlo) / (thi - tlo || 1);
      r += `<rect x="${(L + i * dx).toFixed(1)}" y="${(T + ih - (j + 1) * dy).toFixed(1)}" width="${(dx + 0.6).toFixed(1)}" height="${(dy + 0.6).toFixed(1)}" style="fill-opacity:${(0.05 + 0.5 * (1 - t)).toFixed(3)}" class="fld"/>`;
    }
    g += `<g>${r}</g>`;
  }
  if (P.img) {
    const c = document.createElement("canvas"); c.width = P.img.w; c.height = P.img.h;
    const ctx = c.getContext("2d"), im = ctx.createImageData(P.img.w, P.img.h);
    const q8 = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
    // img.rgb: data 3·w·h uzunluğunda [r, g, b, r, g, b, …] (0..1); yoksa gri seviye.
    for (let k = 0; k < P.img.w * P.img.h; k++) { if (P.img.rgb) im.data.set([q8(P.img.data[3 * k]), q8(P.img.data[3 * k + 1]), q8(P.img.data[3 * k + 2]), 255], 4 * k); else { const v = q8(P.img.data[k]); im.data.set([v, v, v, 255], 4 * k); } }
    ctx.putImageData(im, 0, 0);
    g += `<image href="${c.toDataURL()}" x="${L}" y="${T}" width="${iw}" height="${ih}" preserveAspectRatio="none" style="image-rendering:pixelated"/>`;
  }
  // eksenler ve ızgara
  if (!P.img) {
    for (const [t, lb] of P.xticks || ticks(x0, x1, opt.xt || 6).map((t) => [t, fmt(t, 3)])) g += `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${T}" y2="${T + ih}"/><text class="tk" x="${sx(t)}" y="${T + ih + 15}" text-anchor="middle">${esc(lb)}</text>`;
    const yt = P.ylog ? (() => { const a = []; for (let e = Math.floor(Y0); e <= Math.ceil(Y1); e++) if (e >= Y0 - 1e-9 && e <= Y1 + 1e-9) a.push(10 ** e); return a.length >= 2 ? a : [y0, y1]; })() : ticks(y0, y1, 5);
    for (const t of yt) g += `<line class="grid" x1="${L}" x2="${L + iw}" y1="${sy(t)}" y2="${sy(t)}"/><text class="tk" x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t, 3)}</text>`;
    g += `<rect class="frame" x="${L}" y="${T}" width="${iw}" height="${ih}"/>`;
  }
  // işaret çizgileri
  for (const m of P.marks || []) {
    if (m.x != null && m.x >= x0 && m.x <= x1) { const lb = esc(m.label || ""), right = sx(m.x) + 4 + (m.label || "").length * 7 > L + iw; g += `<line class="mark" x1="${sx(m.x)}" x2="${sx(m.x)}" y1="${T}" y2="${T + ih}"/><text class="mk" x="${right ? sx(m.x) - 4 : sx(m.x) + 4}" y="${T + 12}"${right ? ' text-anchor="end"' : ""}>${lb}</text>`; }
    if (m.y != null && (!P.ylog || m.y > 0) && m.y >= y0 && m.y <= y1) g += `<line class="mark" x1="${L}" x2="${L + iw}" y1="${sy(m.y)}" y2="${sy(m.y)}"/><text class="mk" x="${L + iw - 4}" y="${sy(m.y) - 4}" text-anchor="end">${esc(m.label || "")}</text>`;
  }
  // seriler
  let legend = "";
  all.forEach((s, k) => {
    const c = `s${(s.c ?? k) % 6}${s.dash ? " dash" : ""}`;
    const kind = s.kind || "line";
    const pts = s.x.map((x, i) => [x, s.y[i]]).filter(([x, y]) => fin(x) && fin(y) && (!P.ylog || y > 0));
    let body = "";
    if (kind === "scatter") body = pts.map(([x, y]) => `<circle class="${c}" cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="${s.r || 2.6}"/>`).join("");
    else if (kind === "bar") { const bw = Math.max(2, (iw / Math.max(pts.length, 1)) * (s.bw || 0.7)); const base = sy(P.ylog ? y0 : Math.max(y0, 0)); body = pts.map(([x, y]) => `<rect class="${c} bar" x="${(sx(x) - bw / 2 + (s.off || 0) * bw).toFixed(1)}" y="${Math.min(sy(y), base).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.abs(base - sy(y)).toFixed(1)}"/>`).join(""); }
    else {
      let d = "", pen = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (!fin(x) || !fin(y) || (P.ylog && y <= 0)) { pen = false; return; }
        const X = sx(x).toFixed(1), Y = sy(y).toFixed(1);
        if (kind === "step" && pen) d += `H${X}V${Y}`; else d += (pen ? "L" : "M") + X + "," + Y;
        pen = true;
      });
      body = `<path class="${c} ln" d="${d}"/>` + (s.dots ? pts.map(([x, y]) => `<circle class="${c}" cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="2.4"/>`).join("") : "");
    }
    g += `<g clip-path="url(#${clipId})">${body}</g>`;
    if (s.name) legend += `<span class="lg"><i class="${c}"></i>${esc(s.name)}</span>`;
  });
  if (P.xlabel) g += `<text class="ax" x="${L + iw / 2}" y="${H - 6}" text-anchor="middle">${esc(P.xlabel)}</text>`;
  if (P.ylabel) g += `<text class="ax" transform="translate(13 ${T + ih / 2}) rotate(-90)" text-anchor="middle">${esc(P.ylabel)}${P.ylog ? " (log)" : ""}</text>`;
  const aria = `${P.title || "grafik"}: ${P.xlabel || "x"} ekseninde ${fmt(x0)}–${fmt(x1)}, ${P.ylabel || "y"} ekseninde ${fmt(y0)}–${fmt(y1)}`;
  return `<figure class="lab-plot">${P.title ? `<figcaption>${esc(P.title)}</figcaption>` : ""}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria)}">${g}</svg>${legend ? `<div class="legend">${legend}</div>` : ""}</figure>`;
}

// Tohumlu rastgelelik (deneyler tekrarlanabilir olsun)
export function rng(seed) {
  let a = (seed >>> 0) || 1;
  const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let spare = null;
  u.normal = () => { if (spare != null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0); const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
  return u;
}

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
export const range = (n) => Array.from({ length: n }, (_, i) => i);
