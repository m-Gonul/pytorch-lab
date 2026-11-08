// Deney alanı çalışma zamanı: kaydırıcı + kesin değer, sıfırla, hazır senaryolar, A/B karşılaştırma, tohum.
// Her deney site/js/labs/<ad>.js içinde { params, seed, validate, run } dışa aktarır; hesap gerçektir.
import { plot, fmt, rng } from "./plot.js";

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export async function mountAll(cfgs) {
  for (const sec of document.querySelectorAll("section.lab[data-lab]")) {
    const cfg = cfgs[sec.dataset.labId];
    try {
      const spec = (await import(`./${sec.dataset.lab}.js`)).default;
      mount(sec, spec, cfg || {});
    } catch (e) {
      sec.querySelector(".lab-ui").innerHTML = `<p class="cb-error">Deney yüklenemedi: ${esc(e.message)}</p>`;
    }
  }
}

export function defaults(spec, init = {}) {
  const p = Object.fromEntries(spec.params.map((q) => [q.id, q.value]));
  return { ...p, ...init };
}

// Ortak: bir parametre takımıyla çalıştır (doğrulama + tohumlu rastgelelik)
export function evaluate(spec, p, seed = 1) {
  const issues = (spec.validate ? spec.validate(p) : []) || [];
  for (const q of spec.params) {
    const v = p[q.id];
    if (typeof v !== "number" || Number.isNaN(v)) issues.push({ level: "error", msg: `${q.label}: sayı değil.` });
    else if (q.options) { if (!(v >= 0 && v < q.options.length)) issues.push({ level: "error", msg: `${q.label}: geçersiz seçim.` }); }
    else if (v < (q.hardMin ?? q.min) || v > (q.hardMax ?? q.max)) issues.push({ level: "error", msg: `${q.label} = ${fmt(v)} geçersiz: izin verilen aralık ${fmt(q.hardMin ?? q.min)} … ${fmt(q.hardMax ?? q.max)}.${q.why ? " " + q.why : ""}` });
  }
  if (issues.some((x) => x.level === "error")) return { issues };
  return { issues, out: spec.run(p, rng(seed)) };
}

function mount(sec, spec, cfg) {
  const ui = sec.querySelector(".lab-ui");
  const base = defaults(spec, cfg.init);
  const st = { A: { ...base }, B: { ...base }, cmp: false, seed: 1 };
  const canCmp = spec.compare !== false;
  ui.innerHTML = `
    <div class="lab-bar">
      ${canCmp ? `<label class="lab-cmp"><input type="checkbox" data-k="cmp"> İki ayarı karşılaştır (A / B)</label>` : ""}
      ${spec.seed ? `<label class="lab-seed">Tohum <input type="number" min="1" max="99999" step="1" value="1" data-k="seed"></label><button class="btn small" type="button" data-k="reseed" title="Aynı ayarlarla başka bir rastgele örneklem">Yeni tohum</button>` : ""}
      <button class="btn ghost small" type="button" data-k="reset">Başlangıca dön</button>
    </div>
    <div class="lab-ctrls"><div class="lab-col" data-set="A"></div><div class="lab-col" data-set="B" hidden></div></div>
    <div class="lab-issues" role="status"></div>
    <div class="lab-out"></div>
    <div class="lab-stats"></div>
    <div class="lab-note prose"></div>`;
  const cols = { A: ui.querySelector('[data-set="A"]'), B: ui.querySelector('[data-set="B"]') };
  for (const S of ["A", "B"]) {
    cols[S].innerHTML = `<p class="lab-set">${S === "A" ? "Ayar A" : "Ayar B"}</p>` + spec.params.map((q) => {
      const lg = q.log;
      const toS = (v) => (lg ? Math.log10(v) : v);
      if (q.options) return `<div class="lab-p sel" data-p="${q.id}"><label for="${sec.id}-${S}-${q.id}">${q.label}</label><select id="${sec.id}-${S}-${q.id}" data-s="${q.id}">${q.options.map((o, i) => `<option value="${i}"${i === st[S][q.id] ? " selected" : ""}>${o}</option>`).join("")}</select>${q.help ? `<p class="lab-help">${q.help}</p>` : ""}</div>`;
      return `<div class="lab-p" data-p="${q.id}">
        <label for="${sec.id}-${S}-${q.id}">${q.label}${q.unit ? ` <span class="muted">(${q.unit})</span>` : ""}</label>
        <input type="range" id="${sec.id}-${S}-${q.id}" min="${toS(q.min)}" max="${toS(q.max)}" step="${lg ? (toS(q.max) - toS(q.min)) / 200 : q.step}" value="${toS(st[S][q.id])}" data-r="${q.id}">
        <input type="number" aria-label="${esc(q.label)} kesin değer" step="any" value="${st[S][q.id]}" data-n="${q.id}">
        ${q.help ? `<p class="lab-help">${q.help}</p>` : ""}</div>`;
    }).join("");
    cols[S].addEventListener("input", (e) => {
      const r = e.target.dataset.r, n = e.target.dataset.n, sl = e.target.dataset.s;
      if (sl) st[S][sl] = Number(e.target.value);
      if (r) {
        const q = spec.params.find((x) => x.id === r);
        let v = Number(e.target.value); if (q.log) v = 10 ** v;
        v = q.int ? Math.round(v) : Number(v.toPrecision(4));
        st[S][r] = v; cols[S].querySelector(`[data-n="${r}"]`).value = v;
      }
      if (n) {
        const q = spec.params.find((x) => x.id === n);
        const v = e.target.value === "" ? NaN : Number(e.target.value);
        st[S][n] = q.int && Number.isFinite(v) ? Math.round(v) : v;
        if (Number.isFinite(v)) cols[S].querySelector(`[data-r="${n}"]`).value = q.log ? Math.log10(Math.max(v, 1e-300)) : v;
      }
      paintActive();
      schedule();
    });
  }
  const sync = () => {
    for (const S of ["A", "B"]) for (const q of spec.params) {
      if (q.options) { cols[S].querySelector(`[data-s="${q.id}"]`).value = st[S][q.id]; continue; }
      cols[S].querySelector(`[data-n="${q.id}"]`).value = st[S][q.id];
      cols[S].querySelector(`[data-r="${q.id}"]`).value = q.log ? Math.log10(st[S][q.id]) : st[S][q.id];
    }
    cols.B.hidden = !st.cmp;
    paintActive();
    const c = ui.querySelector('[data-k="cmp"]'); if (c) c.checked = st.cmp;
    const sd = ui.querySelector('[data-k="seed"]'); if (sd) sd.value = st.seed;
  };
  ui.addEventListener("change", (e) => {
    if (e.target.dataset.k === "cmp") { st.cmp = e.target.checked; if (st.cmp) st.B = { ...st.A }; sync(); schedule(); }
    if (e.target.dataset.k === "seed") { st.seed = Math.max(1, Math.round(Number(e.target.value) || 1)); schedule(); }
  });
  ui.addEventListener("click", (e) => {
    const k = e.target.dataset.k;
    if (k === "reset") { st.A = { ...base }; st.B = { ...base }; st.cmp = false; st.seed = 1; sync(); schedule(); const n = sec.querySelector(".lab-preset-note"); if (n) n.textContent = ""; }
    if (k === "reseed") { st.seed = 1 + Math.floor(Math.random() * 99998); sync(); schedule(); }
  });
  sec.querySelector(".lab-presets")?.addEventListener("click", (e) => {
    const i = e.target.closest("[data-preset]")?.dataset.preset;
    if (i == null) return;
    const pr = cfg.presets[i];
    const { B, seed, ...A } = pr.params || {};
    st.A = { ...base, ...A };
    if (B) { st.cmp = true; st.B = { ...base, ...A, ...B }; } else st.cmp = false;
    if (seed) st.seed = seed;
    sync(); schedule();
    const n = sec.querySelector(".lab-preset-note"); if (n) n.innerHTML = pr.note || "";
  });
  for (const b of sec.querySelectorAll(".lab-predict .q-opt")) b.addEventListener("click", () => { b.classList.add(b.dataset.ok === "1" ? "ok" : "bad"); b.nextElementSibling.hidden = false; });

  // Bir kontrol yalnız belirli bir moda anlamlıysa (q.on) öteki modlarda pasif ve açıklamalı
  function paintActive() {
    for (const S of ["A", "B"]) for (const q of spec.params) {
      if (!q.on) continue;
      const ok = Object.entries(q.on).every(([k, vals]) => vals.includes(st[S][k]));
      const box = cols[S].querySelector(`[data-p="${q.id}"]`);
      box.classList.toggle("off", !ok);
      box.querySelectorAll("input,select").forEach((i) => (i.disabled = !ok));
      box.title = ok ? "" : "Bu ayar seçili modda kullanılmıyor.";
    }
    const sd = ui.querySelector('[data-k="seed"]');
    if (sd && spec.seedOn) { const ok = Object.entries(spec.seedOn).every(([k, v]) => st.A[k] >= v || (st.cmp && st.B[k] >= v)); sd.disabled = !ok; ui.querySelector('[data-k="reseed"]').disabled = !ok; sd.title = ok ? "" : "Bu ayarda rastgelelik yok (gürültü 0)."; }
  }
  let pending = false;
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
  function render() {
    const A = evaluate(spec, st.A, st.seed), B = st.cmp ? evaluate(spec, st.B, st.seed) : null;
    const iss = [...A.issues.map((x) => ({ ...x, set: st.cmp ? "A" : "" })), ...(B ? B.issues.map((x) => ({ ...x, set: "B" })) : [])];
    ui.querySelector(".lab-issues").innerHTML = iss.map((x) => `<p class="lab-iss ${x.level}">${x.set ? `<b>${x.set}:</b> ` : ""}${x.msg}</p>`).join("");
    const out = ui.querySelector(".lab-out"), stats = ui.querySelector(".lab-stats"), note = ui.querySelector(".lab-note");
    if (!A.out || (B && !B.out)) { out.classList.add("stale"); return; }
    out.classList.remove("stale");
    const plots = A.out.plots.map((P, i) => {
      if (!B) return plot(P);
      const Q = B.out.plots[i];
      if (P.img || P.side) return `<div class="lab-pair">${plot({ ...P, title: "A · " + (P.title || "") }, { w: 320, h: 260 })}${plot({ ...Q, title: "B · " + (Q.title || "") }, { w: 320, h: 260 })}</div>`;
      const ser = [...P.series.map((s) => ({ ...s, name: s.name ? "A: " + s.name : s.name })), ...Q.series.map((s, k) => ({ ...s, c: s.c ?? k, dash: true, name: s.name ? "B: " + s.name : s.name }))];
      return plot({ ...P, series: ser, xlim: P.xlim, ylim: P.ylim && Q.ylim ? [Math.min(P.ylim[0], Q.ylim[0]), Math.max(P.ylim[1], Q.ylim[1])] : P.ylim });
    });
    out.innerHTML = `<div class="lab-plots">${plots.join("")}</div>`;
    const sa = A.out.stats || [], sb = B?.out.stats || [];
    stats.innerHTML = sa.length ? `<table class="lab-tbl"><thead><tr><th>Ölçüm</th><th>${B ? "A" : "Değer"}</th>${B ? "<th>B</th>" : ""}</tr></thead><tbody>${sa.map((s, i) => `<tr><td>${s.label}</td><td>${typeof s.value === "number" ? fmt(s.value) : s.value}${s.unit ? " " + s.unit : ""}</td>${B ? `<td>${typeof sb[i]?.value === "number" ? fmt(sb[i].value) : sb[i]?.value ?? "—"}${sb[i]?.unit ? " " + sb[i].unit : ""}</td>` : ""}</tr>`).join("")}</tbody></table>` : "";
    note.innerHTML = B ? `${A.out.note ? `<p><b>A:</b> ${A.out.note}</p>` : ""}${B.out.note ? `<p><b>B:</b> ${B.out.note}</p>` : ""}` : A.out.note ? `<p>${A.out.note}</p>` : "";
  }
  sync();
  render();
}
