// M4.6 — Ölçekli nokta çarpımı dikkati, hücre hücre: Q·Kᵀ skorları, ölçekleme, maskeler (nedensel, dolgu), satır softmax'ı ve Σ α·v.
// Elle tasarlanmış 4 boyutlu sorgu/anahtar/değer vektörleri (boyutlar: "kişi", "yer", "eylem", "zaman").
export const TOKENS = ["Kraliçe", "Birleşik", "Krallık'ı", "dün", "ziyaret", "etti", "<pad>"];
const DIMS = ["kişi", "yer", "eylem", "zaman"];
const esc = (t) => t.replace(/</g, "&lt;").replace(/>/g, "&gt;");
// anahtar: "ben neyim"; sorgu: "ne arıyorum"; değer: "bana bakana ne veririm"
export const K = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 2.2, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0], [0, 0, 1.5, 0.5], [0, 0, 0, 0]];
export const Q = [[0, 1.6, 0.8, 0], [0, 1.5, 0, 0], [1.4, 1, 0, 0], [0, 0, 1.5, 0], [1.4, 0.6, 0, 1], [0, 0, 1.2, 1], [0, 0, 0, 0]];
export const V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0.3, 0.3, 0.3, 0.3]];
export function attention(Qm, Km, Vm, { scale = 1 / Math.sqrt(4), causal = false, pad = null, temp = 1 } = {}) {
  const n = Qm.length, S = Qm.map((q, i) => Km.map((k, j) => {
    if ((causal && j > i) || (pad && pad[j])) return -Infinity;
    return (q.reduce((s, v, t) => s + v * k[t], 0) * scale) / temp;
  }));
  const A = S.map((row) => { const m = Math.max(...row.filter(Number.isFinite)); const e = row.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0)); const z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); });
  const Z = A.map((a) => Vm[0].map((_, t) => a.reduce((s, w, j) => s + w * Vm[j][t], 0)));
  return { S, A, Z, n };
}

export default {
  custom: true,
  keys: ["q", "causal", "pad", "sq", "temp"],
  selfTest(assert) {
    const R = attention(Q, K, V);
    for (const row of R.A) assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12, "satır toplamı 1");
    // nedensel maske: üst üçgen sıfır
    const C = attention(Q, K, V, { causal: true }); for (let i = 0; i < C.n; i++) for (let j = i + 1; j < C.n; j++) assert.equal(C.A[i][j], 0);
    // dolgu maskesi: pad sütunu sıfır
    const P = attention(Q, K, V, { pad: TOKENS.map((t) => t === "<pad>") }); assert.ok(P.A.every((r) => r[6] === 0));
    // sıcaklık → 0: argmax; sıcaklık büyük: düzgün dağılıma yaklaşır
    const hot = attention(Q, K, V, { temp: 1e-3 }); assert.ok(Math.abs(Math.max(...hot.A[0]) - 1) < 1e-6);
    const cold = attention(Q, K, V, { temp: 1e4 }); assert.ok(cold.A[0].every((v) => Math.abs(v - 1 / 7) < 1e-3));
    // "Kraliçe" sorgusu en çok yer/eylem anahtarlarına bakar
    assert.ok(R.A[0][2] > R.A[0][3]);
  },
  mount(el, ctx) {
    const st = { q: 0, causal: 0, pad: 1, sq: 1, temp: 1 };
    el.innerHTML = `<div class="lab-bar">
      <label>Maske <select data-k="causal"><option value="0">yok (kodlayıcı öz-dikkati)</option><option value="1">nedensel (kod çözücü: yalnız önceki tokenlar)</option></select></label>
      <label>Dolgu maskesi <select data-k="pad"><option value="1">açık (&lt;pad&gt; yok sayılır)</option><option value="0">kapalı</option></select></label>
      <label>Ölçek <select data-k="sq"><option value="1">÷ √d_k = ÷ 2</option><option value="0">ölçeksiz</option></select></label>
      <label>Sıcaklık τ <input type="number" data-k="temp" min="0.05" max="20" step="0.05" style="width:5em"></label></div>
      <p class="small muted">Satır = sorgu (soran token), sütun = anahtar (cevap veren token). Bir satıra tıkla.</p>
      <div class="ic-scroll"><table class="ic-t at-t"></table></div><div class="nx-read" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s), f = (v) => (Number.isFinite(v) ? ctx.fmt(Math.abs(v) < 5e-13 ? 0 : v, 3) : "−∞");
    function draw() {
      for (const k of ["causal", "pad", "sq"]) $(`[data-k=${k}]`).value = st[k]; if (document.activeElement !== $("[data-k=temp]")) $("[data-k=temp]").value = st.temp;
      const R = attention(Q, K, V, { scale: st.sq ? 0.5 : 1, causal: !!st.causal, pad: st.pad ? TOKENS.map((t) => t === "<pad>") : null, temp: st.temp });
      $(".at-t").innerHTML = `<thead><tr><th>α (satır toplamı 1)</th>${TOKENS.map((t) => `<th>${esc(t)}</th>`).join("")}</tr></thead><tbody>${R.A.map((row, i) => `<tr class="${i === st.q ? "at-sel" : ""}"><th><button type="button" class="ic-h${i === st.q ? " sel" : ""}" data-i="${i}">${esc(TOKENS[i])}</button></th>${row.map((a) => `<td style="background:color-mix(in srgb, var(--c-a) ${Math.round(a * 85)}%, transparent)">${a < 0.005 ? (a === 0 ? "0" : "≈0") : a.toFixed(2)}</td>`).join("")}</tr>`).join("")}</tbody>`;
      const i = st.q, row = R.S[i], best = R.A[i].indexOf(Math.max(...R.A[i]));
      $(".nx-read").innerHTML = `<p><b>${esc(TOKENS[i])}</b> sorgusu q = [${Q[i].join(", ")}] (${DIMS.map((d, t) => (Q[i][t] ? d : null)).filter(Boolean).join(" + ") || "boş"} arıyor).</p>
        <p class="small">Skorlar q·k${st.sq ? " / √4" : ""}${st.temp !== 1 ? ` / τ(${st.temp})` : ""}: ${row.map((s, j) => `${esc(TOKENS[j])} ${f(s)}`).join(" · ")}</p>
        <p>En çok baktığı: <b>${esc(TOKENS[best])}</b> (α = ${R.A[i][best].toFixed(3)}). Çıktı z = Σ α<sub>j</sub>·v<sub>j</sub> = [${R.Z[i].map((v) => v.toFixed(2)).join(", ")}] — boyutlar (${DIMS.join(", ")}): "${esc(TOKENS[i])}" tokenının yeni temsili, baktığı tokenların değerlerinin ağırlıklı ortalaması.</p>
        <p class="small">${st.causal ? "Nedensel maske: j > i skorları −∞ → softmax'tan sonra tam 0. Kod çözücü eğitimde geleceği göremez." : "Maske yok: her token tüm tokenlara (kendisi dahil) bakar."} ${st.pad ? "&lt;pad&gt; sütunu −∞ ile maskeli: dolgu kimsenin temsilini etkilemez." : "<b>Dolgu maskesi kapalı:</b> &lt;pad&gt; bir miktar ağırlık alıyor ve çıktıları sahte bir değerle kirletiyor."} τ küçüldükçe dağılım sertleşir (argmax'a yaklaşır), büyüdükçe düzleşir; √d_k ile bölmek τ = √d_k seçmektir.</p>`;
      for (const b of el.querySelectorAll(".ic-h")) b.addEventListener("click", () => { st.q = +b.dataset.i; draw(); });
    }
    for (const k of ["causal", "pad", "sq"]) $(`[data-k=${k}]`).addEventListener("change", (e) => { st[k] = +e.target.value; draw(); });
    $("[data-k=temp]").addEventListener("input", (e) => { const v = +e.target.value; if (v >= 0.05 && v <= 20) { st.temp = v; draw(); } });
    const set = (p) => { for (const k of Object.keys(st)) if (p[k] != null) st[k] = p[k]; draw(); };
    set(ctx.init);
    return { set };
  },
};
