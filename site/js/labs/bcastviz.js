// M3.3 — Yayın (broadcasting) ve geri geçişi: iki shape gir; hizalama, sonuç shape'i ve geri geçişte her girdi için hangi eksenler üzerinden
// toplanacağı (unbroadcast). Küçük bir sayısal örnekle doğrulama: G = 1'ler için dA = yayılan kopya sayısı.
export function bshape(a, b) {
  const n = Math.max(a.length, b.length), A = [...Array(n - a.length).fill(1), ...a], B = [...Array(n - b.length).fill(1), ...b], out = [];
  for (let i = 0; i < n; i++) { if (A[i] !== B[i] && A[i] !== 1 && B[i] !== 1) throw new Error(`Eksen ${i - n} (sağdan ${n - i}.): ${A[i]} ile ${B[i]} uyuşmuyor — ikisi eşit ya da biri 1 olmalı.`); out.push(Math.max(A[i], B[i])); }
  return { A, B, out };
}
// geri: hangi eksenler toplanır? önce baştaki fazla eksenler (keepdim yok), sonra boyu 1 olup yayılanlar (keepdim)
export function unbAxes(inp, out) { const lead = out.length - inp.length, sumLead = Array.from({ length: lead }, (_, i) => i), keep = []; inp.forEach((d, i) => { if (d === 1 && out[lead + i] !== 1) keep.push(lead + i); }); return { sumLead, keep }; }
export function parseShape(s) { const t = s.replace(/[()\[\]\s]/g, ""); if (t === "") return []; const v = t.split(",").filter((x) => x !== "").map(Number); if (v.some((x) => !Number.isInteger(x) || x < 1 || x > 64)) throw new Error(`'${s}' geçerli bir shape değil: 1–64 arası tam sayılar, ör. (3, 1).`); return v; }
const prod = (a) => a.reduce((x, y) => x * y, 1);

export default {
  custom: true,
  keys: ["a", "b"],
  selfTest(assert) {
    assert.deepEqual(bshape([3, 1], [4]).out, [3, 4]);
    assert.deepEqual(bshape([8, 1, 5], [7, 1]).out, [8, 7, 5]);
    assert.throws(() => bshape([64, 10], [64]));
    assert.deepEqual(unbAxes([10], [64, 10]), { sumLead: [0], keep: [] }, "bias: batch ekseni üzerinden topla");
    assert.deepEqual(unbAxes([3, 1], [3, 4]), { sumLead: [], keep: [1] });
    assert.deepEqual(unbAxes([7, 1], [8, 7, 5]), { sumLead: [0], keep: [2] });
    // G = 1'ler: her girdi elemanının gradyanı = kaç kez kopyalandığı = prod(out)/prod(in)
    assert.equal(prod([8, 7, 5]) / prod([7, 1]), 40);
  },
  mount(el, ctx) {
    el.innerHTML = `<div class="lab-bar"><label>A.shape <input type="text" data-k="a" class="bc-in" spellcheck="false"></label><label>B.shape <input type="text" data-k="b" class="bc-in" spellcheck="false"></label><button class="btn primary small" type="button" data-k="go">Hesapla</button></div><div class="bc-out" aria-live="polite"></div>`;
    const $ = (s) => el.querySelector(s);
    const run = () => {
      try {
        const a = parseShape($("[data-k=a]").value), b = parseShape($("[data-k=b]").value), R = bshape(a, b), n = R.out.length;
        const row = (lab, arr, orig) => `<tr><th>${lab}</th>${arr.map((d, i) => { const pad = i < n - orig.length; const exp = d === 1 && R.out[i] !== 1; return `<td class="${pad ? "bc-pad" : ""}${exp ? " bc-exp" : ""}">${pad ? `(1)` : d}${exp ? " → " + R.out[i] : ""}</td>`; }).join("")}</tr>`;
        const ua = unbAxes(a, R.out), ub = unbAxes(b, R.out);
        const desc = (u, nm, s) => { const parts = []; if (u.sumLead.length) parts.push(`baştaki ${u.sumLead.length} eksen üzerinden topla (<code>.sum(dim=${u.sumLead.length === 1 ? u.sumLead[0] : `(${u.sumLead.join(", ")})`})</code>)`); if (u.keep.length) parts.push(`girdide boyu 1 olan eksen(ler) üzerinden — girdinin ${u.keep.map((k) => k - u.sumLead.length).join(", ")} numaralı ekseni (0'dan sayarak) — <code>keepdim=True</code> ile topla`); return `<li><b>d${nm}</b> (shape (${s.join(", ")})): ${parts.length ? parts.join(", sonra ") : "yayın yok — G olduğu gibi"}. Her elemanı ${prod(R.out) / Math.max(prod(s), 1)} kez kopyalandı; G = 1'lerken d${nm}'nin her elemanı ${prod(R.out) / Math.max(prod(s), 1)}.</li>`; };
        $(".bc-out").innerHTML = `<table class="lab-tbl bc-tbl"><thead><tr><th></th>${R.out.map((_, i) => `<th>eksen ${i - n}</th>`).join("")}</tr></thead><tbody>${row("A", R.A, a)}${row("B", R.B, b)}<tr class="bc-res"><th>A ⊕ B</th>${R.out.map((d) => `<td><b>${d}</b></td>`).join("")}</tr></tbody></table>
          <p class="small">Sağdan hizalandı; parantezli 1'ler eksik eksenlere eklenen boyutlar, oklar yayılan eksenler (bellekte stride 0, M0.5). Sonuç shape'i: <b>(${R.out.join(", ")})</b>, ${prod(R.out)} eleman.</p>
          <p><b>Geri geçiş</b> — gelen gradyan G, sonuçla aynı shape'te (${R.out.join(", ")}):</p><ul class="small">${desc(ua, "A", a)}${desc(ub, "B", b)}</ul>
          <p class="small muted">Neden toplam? Yayılan bir eleman sonuçta birçok yerde kullanıldı; çok değişkenli zincir kuralı (M0.3) her kullanımın katkısını toplar. Bias gradyanının batch üzerinden toplam olması bunun en yaygın örneğidir.</p>`;
      } catch (e) { $(".bc-out").innerHTML = `<p class="lab-iss error">${e.message}</p>`; }
    };
    const set = (p) => { $("[data-k=a]").value = p.a ?? "(64, 10)"; $("[data-k=b]").value = p.b ?? "(10,)"; run(); };
    $("[data-k=go]").addEventListener("click", run); for (const i of el.querySelectorAll("input")) i.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
    set(ctx.init);
    return { set };
  },
};
