// M0.2 — Aynı matris çarpımı, farklı döngü sırası: bellek erişim deseni hızı nasıl değiştirir?
// Tarayıcında gerçek zaman ölçümü yapar (sonuçlar makineye göre değişir; oranlar önemli).
import { plot, fmt } from "./plot.js";

export function matmul(order, A, B, C, n) {
  C.fill(0);
  if (order === "ijk") { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j]; C[i * n + j] = s; } }
  else if (order === "ikj") { for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const a = A[i * n + k], r = i * n, q = k * n; for (let j = 0; j < n; j++) C[r + j] += a * B[q + j]; } }
  else if (order === "jki") { for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) { const b = B[k * n + j]; for (let i = 0; i < n; i++) C[i * n + j] += A[i * n + k] * b; } }
  else if (order === "Bᵀ") { // B önceden transpoze: iç döngü iki satırı ardışık okur
    const T = new Float64Array(n * n); for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) T[j * n + k] = B[k * n + j];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; const r = i * n, q = j * n; for (let k = 0; k < n; k++) s += A[r + k] * T[q + k]; C[i * n + j] = s; }
  }
  return C;
}
const ORDERS = ["ijk", "ikj", "jki", "Bᵀ"];
const ACCESS = { ijk: "A satır boyunca (ardışık), B sütun boyunca (n adım atlayarak)", ikj: "B ve C satır boyunca (ardışık) — iç döngü vektörleşebilir", jki: "A ve C sütun boyunca (hep n adım atlayarak) — en kötü", "Bᵀ": "B'yi bir kez transpoze et, sonra iki satırın iç çarpımı (ardışık)" };

export default {
  custom: true,
  keys: ["n"],
  selfTest(assert) {
    const n = 7, A = Float64Array.from({ length: n * n }, (_, i) => Math.sin(i)), B = Float64Array.from({ length: n * n }, (_, i) => Math.cos(i * 0.7));
    const ref = Array.from(matmul("ijk", A, B, new Float64Array(n * n), n));
    for (const o of ORDERS) matmul(o, A, B, new Float64Array(n * n), n).forEach((v, i) => assert.ok(Math.abs(v - ref[i]) < 1e-12, `${o} aynı sonucu vermeli`));
  },
  mount(el, ctx) {
    let n = ctx.init.n || 256;
    el.innerHTML = `<div class="lab-bar"><label>Matris boyu n <select data-k="n">${[64, 128, 256, 384, 512].map((v) => `<option${v === n ? " selected" : ""}>${v}</option>`).join("")}</select></label>
      <button class="btn primary small" type="button" data-k="run">Ölç (dört sıra, en iyi 3 deneme)</button><span class="muted small" data-k="st" role="status"></span></div><div class="lab-out"></div><div class="lab-stats"></div><div class="lab-note prose"></div>`;
    const out = el.querySelector(".lab-out"), stats = el.querySelector(".lab-stats"), note = el.querySelector(".lab-note"), stt = el.querySelector("[data-k=st]");
    el.querySelector("[data-k=n]").addEventListener("change", (e) => { n = Number(e.target.value); });
    el.querySelector("[data-k=run]").addEventListener("click", async () => {
      stt.textContent = "Ölçülüyor…";
      await new Promise((r) => setTimeout(r, 30));
      const A = Float64Array.from({ length: n * n }, () => Math.random()), B = Float64Array.from({ length: n * n }, () => Math.random()), C = new Float64Array(n * n);
      const res = {};
      for (const o of ORDERS) {
        let best = Infinity;
        for (let r = 0; r < 3; r++) { const t = performance.now(); matmul(o, A, B, C, n); best = Math.min(best, performance.now() - t); await new Promise((q) => setTimeout(q, 0)); }
        res[o] = best;
      }
      const flops = 2 * n ** 3;
      out.innerHTML = `<div class="lab-plots">${plot({ title: `n = ${n}: süre (ms, küçük daha iyi)`, xlabel: "döngü sırası: 1 ijk · 2 ikj · 3 jki · 4 Bᵀ", ylabel: "ms", yzero: true, series: [{ name: "süre", x: [1, 2, 3, 4], y: ORDERS.map((o) => res[o]), kind: "bar" }] })}</div>`;
      stats.innerHTML = `<table class="lab-tbl"><thead><tr><th>sıra</th><th>iç döngünün bellek erişimi</th><th>süre</th><th>GFLOP/s</th><th>en hızlıya göre</th></tr></thead><tbody>${ORDERS.map((o) => `<tr><td>${o}</td><td>${ACCESS[o]}</td><td>${fmt(res[o], 3)} ms</td><td>${fmt(flops / res[o] / 1e6, 3)}</td><td>${fmt(res[o] / Math.min(...Object.values(res)), 3)}×</td></tr>`).join("")}</tbody></table>`;
      note.innerHTML = `<p>Dört sıra da <b>aynı</b> ${fmt(flops, 3)} kayan nokta işlemini yapar; fark yalnız belleğe erişim sırasıdır. Bellek önbellek satırları (64 bayt = 8 float64) hâlinde okunur: ardışık erişim bir satırı tamamen kullanır, n adım atlayarak erişim her eleman için yeni bir satır getirir. n büyüdükçe matrisler önbelleğe sığmaz ve fark büyür. NumPy/PyTorch'un BLAS kütüphaneleri ayrıca <b>bloklama</b> (önbelleğe sığan alt matrisler), SIMD ve çok çekirdek kullanır: aynı çarpım yüzlerce kat daha hızlıdır (e aşamasında ölçeceksin).</p>`;
      stt.textContent = "Bitti — tekrar ölçebilir ya da n'yi değiştirebilirsin.";
    });
    return { set(p) { if (p.n) { n = p.n; el.querySelector("[data-k=n]").value = n; } } };
  },
};
