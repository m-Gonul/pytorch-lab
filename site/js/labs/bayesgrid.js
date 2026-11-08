// M0.4 — Bayes kuralı doğal frekanslarla: 1000 kişilik bir topluluk, bir hastalık ve kusurlu bir test.
// P(hasta | pozitif) = duyarlılık·önsel / [duyarlılık·önsel + (1−özgüllük)(1−önsel)]
export function posterior(prior, sens, spec) { const tp = sens * prior, fp = (1 - spec) * (1 - prior); return tp + fp > 0 ? tp / (tp + fp) : 0; }
export function counts(N, prior, sens, spec) {
  const sick = Math.round(N * prior), tp = Math.round(sick * sens), fn = sick - tp, healthy = N - sick, fp = Math.round(healthy * (1 - spec)), tn = healthy - fp;
  return { sick, healthy, tp, fn, fp, tn };
}
const pct = (v) => (v * 100 >= 10 ? (v * 100).toFixed(1) : (v * 100).toPrecision(2)) + " %";

export default {
  custom: true,
  keys: ["prev", "sens", "spec", "second"],
  selfTest(assert) {
    assert.ok(Math.abs(posterior(0.01, 0.9, 0.95) - 0.009 / (0.009 + 0.0495)) < 1e-12);
    const c = counts(1000, 0.01, 0.9, 0.95);
    assert.equal(c.sick + c.healthy, 1000); assert.equal(c.tp, 9); assert.equal(c.fp, 50);
    // ikinci pozitif test: sonsal yeni önsel olur
    const p1 = posterior(0.01, 0.9, 0.95), p2 = posterior(p1, 0.9, 0.95);
    const joint = (0.01 * 0.81) / (0.01 * 0.81 + 0.99 * 0.0025); assert.ok(Math.abs(p2 - joint) < 1e-12, "ardışık güncelleme = birlikte güncelleme (koşullu bağımsız testler)");
  },
  mount(el, ctx) {
    const st = { prev: 0.01, sens: 0.9, spec: 0.95, second: false };
    const set = (p) => { for (const k of ["prev", "sens", "spec"]) if (p[k] != null) st[k] = p[k]; if (p.second != null) st.second = !!p.second; };
    set(ctx.init);
    const sl = (id, label, min, max, step, log) => `<div class="lab-p"><label for="bg-${id}">${label} <output data-o="${id}"></output></label><div class="bg-row"><input id="bg-${id}" type="range" data-k="${id}" min="${min}" max="${max}" step="${step}"${log ? ' data-log="1"' : ""}><input type="number" class="bg-num" data-n="${id}" min="0.1" max="100" step="0.1" aria-label="${label}, yüzde olarak kesin değer"> %</div></div>`;
    el.innerHTML = `<div class="lab-ctrls bg-ctrls">${sl("prev", "Hastalığın yaygınlığı (önsel) P(H)", -3, -0.3, 0.01, true)}${sl("sens", "Duyarlılık P(+ | H)", 0.5, 1, 0.005)}${sl("spec", "Özgüllük P(− | sağlıklı)", 0.5, 1, 0.001)}
      <label class="bg-sec"><input type="checkbox" data-k="second"> Pozitif çıkanlara aynı testi bağımsız olarak <b>bir kez daha</b> uygula</label></div>
      <div class="bg-wrap"><svg class="bg" viewBox="0 0 500 200" role="img" aria-label="1000 kişilik ızgara: renkler test sonucunu ve gerçek durumu gösterir"></svg>
      <div class="bg-legend small"><span><i class="bg-tp"></i>hasta, test + (doğru pozitif)</span><span><i class="bg-fn"></i>hasta, test − (yanlış negatif)</span><span><i class="bg-fp"></i>sağlıklı, test + (yanlış pozitif)</span><span><i class="bg-tn"></i>sağlıklı, test −</span></div></div>
      <div class="bg-read" aria-live="polite"></div>`;
    const svg = el.querySelector("svg.bg"), read = el.querySelector(".bg-read");
    function draw() {
      const N = 1000, c = counts(N, st.prev, st.sens, st.spec);
      // sıralama: önce hasta+ , hasta−, sağlıklı+, sağlıklı−
      const cls = [...Array(c.tp).fill("bg-tp"), ...Array(c.fn).fill("bg-fn"), ...Array(c.fp).fill("bg-fp"), ...Array(c.tn).fill("bg-tn")];
      let g = "";
      cls.forEach((k, i) => { const col = Math.floor(i / 20), row = i % 20; g += `<rect x="${col * 10 + 1}" y="${row * 10 + 1}" width="8" height="8" rx="1.5" class="${k}"/>`; });
      svg.innerHTML = g;
      const post = posterior(st.prev, st.sens, st.spec), post2 = posterior(post, st.sens, st.spec);
      const pPos = st.sens * st.prev + (1 - st.spec) * (1 - st.prev);
      for (const k of ["prev", "sens", "spec"]) { const inp = el.querySelector(`[data-k=${k}]`); inp.value = inp.dataset.log ? Math.log10(st[k]) : st[k]; el.querySelector(`[data-o=${k}]`).textContent = pct(st[k]); const nb = el.querySelector(`[data-n=${k}]`); if (document.activeElement !== nb) nb.value = +(st[k] * 100).toPrecision(4); }
      el.querySelector("[data-k=second]").checked = st.second;
      read.innerHTML = `<table class="lab-tbl"><tbody>
        <tr><td>Pozitif çıkan</td><td>${c.tp} hasta + ${c.fp} sağlıklı = <b>${c.tp + c.fp}</b> kişi</td></tr>
        <tr><td>Sayarak: P(H | +)</td><td>${c.tp} / ${c.tp + c.fp} = <b>${c.tp + c.fp ? pct(c.tp / (c.tp + c.fp)) : "—"}</b></td></tr>
        <tr><td>Bayes formülüyle</td><td>P(+|H)·P(H) / P(+) = ${st.sens.toFixed(3)} · ${st.prev.toPrecision(3)} / ${pPos.toPrecision(3)} = <b>${pct(post)}</b></td></tr>
        <tr><td>Negatif çıkanın hasta olma olasılığı P(H | −)</td><td>${pct(((1 - st.sens) * st.prev) / (1 - pPos))}</td></tr>
        ${st.second ? `<tr><td>İkinci test de pozitifse</td><td>önsel artık ${pct(post)} → P(H | +, +) = <b>${pct(post2)}</b></td></tr>` : ""}</tbody></table>
        <p class="small">Izgaradaki her kare 1000 kişiden biridir (sayılar yuvarlanmıştır). Pozitif sonucun anlamı testin kalitesi kadar <b>önsele</b> bağlıdır: hastalık nadirse, sağlıklıların küçük bir yüzdesi bile hastaların tamamından daha çok sayıda yanlış pozitif üretir (taban oranı yanılgısı).</p>`;
    }
    for (const k of ["prev", "sens", "spec"]) el.querySelector(`[data-k=${k}]`).addEventListener("input", (e) => { st[k] = e.target.dataset.log ? 10 ** Number(e.target.value) : Number(e.target.value); draw(); });
    for (const k of ["prev", "sens", "spec"]) el.querySelector(`[data-n=${k}]`).addEventListener("change", (e) => {
      const v = Number(e.target.value) / 100, lo = k === "prev" ? 0.001 : 0.5, hi = k === "prev" ? 0.5 : 1;
      if (!(v >= lo && v <= hi)) { e.target.setAttribute("aria-invalid", "true"); read.insertAdjacentHTML("afterbegin", `<p class="lab-iss error">Geçersiz değer: ${k === "prev" ? "yaygınlık %0.1 – %50" : "duyarlılık ve özgüllük %50 – %100"} aralığında olmalı (bu şekil o aralıklar için kuruldu; %50'nin altındaki bir test, sonucu tersine okunursa daha iyidir).</p>`); return; }
      e.target.removeAttribute("aria-invalid"); st[k] = v; draw();
    });
    el.querySelector("[data-k=second]").addEventListener("change", (e) => { st.second = e.target.checked; draw(); });
    draw();
    return { set(p) { set(p); draw(); } };
  },
};
