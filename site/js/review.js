// Karışık tekrar: tüm modüllerin sınama soruları üzerinde Leitner kutuları (aralıklı tekrar).
import { progress, kv } from "./store.js";

const DAY = 864e5;
const GAP = [0, 0, 1, 3, 7, 16]; // kutu → gün (kutu 1: aynı oturumda tekrar)

export async function initReview() {
  const card = document.getElementById("rv-card"), stat = document.getElementById("rv-stat"), scopeSel = document.getElementById("rv-scope");
  let all = [];
  try { all = await (await fetch("review.json")).json(); } catch { card.innerHTML = "<p>Soru listesi yüklenemedi.</p>"; return; }
  scopeSel.value = kv.get("rv.scope", "touched");
  const touched = (m) => progress.count(m) > 0 || Object.keys(kv.get(`quiz.${m}`, {})).length > 0 || Object.values(kv.get(`self.${m}`, {})).some(Boolean);
  let queue = [], cur = null, seen = 0, right = 0;

  function build() {
    const st = kv.get("rv", {}), now = Date.now();
    const pool = all.filter((q) => scopeSel.value === "all" || touched(q.mod));
    const due = pool.filter((q) => st[q.id] && st[q.id].due <= now).sort((a, b) => st[a.id].due - st[b.id].due);
    const fresh = pool.filter((q) => !st[q.id]);
    // yeni soruları modüller arasında karıştır (aynı modülden art arda gelmesin)
    for (let i = fresh.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [fresh[i], fresh[j]] = [fresh[j], fresh[i]]; }
    queue = [...due, ...fresh.slice(0, Math.max(0, 15 - due.length))];
    const waiting = pool.filter((q) => st[q.id] && st[q.id].due > now).length;
    stat.textContent = `${pool.length} soru kapsamda · ${due.length} tekrar zamanı gelmiş · ${waiting} bekliyor`;
    if (!pool.length) { card.innerHTML = `<p>Henüz çalıştığın modül yok. Bir modülde bir aşamayı geç ya da sınamayı yanıtla, ya da kapsamı “Tüm modüller” yap.</p>`; return; }
    next();
  }

  function next() {
    cur = queue.shift();
    if (!cur) { card.innerHTML = `<p><b>Bu oturum bitti.</b> ${seen} soru, ${right} doğru. Sıradaki tekrarlar zamanı gelince burada görünür.</p><button class="btn" type="button" id="rv-more">Yeni sorularla devam et</button>`; document.getElementById("rv-more").onclick = build; return; }
    const opts = cur.o.map((o, i) => ({ ...o, i })).sort(() => Math.random() - 0.5);
    const box = kv.get("rv", {})[cur.id]?.b || 0;
    card.innerHTML = `<p class="rv-meta"><a href="${cur.mod}.html#sinama">${cur.code}</a> · kutu ${box || "yeni"}</p><div class="q-text">${cur.q}</div><ul class="q-opts">${opts.map((o) => `<li><button type="button" class="q-opt" data-ok="${o.ok ? 1 : 0}">${o.t}</button><div class="q-why" hidden>${o.why}</div></li>`).join("")}</ul><p class="rv-after" hidden></p>`;
    card.querySelector(".q-opts").addEventListener("click", (e) => {
      const b = e.target.closest(".q-opt");
      if (!b || card.dataset.done === "1") return;
      card.dataset.done = "1";
      const ok = b.dataset.ok === "1";
      b.classList.add(ok ? "ok" : "bad"); b.nextElementSibling.hidden = false;
      const good = card.querySelector('.q-opt[data-ok="1"]'); good.classList.add("ok"); good.nextElementSibling.hidden = false;
      const st = kv.get("rv", {}), s = st[cur.id] || { b: 0 };
      s.b = ok ? Math.min(5, s.b + 1) : 1;
      s.due = Date.now() + GAP[s.b] * DAY;
      st[cur.id] = s; kv.set("rv", st);
      seen++; if (ok) right++;
      if (!ok) queue.splice(Math.min(3, queue.length), 0, cur); // yanlışı birkaç soru sonra yeniden sor
      const after = card.querySelector(".rv-after");
      after.hidden = false;
      after.innerHTML = `${ok ? "Doğru — kutu " + s.b + `, ${GAP[s.b]} gün sonra yeniden.` : `Yanlış — bu oturumda yeniden gelecek. İlgili bölüm: <a href="${cur.mod}.html#sinama">${cur.code}</a>.`} <button class="btn primary small" type="button" id="rv-next">Sonraki</button>`;
      const nb = document.getElementById("rv-next"); nb.focus(); nb.onclick = () => { card.dataset.done = "0"; next(); };
    });
  }

  scopeSel.addEventListener("change", () => { kv.set("rv.scope", scopeSel.value); build(); });
  document.getElementById("rv-reset").addEventListener("click", () => { if (confirm("Tekrar kutuları sıfırlansın mı?")) { kv.del("rv"); build(); } });
  build();
}
