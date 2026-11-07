// Her sayfada çalışan giriş noktası.
import { progress } from "./store.js";
import { onChange, checkLocal, captureRunnerLink, runnerConf, setRunnerConf, warmBrowser } from "./engines.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// --- tema
$(".theme-toggle")?.addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("ptl.theme", next); } catch {}
});

// --- mobil müfredat menüsü
const tog = $(".nav-toggle");
tog?.addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  tog.setAttribute("aria-expanded", String(open));
});

// --- sözlük terimleri: dokunmatikte dokununca aç/kapa
document.addEventListener("click", (e) => {
  const t = e.target.closest("dfn.term");
  $$("dfn.term.open").forEach((x) => x !== t && x.classList.remove("open"));
  if (t) t.classList.toggle("open");
});

// --- yan menüde modül ilerlemesi
function paintNav() {
  for (const a of $$("a.nav-mod[data-mod]")) {
    const n = progress.count(a.dataset.mod);
    const el = $(".nav-prog", a);
    if (el) el.innerHTML = Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("");
    a.classList.toggle("complete", n === 5);
  }
  for (const s of $$("[data-prog]")) { const n = progress.count(s.dataset.prog); s.textContent = n ? `${n}/5` : "hazır"; s.classList.toggle("complete", n === 5); }
  const sum = $("#prog-summary");
  if (sum) {
    const mods = window.PTL.modules;
    const done = mods.filter((m) => progress.count(m) === 5).length;
    const stages = mods.reduce((a, m) => a + progress.count(m), 0);
    sum.textContent = `${done}/${mods.length} hazır modül tamam · ${stages}/${mods.length * 5} aşama`;
  }
}
paintNav();
addEventListener("ptl:progress", paintNav);

// --- ilerleme dışa/içe aktarma (müfredat sayfası)
document.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "prog-export") {
    const blob = new Blob([JSON.stringify({ app: "pytorch-lab", v: 1, progress: progress.all() }, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "pytorch-lab-ilerleme.json" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  if (act === "prog-clear" && confirm("Tüm ilerleme işaretleri silinsin mi? (Yazdığın kodlar kalır.)")) progress.clear();
});
document.addEventListener("change", async (e) => {
  if (e.target.dataset?.act !== "prog-import") return;
  try {
    const j = JSON.parse(await e.target.files[0].text());
    if (j.app !== "pytorch-lab" || typeof j.progress !== "object") throw new Error();
    progress.replace(j.progress);
  } catch { alert("Bu dosya bir pytorch-lab ilerleme dosyası değil."); }
  e.target.value = "";
});

// --- başlıktaki motor göstergesi
const pill = $("#engine-pill");
if (pill) {
  pill.addEventListener("click", () => (location.href = "kurulum.html#motorlar"));
  onChange((s) => {
    const b = s.browser.status === "ready" ? "on" : s.browser.status === "loading" ? "busy" : s.browser.status === "error" ? "err" : "off";
    const l = s.local.status === "ok" ? "on" : s.local.status === "checking" ? "busy" : s.local.status === "auth" ? "err" : "off";
    pill.innerHTML = `<span class="dot ${b}"></span>Python <span class="dot ${l}"></span>PyTorch`;
    pill.title = `Tarayıcı: ${s.browser.text}\nYerel PyTorch: ${s.local.text}`;
  });
}

// --- kurulum sayfası: yerel motor bağlantı paneli
function runnerPanel() {
  const root = $("#runner-connect");
  if (!root) return;
  const captured = captureRunnerLink();
  const c = runnerConf();
  root.innerHTML = `
    <div class="rc-row"><label>Port <input id="rc-port" type="number" min="1" max="65535" value="${c.port}"></label>
    <label class="grow">Token <input id="rc-token" type="password" autocomplete="off" spellcheck="false" placeholder="motor başlarken terminalde yazar" value="${c.token || ""}"></label>
    <button class="btn primary" id="rc-save" type="button">Bağlan</button></div>
    <p class="rc-status" id="rc-status" role="status"></p>
    <div class="rc-row"><button class="btn" id="rc-warm" type="button">Tarayıcı Python'unu şimdi indir</button><span class="muted" id="rc-browser"></span></div>`;
  const st = $("#rc-status"), br = $("#rc-browser");
  onChange((s) => {
    const cls = s.local.status === "ok" ? "good" : s.local.status === "auth" ? "bad" : "";
    st.className = "rc-status " + cls;
    st.textContent = (s.local.status === "ok" ? "Bağlı: " : "Yerel motor: ") + s.local.text;
    br.textContent = s.browser.text;
  });
  $("#rc-save").addEventListener("click", () => { setRunnerConf({ port: Number($("#rc-port").value) || window.PTL.port, token: $("#rc-token").value.trim() }); checkLocal(); });
  $("#rc-warm").addEventListener("click", warmBrowser);
  if (captured) setTimeout(() => $("#motorlar")?.scrollIntoView(), 50);
}
runnerPanel();

// --- modül sayfası
if ($("#page-data")) import("./module.js").then((m) => m.initModule());
else checkLocal();
