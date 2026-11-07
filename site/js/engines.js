// Kod çalıştırma motorları: tarayıcı (Pyodide) ve yerel PyTorch runner'ı.
// Her ikisi de aynı labcheck.run_stage sonucunu döndürür.
const cfg = window.PTL || {};
const listeners = new Set();
export const state = {
  browser: { status: "idle", text: "henüz yüklenmedi", python: null },  // idle | loading | ready | error
  local: { status: "unknown", text: "bağlı değil", info: null },         // unknown | checking | ok | off | auth
};
const emit = () => listeners.forEach((f) => f(state));
export const onChange = (f) => { listeners.add(f); f(state); return () => listeners.delete(f); };

// ------------------------------------------------------------------ Pyodide
let worker = null, ready = null, seq = 0;
const pending = new Map();

function startWorker() {
  worker = new Worker(new URL("./pyodide-worker.js", import.meta.url), { type: "module" });
  state.browser = { status: "loading", text: "Python indiriliyor (ilk sefer ~6 MB)" };
  emit();
  ready = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") { state.browser = { status: "ready", text: `Python ${m.python} hazır`, python: m.python }; emit(); resolve(); }
      else if (m.type === "status" && m.id != null) pending.get(m.id)?.onStatus?.(m.text);
      else if (m.type === "status") { state.browser.text = m.text; emit(); }
      else if (m.type === "result") { const p = pending.get(m.id); pending.delete(m.id); p?.resolve(m.result); }
    };
    worker.onerror = (e) => {
      state.browser = { status: "error", text: "Python yüklenemedi (ağ? CDN engelli olabilir)" };
      emit();
      reject(e);
      for (const p of pending.values()) p.resolve({ ok: false, error: "tarayıcı motoru yüklenemedi: " + (e.message || "bilinmeyen hata"), items: [], tables: [], notes: [], stdout: "" });
      pending.clear();
    };
  });
  let base = null;
  try { base = localStorage.getItem("ptl.pyodideBase"); } catch {}
  worker.postMessage({ type: "init", version: cfg.pyodide, base });
  return ready;
}

export function warmBrowser() { if (!worker) startWorker().catch(() => {}); }

export async function runBrowser(code, check, { onStatus } = {}) {
  if (!worker) startWorker().catch(() => {});
  try { await ready; } catch { return { ok: false, error: "tarayıcı motoru yüklenemedi", items: [], tables: [], notes: [], stdout: "" }; }
  const id = ++seq;
  return new Promise((resolve) => { pending.set(id, { resolve, onStatus }); worker.postMessage({ type: "run", id, code, check }); });
}

/** Sonsuz döngü vb. için: worker'ı öldür, yeniden başlat. */
export function stopBrowser() {
  if (!worker) return;
  worker.terminate();
  for (const p of pending.values()) p.resolve({ ok: false, error: "durduruldu", items: [], tables: [], notes: [], stdout: "" });
  pending.clear();
  worker = null;
  state.browser = { status: "idle", text: "durduruldu; bir sonraki çalıştırmada yeniden yüklenecek" };
  emit();
}

// ------------------------------------------------------------------ yerel runner
const KEY = "ptl.runner";
export function runnerConf() {
  try { return { port: cfg.port, token: "", ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return { port: cfg.port, token: "" }; }
}
export function setRunnerConf(c) { try { localStorage.setItem(KEY, JSON.stringify(c)); } catch {} }

let labcheckSrc = null;
async function labcheck() {
  labcheckSrc ??= await (await fetch(new URL("../py/labcheck.py", import.meta.url))).text();
  return labcheckSrc;
}

async function call(path, opts = {}, timeoutMs = 4000) {
  const { port, token } = runnerConf();
  const ctl = new AbortController();
  const t = timeoutMs ? setTimeout(() => ctl.abort(), timeoutMs) : null;
  try {
    const r = await fetch(`http://127.0.0.1:${port}${path}`, { ...opts, signal: ctl.signal, headers: { "content-type": "application/json", "x-ptl-token": token || "", ...(opts.headers || {}) } });
    const body = await r.json().catch(() => ({}));
    return { status: r.status, body };
  } finally { if (t) clearTimeout(t); }
}

export async function checkLocal() {
  const { token } = runnerConf();
  if (!token) { state.local = { status: "off", text: "bağlı değil" }; emit(); return false; }
  state.local = { ...state.local, status: "checking", text: "kontrol ediliyor" }; emit();
  try {
    const { status, body } = await call("/health");
    if (status === 200 && body.ok) {
      const dev = body.cuda ? `CUDA (${body.gpu})` : body.mps ? "MPS" : "CPU";
      state.local = { status: "ok", text: `torch ${body.torch} · ${dev}`, info: body };
    } else if (status === 401) state.local = { status: "auth", text: "token geçersiz — motoru yeniden başlatınca açılan bağlantıyı kullan" };
    else state.local = { status: "off", text: body.error || `HTTP ${status}` };
  } catch {
    state.local = { status: "off", text: "motor çalışmıyor" };
  }
  emit();
  return state.local.status === "ok";
}

export async function runLocal(code, check, { tag = "" } = {}) {
  if (state.local.status !== "ok" && !(await checkLocal())) {
    return { ok: false, offline: true, error: "Yerel PyTorch motoru bağlı değil.", items: [], tables: [], notes: [], stdout: "" };
  }
  try {
    const { status, body } = await call("/run", { method: "POST", body: JSON.stringify({ code, check, labcheck: await labcheck(), tag }) }, 0);
    if (status !== 200) return { ok: false, error: body.error || `HTTP ${status}`, items: [], tables: [], notes: [], stdout: "" };
    return body;
  } catch (e) {
    state.local = { status: "off", text: "bağlantı koptu" }; emit();
    return { ok: false, offline: true, error: "Yerel motora ulaşılamadı (kapandı mı?).", items: [], tables: [], notes: [], stdout: "" };
  }
}

/** Sayfa açılırken: kurulum bağlantısındaki #ptl-token=… ve ?runner=PORT değerlerini kaydet. */
export function captureRunnerLink() {
  const h = new URLSearchParams(location.hash.slice(1));
  const q = new URLSearchParams(location.search);
  if (h.get("ptl-token")) {
    setRunnerConf({ port: Number(q.get("runner")) || cfg.port, token: h.get("ptl-token") });
    history.replaceState(null, "", location.pathname + "#motorlar");
    return true;
  }
  return false;
}
