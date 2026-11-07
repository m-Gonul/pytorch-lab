// Pyodide'i sayfayı dondurmadan bir Web Worker içinde çalıştırır.
// Mesajlar: {type:"init", version} → {type:"ready"} | {type:"run", id, code, check} → {type:"result", id, result}
let py = null;
let booting = null;

async function boot(version, override) {
  // override: yalnız geliştirme/test için yerel Pyodide kopyası (localStorage "ptl.pyodideBase")
  const base = override || `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  const { loadPyodide } = await import(base + "pyodide.mjs");
  postMessage({ type: "status", text: "Python indiriliyor" });
  py = await loadPyodide({ indexURL: base });
  const src = await (await fetch(new URL("../py/labcheck.py", import.meta.url))).text();
  py.FS.writeFile("/home/pyodide/labcheck.py", src);
  await py.runPythonAsync("import sys; sys.path.insert(0, '/home/pyodide'); import labcheck");
  postMessage({ type: "ready", python: py.runPython("import sys; sys.version.split()[0]") });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      booting ??= boot(m.version, m.base);
      await booting;
      return;
    }
    if (m.type === "run") {
      await booting;
      // Kodda ya da kontrolde geçen paketleri (numpy, pandas, …) yükle; torch Pyodide'de yok.
      if (/\bimport\s+torch|\bfrom\s+torch\b/.test(m.code + "\n" + (m.check || ""))) {
        postMessage({ type: "result", id: m.id, result: { ok: false, error: "Bu kod torch kullanıyor; tarayıcı motorunda PyTorch yok. Yerel motoru bağla ya da Colab'da aç.", items: [], tables: [], notes: [], stdout: "" } });
        return;
      }
      postMessage({ type: "status", id: m.id, text: "paketler hazırlanıyor" });
      await py.loadPackagesFromImports(m.code + "\n" + (m.check || ""));
      postMessage({ type: "status", id: m.id, text: "çalışıyor" });
      const run = py.globals.get("labcheck").run_stage_json;
      const json = run(m.code, m.check ?? null);
      run.destroy?.();
      postMessage({ type: "result", id: m.id, result: JSON.parse(json) });
    }
  } catch (err) {
    postMessage({ type: "result", id: m.id, result: { ok: false, error: String(err && err.message || err), items: [], tables: [], notes: [], stdout: "" } });
  }
};
