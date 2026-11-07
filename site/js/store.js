// Tarayıcıda kalan ilerleme ve kod kaydı (sunucu yok; veri cihazdan çıkmaz).
const P = "ptl.progress.v1";
const safe = (f, d) => { try { return f(); } catch { return d; } };

export const progress = {
  all: () => safe(() => JSON.parse(localStorage.getItem(P) || "{}"), {}),
  get: (mod) => progress.all()[mod] || {},
  set(mod, stage, done = true) {
    const a = progress.all();
    a[mod] = { ...(a[mod] || {}), [stage]: done ? 1 : 0 };
    safe(() => localStorage.setItem(P, JSON.stringify(a)));
    dispatchEvent(new CustomEvent("ptl:progress", { detail: { mod, stage, done } }));
  },
  count: (mod) => Object.entries(progress.get(mod)).filter(([k, v]) => "abcde".includes(k) && v).length,
  replace(obj) { safe(() => localStorage.setItem(P, JSON.stringify(obj))); dispatchEvent(new CustomEvent("ptl:progress")); },
  clear() { safe(() => localStorage.removeItem(P)); dispatchEvent(new CustomEvent("ptl:progress")); },
};

export const saved = {
  get: (k) => safe(() => localStorage.getItem("ptl.code." + k), null),
  set: (k, v) => safe(() => localStorage.setItem("ptl.code." + k, v)),
  del: (k) => safe(() => localStorage.removeItem("ptl.code." + k)),
};
