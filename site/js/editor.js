// Hafif Python editörü: textarea + girinti davranışları + kısayollar.
const IND = "    ";

export function enhance(ta, { onCheck, onRun, onChange } = {}) {
  const fit = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; };
  const edit = (start, end, text, selStart, selEnd) => {
    ta.setSelectionRange(start, end);
    // execCommand korursa geri al (Ctrl+Z) çalışır; yoksa doğrudan yaz.
    if (!document.execCommand?.("insertText", false, text)) ta.setRangeText(text, start, end, "end");
    if (selStart != null) ta.setSelectionRange(selStart, selEnd ?? selStart);
  };
  ta.addEventListener("keydown", (e) => {
    const { selectionStart: s, selectionEnd: t, value: v } = ta;
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onCheck?.(); return; }
    if (e.shiftKey && e.key === "Enter") { e.preventDefault(); onRun?.(); return; }
    if (e.key === "Escape") { ta.blur(); return; } // klavye tuzağına düşmemek için
    if (e.key === "Tab") {
      e.preventDefault();
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      if (s === t && !e.shiftKey) { edit(s, t, IND); return; }
      const le = v.indexOf("\n", t - (t > s && v[t - 1] === "\n" ? 1 : 0));
      const block = v.slice(ls, le < 0 ? v.length : le);
      const lines = block.split("\n");
      const out = e.shiftKey ? lines.map((l) => l.replace(/^ {1,4}/, "")) : lines.map((l) => IND + l);
      const text = out.join("\n");
      edit(ls, ls + block.length, text, ls, ls + text.length);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && s === t) {
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      const line = v.slice(ls, s);
      let ind = line.match(/^\s*/)[0];
      if (/:\s*(#.*)?$/.test(line)) ind += IND;
      e.preventDefault();
      edit(s, t, "\n" + ind);
      return;
    }
    if (e.key === "Backspace" && s === t && s > 0) {
      const ls = v.lastIndexOf("\n", s - 1) + 1;
      const before = v.slice(ls, s);
      if (before.length && /^ +$/.test(before) && before.length % 4 === 0) { e.preventDefault(); edit(s - 4, s, ""); }
    }
  });
  ta.addEventListener("input", () => { fit(); onChange?.(ta.value); });
  requestAnimationFrame(fit);
  return { fit, set(v) { ta.value = v; fit(); onChange?.(v); } };
}
