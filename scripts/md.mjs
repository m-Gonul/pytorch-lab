// Markdown → HTML: marked + sunucu tarafı KaTeX + [[terim]] sözlük bağlantıları.
import { Marked } from "marked";
import katex from "katex";

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const tex = (src, displayMode) =>
  katex.renderToString(src, { displayMode, throwOnError: true, strict: "ignore", output: "html" });

const BOX = {
  "türetim": { label: "Türetim", cls: "deriv", fold: true },
  "kanıt": { label: "Kanıt", cls: "proof", fold: true },
  "ayrıntı": { label: "Ayrıntı", cls: "detail", fold: true },
  "çözüm": { label: "Çözüm", cls: "sol", fold: true },
  "sezgi": { label: "Sezgi", cls: "intuition", fold: false },
  "tanım": { label: "Tanım", cls: "def", fold: false },
  "örnek": { label: "Örnek", cls: "example", fold: false },
  "uyarı": { label: "Dikkat", cls: "warn", fold: false },
  "yanılgı": { label: "Sık yanılgı", cls: "pitfall", fold: false },
  "koşul": { label: "Bu koşul değişirse", cls: "whatif", fold: true },
};

export function makeMarkdown(glossary) {
  const used = new Set();
  const marked = new Marked({ gfm: true });
  marked.use({
    extensions: [
      {
        // :::türetim Başlık … ::: — katmanlı anlatım kutuları. Açılır (details) olanlar:
        // türetim, kanıt, ayrıntı, çözüm; her zaman açık olanlar: sezgi, örnek, uyarı, tanım, yanılgı.
        name: "box",
        level: "block",
        start: (s) => { const i = s.search(/^:::[^\s:]/m); return i < 0 ? undefined : i; },
        tokenizer(src) {
          const m = /^:::([^\s:]+)[ \t]*([^\n]*)\n([\s\S]*?)\n:::[ \t]*(?:\n+|$)/.exec(src);
          if (!m) return;
          const kind = m[1].toLocaleLowerCase("tr");
          if (!BOX[kind]) throw new Error(`bilinmeyen kutu türü: :::${m[1]}`);
          const t = { type: "box", raw: m[0], kind, title: m[2].trim(), tokens: [] };
          this.lexer.blockTokens(m[3], t.tokens);
          return t;
        },
        renderer(t) {
          const b = BOX[t.kind];
          const head = `<span class="box-kind">${b.label}</span>${t.title ? ` ${marked.parseInline(t.title)}` : ""}`;
          const inner = this.parser.parse(t.tokens);
          return b.fold
            ? `<details class="box box-${b.cls}"><summary>${head}</summary><div class="box-body">${inner}</div></details>\n`
            : `<aside class="box box-${b.cls}"><p class="box-head">${head}</p><div class="box-body">${inner}</div></aside>\n`;
        },
      },
      {
        name: "mathBlock",
        level: "block",
        start: (s) => s.indexOf("$$"),
        tokenizer(src) {
          const m = /^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(src);
          if (m) return { type: "mathBlock", raw: m[0], text: m[1].trim() };
        },
        renderer: (t) => `<div class="math-display">${tex(t.text, true)}</div>\n`,
      },
      {
        name: "mathInline",
        level: "inline",
        start: (s) => s.indexOf("$"),
        tokenizer(src) {
          const m = /^\$(?!\$)((?:\\.|[^\\$\n])+?)\$/.exec(src);
          if (m) return { type: "mathInline", raw: m[0], text: m[1] };
        },
        renderer: (t) => tex(t.text, false),
      },
      {
        name: "term",
        level: "inline",
        start: (s) => s.indexOf("[["),
        tokenizer(src) {
          const m = /^\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/.exec(src);
          if (m) return { type: "term", raw: m[0], key: m[1].trim(), text: (m[2] || m[1]).trim() };
        },
        renderer(t) {
          const k = t.key.toLowerCase();
          const def = glossary[k];
          if (!def) throw new Error(`sözlükte olmayan terim: [[${t.key}]]`);
          used.add(k);
          return `<dfn class="term" tabindex="0" data-tip="${esc(def)}">${esc(t.text)}</dfn>`;
        },
      },
    ],
    renderer: {
      heading({ tokens, depth, text }) {
        // "## Başlık {#kimlik}" → id="kimlik"; yoksa başlıktan slug üret
        let id = null;
        const m = /\s*\{#([\w-]+)\}\s*$/.exec(text);
        if (m) {
          const last = tokens[tokens.length - 1];
          if (last?.type === "text") last.text = last.raw = last.text.replace(/\s*\{#[\w-]+\}\s*$/, "");
          id = m[1];
        } else {
          id = text.toLocaleLowerCase("tr").replace(/[çğıöşü]/g, (c) => ({ ç: "c", ğ: "g", ı: "i", ö: "o", ş: "s", ü: "u" })[c]).replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
        }
        return `<h${depth} id="${esc(id)}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
      },
      code({ text, lang }) {
        const l = (lang || "").split(/\s/)[0];
        return `<pre class="code${l ? " lang-" + esc(l) : ""}"><code>${esc(text)}</code></pre>\n`;
      },
      link({ href, title, tokens }) {
        const text = this.parser.parseInline(tokens);
        const ext = /^https?:/.test(href);
        return `<a href="${esc(href)}"${title ? ` title="${esc(title)}"` : ""}${ext ? ' rel="noopener" target="_blank"' : ""}>${text}</a>`;
      },
    },
  });
  return {
    block: (s) => marked.parse(String(s ?? "")),
    inline: (s) => marked.parseInline(String(s ?? "")),
    used,
  };
}
