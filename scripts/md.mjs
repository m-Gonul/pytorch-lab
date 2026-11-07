// Markdown → HTML: marked + sunucu tarafı KaTeX + [[terim]] sözlük bağlantıları.
import { Marked } from "marked";
import katex from "katex";

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const tex = (src, displayMode) =>
  katex.renderToString(src, { displayMode, throwOnError: true, strict: "ignore", output: "html" });

export function makeMarkdown(glossary) {
  const used = new Set();
  const marked = new Marked({ gfm: true });
  marked.use({
    extensions: [
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
