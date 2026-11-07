// Geliştirme sunucusu: dist/ klasörünü http://localhost:8000 üzerinden sunar.
//   npm run dev            → önce build, sonra sun
//   node scripts/serve.mjs dist 8000
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const dir = join(import.meta.dirname, "..", process.argv[2] || "dist");
const port = Number(process.argv[3] || process.env.PORT || 8000);
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json", ".py": "text/x-python; charset=utf-8", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".wasm": "application/wasm", ".zip": "application/zip", ".ipynb": "application/json" };

createServer(async (req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  const file = normalize(join(dir, p));
  if (!file.startsWith(dir)) { res.writeHead(403).end(); return; }
  try {
    const s = await stat(file);
    if (!s.isFile()) throw 0;
    res.writeHead(200, { "content-type": types[extname(file)] || "application/octet-stream", "cache-control": "no-store" });
    res.end(await readFile(file));
  } catch { res.writeHead(404, { "content-type": "text/plain; charset=utf-8" }).end("bulunamadı"); }
}).listen(port, "127.0.0.1", () => console.log(`http://localhost:${port}/`));
