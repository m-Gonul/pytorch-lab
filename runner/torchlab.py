# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.1",
#   "torch>=2.3",
#   "ninja>=1.11",
#   "triton>=3.0; sys_platform == 'linux' and platform_machine == 'x86_64'",
# ]
# ///
"""pytorch-lab yerel motoru.

Sitedeki PyTorch aşamalarını senin bilgisayarında, gerçek PyTorch ile çalıştırır.

    uv run https://m-gonul.github.io/pytorch-lab/runner/torchlab.py

`uv` ilk çalıştırmada izole bir ortam kurar (torch, numpy, pandas) ve önbelleğe alır.
Motor yalnızca 127.0.0.1'i dinler, her istekte rastgele bir oturum token'ı ister ve
yalnızca izin verilen sitelerden (origin) gelen istekleri kabul eder.

Seçenekler:  --port 8765   --timeout 180   --origin https://ornek.site   --no-open
CPU-only torch (Linux'ta CUDA paketlerini indirmemek için):
    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match <url>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

VERSION = "1.0"
SITE = "https://m-gonul.github.io/pytorch-lab/"
DEFAULT_ORIGINS = {"https://m-gonul.github.io"}
MAX_BODY = 2_000_000

# Isıtılmış worker: ağır import'ları önceden yapar, stdin'den tek bir iş bekler.
WORKER = r'''
import sys, json, types, io
import numpy, torch  # ısınma: ilk import en pahalı kısım
try:
    import pandas
except Exception:
    pass
sys.stdout.write("READY\n"); sys.stdout.flush()
job = json.loads(sys.stdin.readline())
mod = types.ModuleType("labcheck")
exec(compile(job["labcheck"], "labcheck.py", "exec"), mod.__dict__)
sys.modules["labcheck"] = mod
res = mod.run_stage(job["code"], job.get("check"))
sys.stdout.write("\x1eRESULT" + json.dumps(res, ensure_ascii=False) + "\n"); sys.stdout.flush()
'''


def origin_ok(origin: str, extra: set[str]) -> bool:
    if not origin:
        return False
    if origin in DEFAULT_ORIGINS or origin in extra:
        return True
    u = urlparse(origin)
    return u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1")


class Pool:
    """Her zaman bir sonraki iş için hazır bekleyen tek bir worker süreci tutar."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next = None
        self._spawn()

    def _spawn(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        p = subprocess.Popen([sys.executable, "-I", "-c", WORKER], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", env=env)
        self.next = p

    def run(self, job: dict, timeout: float) -> dict:
        with self.lock:  # sıralı çalıştır: hız ölçümleri birbirini bozmasın
            p, self.next = self.next, None
            threading.Thread(target=self._spawn, daemon=True).start()
            t0 = time.time()
            try:
                line = p.stdout.readline()  # READY
                if line.strip() != "READY":
                    err = p.stderr.read()
                    return {"ok": False, "error": f"worker başlatılamadı: {line}{err[-2000:]}", "items": [], "tables": [], "notes": [], "stdout": ""}
                out, err = p.communicate(json.dumps(job) + "\n", timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                return {"ok": False, "error": f"zaman aşımı ({timeout:.0f} s) — sonsuz döngü ya da çok büyük bir hesap olabilir", "items": [], "tables": [], "notes": [], "stdout": ""}
            marker = out.rfind("\x1eRESULT")
            if marker < 0:
                tail = (out + err)[-3000:]
                return {"ok": False, "error": f"worker beklenmedik biçimde sonlandı (çıkış kodu {p.returncode}).\n{tail}", "items": [], "tables": [], "notes": [], "stdout": ""}
            res = json.loads(out[marker + len("\x1eRESULT"):])
            res["wall"] = time.time() - t0
            return res


def info() -> dict:
    import torch
    d = {"ok": True, "runner": VERSION, "python": platform.python_version(), "platform": platform.platform(),
         "torch": torch.__version__, "cuda": torch.cuda.is_available(),
         "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
         "threads": torch.get_num_threads()}
    if d["cuda"]:
        d["gpu"] = torch.cuda.get_device_name(0)
    return d


def make_handler(token: str, extra: set[str], pool: Pool, timeout: float, meta: dict):
    class H(BaseHTTPRequestHandler):
        server_version = "torchlab/" + VERSION

        def log_message(self, fmt, *args):  # sessiz; yalnız çalıştırmaları yaz
            pass

        def _cors(self):
            o = self.headers.get("Origin", "")
            if origin_ok(o, extra):
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Vary", "Origin")
            return origin_ok(o, extra)

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            if self._cors():
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "content-type, x-ptl-token")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def _authed(self) -> bool:
            if not origin_ok(self.headers.get("Origin", ""), extra) and self.headers.get("Origin"):
                self._json(403, {"ok": False, "error": "bu site için izin yok (--origin ile ekleyebilirsin)"})
                return False
            if not secrets.compare_digest(self.headers.get("X-PTL-Token", ""), token):
                self._json(401, {"ok": False, "error": "token geçersiz — motoru başlatınca açılan bağlantıyı kullan"})
                return False
            return True

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                if not self._authed():
                    return
                self._json(200, meta)
            else:
                self._json(404, {"ok": False, "error": "yok"})

        def do_POST(self):
            if self.path.split("?")[0] != "/run":
                return self._json(404, {"ok": False, "error": "yok"})
            if not self._authed():
                return
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                return self._json(413, {"ok": False, "error": "istek boyutu geçersiz"})
            try:
                job = json.loads(self.rfile.read(n))
                assert isinstance(job.get("code"), str) and isinstance(job.get("labcheck"), str)
            except Exception:
                return self._json(400, {"ok": False, "error": "geçersiz istek"})
            tag = job.get("tag", "?")
            t0 = time.time()
            res = pool.run({"code": job["code"], "check": job.get("check"), "labcheck": job["labcheck"]}, timeout)
            mark = "✓" if res.get("ok") else "✗"
            print(f"  {time.strftime('%H:%M:%S')}  {mark} {tag:<24} {time.time() - t0:5.2f} s", flush=True)
            self._json(200, res)

    return H


def main():
    ap = argparse.ArgumentParser(description="pytorch-lab yerel PyTorch motoru")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--origin", action="append", default=[], help="ek izinli site origin'i (ör. https://kullanici.github.io)")
    ap.add_argument("--site", default=SITE, help="açılacak site adresi")
    ap.add_argument("--token", default=None, help="sabit token (varsayılan: her başlatmada rastgele)")
    ap.add_argument("--no-open", action="store_true", help="tarayıcıyı otomatik açma")
    a = ap.parse_args()

    token = a.token or secrets.token_urlsafe(18)
    extra = {o.rstrip("/") for o in a.origin}
    site_origin = "{0.scheme}://{0.netloc}".format(urlparse(a.site))
    extra.add(site_origin)
    print("pytorch-lab motoru başlıyor… (ilk seferde torch import'u birkaç saniye sürebilir)", flush=True)
    meta = info()
    pool = Pool()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(token, extra, pool, a.timeout, meta))
    link = f"{a.site.rstrip('/')}/kurulum.html?runner={a.port}#ptl-token={token}"
    dev = "CUDA: " + meta["gpu"] if meta.get("cuda") else ("MPS (Apple GPU)" if meta.get("mps") else "yalnız CPU")
    print(f"""
  torch {meta['torch']} · Python {meta['python']} · {dev}
  dinleniyor: http://127.0.0.1:{a.port}   (yalnız bu bilgisayar)

  Siteyi bu motora bağlamak için şu bağlantıyı aç (token içerir, kimseyle paylaşma):
  {link}

  Durdurmak için Ctrl+C.
""", flush=True)
    if not a.no_open:
        try:
            webbrowser.open(link)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatıldı.")


if __name__ == "__main__":
    main()
