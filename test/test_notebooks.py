"""Üretilen notebook'ları Jupyter'siz yürütür: `%%writefile` hücreleri dosya yazar, diğerleri exec edilir.
Aşama dosyalarına model çözümü konur; her `labcheck.run_file(...)` True dönmeli.

Çalıştır:  python test/test_notebooks.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(nb_path: Path) -> list[str]:
    mod = ROOT / "content" / "modules" / nb_path.stem
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    errs: list[str] = []
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        sys.path.insert(0, td)
        ns: dict = {"__name__": "__main__"}
        try:
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                lines = [l for l in src.split("\n") if not l.startswith("#@title")]
                if lines and lines[0].startswith("%%writefile "):
                    name = lines[0].split(None, 1)[1].strip()
                    body = "\n".join(lines[1:])
                    if name.startswith("stage_"):
                        body = (mod / f"{name[6]}.sol.py").read_text(encoding="utf-8")
                    Path(name).write_text(body, encoding="utf-8")
                    continue
                code = "\n".join(("_r = " + l) if l.startswith("labcheck.run_file(") else l for l in lines)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, "<hücre>", "exec"), ns)
                if "_r" in ns:
                    if ns["_r"] is not True:
                        errs.append("kontrol geçmedi:\n" + buf.getvalue()[-1500:])
                    del ns["_r"]
        except Exception as e:  # hücre patladı
            errs.append(f"{type(e).__name__}: {e}")
        finally:
            sys.path.remove(td)
            for m in [m for m in sys.modules if m == "labcheck" or m.startswith("checks_")]:
                del sys.modules[m]
            os.chdir(old)
    return errs


def main():
    # Her notebook ayrı bir süreçte: Colab'da da her notebook kendi çekirdeğinde çalışır
    # (ör. torch.compile'ın import ettiği triton, sonraki notebook'un TRITON_INTERPRET ayarını bozmasın).
    import subprocess
    nbs = sorted((ROOT / "notebooks").glob("*.ipynb"))
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if only:
        nbs = [p for p in nbs if p.stem in only]
    bad = 0
    for p in nbs:
        r = subprocess.run([sys.executable, __file__, "--one", str(p)], capture_output=True, text=True, encoding="utf-8")
        ok = r.returncode == 0
        print(("✓ " if ok else "✗ ") + p.name)
        if not ok:
            print("    " + (r.stdout + r.stderr)[-3000:].replace("\n", "\n    "))
        bad += not ok
    print(f"\n{len(nbs) - bad}/{len(nbs)} notebook temiz")
    return 1 if bad or not nbs else 0


def one(path: str) -> int:
    errs = run(Path(path))
    for e in errs:
        print(e)
    return 1 if errs else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        sys.exit(one(sys.argv[2]))
    sys.exit(main())
