"""Her modül için CI denetimi (referans repodaki "model çözüm gerçek motorda çalışır" ilkesi).

  * Her aşamanın model çözümü (X.sol.py) kendi kontrolünden (X.check.py) GEÇMELİ.
  * Her başlangıç iskeleti (X.py) kontrolden GEÇMEMELİ ama sözdizimi geçerli olmalı.
  * El hesabı (a) cevapları, module.yaml'daki `verify` betiğinin hesapladığıyla birebir aynı olmalı.

Çalıştır:  python test/test_modules.py [modül-id ...]
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "site" / "py"))
import labcheck  # noqa: E402

MODULES = ROOT / "content" / "modules"
STAGES = "bcde"


def check_module(d: Path) -> list[str]:
    errs: list[str] = []
    meta = yaml.safe_load((d / "module.yaml").read_text(encoding="utf-8"))
    if meta.get("id") != d.name:
        errs.append(f"id {meta.get('id')!r} klasör adıyla uyuşmuyor")

    # --- (a) el hesabı
    hand = meta.get("hand") or {}
    ns: dict = {}
    exec(hand.get("verify", "answers = {}"), ns)
    answers = ns["answers"]
    seen = set()
    for g in hand.get("groups", []):
        for f in g["fields"]:
            fid = f["id"]
            if not isinstance(fid, str):
                errs.append(f"a: alan id {fid!r} metin değil (YAML 1.1: on/off/yes/no boolean okunur)")
                continue
            if fid in seen:
                errs.append(f"a: alan id tekrar ediyor: {fid}")
            seen.add(fid)
            if fid not in answers:
                errs.append(f"a: verify `{fid}` değerini üretmiyor")
                continue
            want = float(answers[fid])
            tol = float(f.get("tol", 0))
            # Yazılan cevap verify'ın hesapladığıyla (yuvarlama payı dışında) aynı olmalı; tol yalnız öğrenciye tanınan pay.
            if not math.isclose(float(f["answer"]), want, rel_tol=2e-6, abs_tol=max(tol * 0.01, 1e-9)):
                errs.append(f"a: {fid} cevabı {f['answer']} ama verify {want}")
            if g.get("layout") == "matrix" and len(g["fields"]) != g["rows"] * g["cols"]:
                errs.append(f"a: '{g['title']}' matris düzeninde alan sayısı rows*cols değil")
    extra = set(answers) - seen
    if extra:
        errs.append(f"a: verify fazladan değer üretiyor: {sorted(extra)}")

    # --- (b–e) kod aşamaları
    for s in STAGES:
        if s not in (meta.get("stages") or {}):
            errs.append(f"{s}: module.yaml stages içinde yok")
            continue
        for fn in (f"{s}.md", f"{s}.py", f"{s}.sol.py", f"{s}.check.py"):
            if not (d / fn).exists():
                errs.append(f"{s}: {fn} eksik")
        if errs and any(e.startswith(f"{s}:") for e in errs):
            continue
        check = (d / f"{s}.check.py").read_text(encoding="utf-8")
        sol = (d / f"{s}.sol.py").read_text(encoding="utf-8")
        start = (d / f"{s}.py").read_text(encoding="utf-8")
        try:
            ast.parse(start)
        except SyntaxError as e:
            errs.append(f"{s}: başlangıç kodu sözdizimi hatası: {e}")
        r = labcheck.run_stage(sol, check)
        if not r["ok"]:
            bad = [f"{i['name']} — {i['msg']}" for i in r["items"] if not i["ok"]]
            errs.append(f"{s}: model çözüm geçmedi: {r['error'] or bad}")
        r0 = labcheck.run_stage(start, check)
        if r0["ok"]:
            errs.append(f"{s}: başlangıç iskeleti kontrolü GEÇİYOR (kontrol çok gevşek)")
    return errs


def main(argv):
    mods = sorted(p for p in MODULES.iterdir() if (p / "module.yaml").exists())
    if argv:
        mods = [m for m in mods if m.name in argv]
    failed = 0
    for d in mods:
        try:
            errs = check_module(d)
        except Exception as e:  # YAML/verify hatası
            errs = [f"{type(e).__name__}: {e}".splitlines()[0][:400]]
        print(("✓ " if not errs else "✗ ") + d.name)
        for e in errs:
            print("    ", e)
        failed += bool(errs)
    print(f"\n{len(mods) - failed}/{len(mods)} modül temiz")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
