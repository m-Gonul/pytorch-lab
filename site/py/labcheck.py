"""labcheck — pytorch-lab'in ortak kontrol çekirdeği.

Aynı dosya üç yerde birebir çalışır:
  * tarayıcıda (Pyodide, Web Worker içinde),
  * yerel runner'da (uv run runner/torchlab.py),
  * CI'da ve Colab notebook'larında.

Protokol: öğrenci kodu `exec` ile ayrı bir namespace'te çalıştırılır, ardından
aşamanın kontrol dosyası (`check(ns, R)` fonksiyonu tanımlar) aynı namespace
üzerinde çalışır. Sonuç JSON'a çevrilebilir bir dict olarak döner.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import time
import traceback

__all__ = ["Report", "run_stage", "close", "to_list", "no_loops", "no_imports",
           "uses_any", "need", "ensure", "bench", "fmt_time", "loop_depth", "max_loops"]


# --------------------------------------------------------------------------- yardımcılar
def to_list(x):
    """Tensor / ndarray / liste / skaleri iç içe Python listesine (float) çevirir."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu()
        x = x.tolist()
    elif hasattr(x, "tolist"):  # numpy
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [to_list(v) for v in x]
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return x


def _shape(x):
    if isinstance(x, list):
        return (len(x),) + (_shape(x[0]) if x else ())
    return ()


def _flat(x):
    if isinstance(x, list):
        for v in x:
            yield from _flat(v)
    else:
        yield x


def close(got, want, rtol=1e-5, atol=1e-6, what="sonuç"):
    """`got` ile `want` aynı shape'te ve sayısal olarak yakın mı? Değilse AssertionError."""
    g, w = to_list(got), to_list(want)
    if _shape(g) != _shape(w):
        raise AssertionError(f"{what}: shape {_shape(g)} bekleniyordu {_shape(w)}")
    worst = 0.0
    for a, b in zip(_flat(g), _flat(w)):
        if not isinstance(a, float):
            raise AssertionError(f"{what}: sayı olmayan değer {a!r}")
        if math.isnan(a) and math.isnan(b):
            continue
        err = abs(a - b)
        if err > atol + rtol * abs(b):
            raise AssertionError(f"{what}: {a:.6g} bulundu, {b:.6g} bekleniyordu")
        worst = max(worst, err)
    return worst


def _tree(src):
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def no_loops(src, allow_comprehension=False):
    """Kaynakta `for`/`while` döngüsü (ve istenirse comprehension) var mı? Varsa AssertionError."""
    t = _tree(src)
    if t is None:
        return
    banned = (ast.For, ast.While, ast.AsyncFor)
    if not allow_comprehension:
        banned += (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for node in ast.walk(t):
        if isinstance(node, banned):
            kind = "döngü" if isinstance(node, (ast.For, ast.While, ast.AsyncFor)) else "comprehension"
            raise AssertionError(f"satır {node.lineno}: bu aşamada {kind} yasak — işlemi vektörel yaz")


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def loop_depth(src, func=None):
    """Kaynakta (ya da yalnız `func` fonksiyonunda) iç içe for/while derinliği. Comprehension'lar da sayılır."""
    t = _tree(src)
    if t is None:
        return 0
    if func is not None:
        t = _func(t, func)
        if t is None:
            raise AssertionError(f"`{func}` fonksiyonu bulunamadı")
    loops = (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def depth(node):
        own = 1 if isinstance(node, loops) else 0
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            own = len(node.generators)
        return own + max((depth(c) for c in ast.iter_child_nodes(node)), default=0)
    return depth(t)


def max_loops(src, n, func=None, why=""):
    """Döngü derinliği en fazla n mi? (ör. eğitim adımları için tek döngü serbest, örnekler üzerinde döngü yasak)"""
    d = loop_depth(src, func)
    if d > n:
        where = f"`{func}` içinde " if func else ""
        raise AssertionError(f"{where}{d} katmanlı döngü var; en fazla {n} olmalı{(' — ' + why) if why else ''}")


def no_imports(src, names):
    """Belirtilen modüllerin import edilmediğini doğrular (ör. b aşamasında numpy yok)."""
    t = _tree(src)
    if t is None:
        return
    for node in ast.walk(t):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m in names:
                raise AssertionError(f"satır {node.lineno}: bu aşamada `{m}` kullanılmamalı")


def uses_any(src, words):
    """Kaynak, verilen isimlerden en az birini (attribute ya da isim olarak) kullanıyor mu?"""
    t = _tree(src)
    if t is None:
        return False
    found = set()
    for node in ast.walk(t):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return any(w in found for w in words)


def ensure(cond, msg):
    """`cond` yanlışsa `msg` ile AssertionError."""
    if not cond:
        raise AssertionError(msg)


def need(ns, name, kind="fonksiyon"):
    if name not in ns:
        raise AssertionError(f"`{name}` adlı {kind} tanımlanmamış")
    return ns[name]


def fmt_time(s):
    if s < 1e-6:
        return f"{s * 1e9:.0f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.2f} s"


def bench(fn, repeat=5, number=None, budget=0.25):
    """`fn()`'in medyan süresi (saniye). number verilmezse ~budget saniyeye göre ayarlanır."""
    fn()  # ısınma (lazy init, torch.compile vb.)
    if number is None:
        t0 = time.perf_counter()
        fn()
        one = max(time.perf_counter() - t0, 1e-7)
        number = max(1, min(10_000, int(budget / one / repeat)))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------- rapor
class Report:
    def __init__(self):
        self.items = []    # {"name", "ok", "msg"}
        self.tables = []   # {"title", "cols", "rows", "note"}
        self.notes = []

    def test(self, name, fn=None):
        """`R.test("ad", lambda: ...)` ya da dekoratör olarak `@R.test("ad")`."""
        if fn is None:
            return lambda f: (self.test(name, f), f)[1]
        try:
            res = fn()
            msg = res if isinstance(res, str) else ""
            self.items.append({"name": name, "ok": True, "msg": msg})
        except AssertionError as e:
            self.items.append({"name": name, "ok": False, "msg": str(e) or "koşul sağlanmadı"})
        except Exception as e:  # öğrenci kodu patladı
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"satır {f.lineno}: " for f in reversed(tb) if f.filename == "<öğrenci>"), "")
            self.items.append({"name": name, "ok": False, "msg": f"{where}{type(e).__name__}: {e}"})
        return fn

    def table(self, title, cols, rows, note=""):
        self.tables.append({"title": title, "cols": list(cols), "rows": [list(map(str, r)) for r in rows], "note": note})

    def note(self, text):
        self.notes.append(str(text))

    @property
    def ok(self):
        return bool(self.items) and all(i["ok"] for i in self.items)


def _short_tb(e):
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if f.filename == "<öğrenci>"]
    loc = f"satır {frames[-1].lineno}: " if frames else ""
    return f"{loc}{type(e).__name__}: {e}"


def run_stage(code, check_code=None, max_out=20_000):
    """Öğrenci kodunu çalıştırır, istenirse kontrolleri uygular. JSON'a çevrilebilir dict döner."""
    out = io.StringIO()
    ns = {"__name__": "__main__", "__source__": code}
    t0 = time.perf_counter()
    error = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(compile(code, "<öğrenci>", "exec"), ns)
        except SyntaxError as e:
            error = f"satır {e.lineno}: SyntaxError: {e.msg}"
        except Exception as e:
            error = _short_tb(e)
    elapsed = time.perf_counter() - t0
    R = Report()
    if check_code is not None and error is None:
        cns = {"__name__": "labcheck_stage"}
        with contextlib.redirect_stdout(out):
            try:
                exec(compile(check_code, "<kontrol>", "exec"), cns)
                cns["check"](ns, R)
            except Exception as e:  # kontrol dosyası hatası: kullanıcıya göster
                R.items.append({"name": "kontrol çalıştırılamadı", "ok": False,
                                "msg": f"{type(e).__name__}: {e}"})
    text = out.getvalue()
    if len(text) > max_out:
        text = text[:max_out] + f"\n… (çıktı {len(text) - max_out} karakter kısaltıldı)"
    return {
        "stdout": text,
        "error": error,
        "elapsed": elapsed,
        "checked": check_code is not None,
        "ok": error is None and (check_code is None or R.ok),
        "items": R.items,
        "tables": R.tables,
        "notes": R.notes,
    }


def run_stage_json(code, check_code=None):
    return json.dumps(run_stage(code, check_code), ensure_ascii=False)


def run_file(path, check_code):
    """Colab/Jupyter yardımcısı: `%%writefile` ile yazılmış aşama dosyasını kontrol eder ve özetler."""
    with open(path, encoding="utf-8") as f:
        res = run_stage(f.read(), check_code)
    if res["stdout"].strip():
        print(res["stdout"].rstrip())
    if res["error"]:
        print("✗ hata:", res["error"])
    for it in res["items"]:
        print(("✓ " if it["ok"] else "✗ ") + it["name"] + (f" — {it['msg']}" if it["msg"] else ""))
    for t in res["tables"]:
        print(f"\n{t['title']}")
        w = [max(len(str(c)), *(len(r[i]) for r in t["rows"])) for i, c in enumerate(t["cols"])]
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(t["cols"])))
        for r in t["rows"]:
            print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))
        if t["note"]:
            print(t["note"])
    print("\nSONUÇ:", "geçti ✓" if res["ok"] else "henüz değil ✗")
    return res["ok"]
