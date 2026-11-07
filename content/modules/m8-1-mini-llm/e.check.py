import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("FusedAttention.forward'da döngü yok", lambda: max_loops(src, 0, func="forward") if False else _no_loop_in_fused(src))
    Loop = need(ns, "LoopAttention", "sınıf")
    torch.manual_seed(0)
    loop = Loop()
    x = torch.randn(16, 64, 128)

    def tops():
        names = need(ns, "top_ops")(lambda: loop(x), 5)
        ensure(isinstance(names, list) and len(names) == 5 and all(isinstance(n, str) for n in names), "5 işlem adı")
        ensure(any(n.startswith("aten::") for n in names), "aten işlem adları bekleniyordu")
        return "en pahalı işlemler: " + ", ".join(names)
    R.test("top_ops (profiler)", tops)

    def eq():
        f = need(ns, "FusedAttention", "sınıf").from_loop(loop)
        with torch.no_grad():
            close(f(x), loop(x), atol=1e-5)
        ensure(sum(isinstance(m, torch.nn.Linear) for m in f.modules()) == 2, "tek qkv + proj: 2 Linear")
    R.test("FusedAttention = LoopAttention", eq)
    if "FusedAttention" not in ns:
        return
    f = ns["FusedAttention"].from_loop(loop)

    def step(m):
        m.zero_grad()
        m(x).pow(2).mean().backward()
    t_loop = bench(lambda: step(loop), repeat=5)
    t_fused = bench(lambda: step(f), repeat=5)
    R.table("İleri + geri, B = 16, T = 64, d = 128, 8 head", ["", "süre"], [["head döngüsü", fmt_time(t_loop)], ["birleşik", fmt_time(t_fused)]],
            note=f"{t_loop / t_fused:.1f}× — 24 küçük matmul yerine 1 büyük matmul + fused attention.")


def _no_loop_in_fused(src):
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "FusedAttention":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                    for sub in ast.walk(fn):
                        if isinstance(sub, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)):
                            raise AssertionError("FusedAttention.forward içinde döngü var")
            return
    raise AssertionError("FusedAttention sınıfı yok")
