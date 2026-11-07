import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def seq(a, b):
    h, out = torch.zeros_like(b[0]), []
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]; out.append(h)
    return torch.stack(out)


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız offset döngüsü", lambda: max_loops(src, 1, func="linear_scan", why="zaman üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)

    def eq():
        for T in (1, 7, 64, 100):
            a = torch.rand(T, 3, 5, generator=g, dtype=torch.float64) * 0.9 + 0.05
            b = torch.randn(T, 3, 5, generator=g, dtype=torch.float64)
            close(need(ns, "linear_scan")(a, b), seq(a, b), what=f"T = {T}")
    R.test("sıralı rekürsiyonla aynı (T = 1, 7, 64, 100)", eq)

    def gr():
        a = (torch.rand(33, 4, generator=g, dtype=torch.float64) * 0.9).requires_grad_(True)
        b = torch.randn(33, 4, generator=g, dtype=torch.float64).requires_grad_(True)
        need(ns, "linear_scan")(a, b).sum().backward()
        ga, gb = a.grad.clone(), b.grad.clone()
        a.grad = b.grad = None
        seq(a, b).sum().backward()
        close(ga, a.grad, what="∂a"); close(gb, b.grad, what="∂b")
    R.test("autograd scan içinden de doğru akıyor", gr)
    if "linear_scan" not in ns:
        return
    rows = []
    for T in (1024, 8192):
        a = torch.rand(T, 64, generator=g) * 0.99; b = torch.randn(T, 64, generator=g)
        t1, t2 = bench(lambda: seq(a, b), repeat=3, number=1), bench(lambda: ns["linear_scan"](a, b), repeat=3)
        rows.append([str(T), fmt_time(t1), fmt_time(t2), f"{t1 / t2:,.1f}×"])
    R.table("Doğrusal rekürsiyon, 64 kanal", ["T", "sıralı (T adım)", "scan (log₂T tur)", "kazanç"], rows,
            note="Scan O(T log T) iş yapar ama derinliği O(log T). GPU'da Blelloch (O(T) iş) ya da çip-içi kernel'lar kullanılır.")
