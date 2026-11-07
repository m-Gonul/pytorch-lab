import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)
    tol = 1e-4 if dev == "cpu" else 2e-2   # GPU'da tl.dot TF32 kullanabilir

    def mm():
        for m, k, n in ((70, 50, 40), (1, 33, 65), (64, 64, 64)):
            a, b = torch.randn(m, k, generator=g).to(dev), torch.randn(k, n, generator=g).to(dev)
            close(need(ns, "matmul")(a, b), a @ b, atol=tol, rtol=tol, what=f"{m}×{k}×{n}")
    R.test("matmul: tek sayılı şekiller", mm)

    def relu():
        a, b = torch.randn(45, 37, generator=g).to(dev), torch.randn(37, 29, generator=g).to(dev)
        out = need(ns, "matmul")(a, b, "relu")
        close(out, torch.relu(a @ b), atol=tol, rtol=tol)
        ensure((out >= 0).all(), "ReLU epilogue negatif bırakmamalı")
    R.test("ReLU epilogue", relu)

    def tr():
        a = torch.randn(40, 30, generator=g).to(dev).t()
        b = torch.randn(40, 20, generator=g).to(dev)
        close(need(ns, "matmul")(a, b), a @ b, atol=tol, rtol=tol, what="transpoze A (stride'lar)")
    R.test("bitişik olmayan A (stride'larla)", tr)
    if dev == "cuda" and "matmul" in ns:
        a, b = torch.randn(2048, 2048, device="cuda"), torch.randn(2048, 2048, device="cuda")
        t1 = bench(lambda: ns["matmul"](a, b), repeat=10)
        t2 = bench(lambda: a @ b, repeat=10)
        R.table("2048³ matmul (GPU)", ["yöntem", "süre", "TFLOP/s"],
                [["senin Triton", fmt_time(t1), f"{2 * 2048 ** 3 / t1 / 1e12:.1f}"], ["torch (cuBLAS)", fmt_time(t2), f"{2 * 2048 ** 3 / t2 / 1e12:.1f}"]])
    else:
        R.note("CPU yorumlayıcısında çalıştı: doğruluk sınandı; hız için GPU gerekir.")
