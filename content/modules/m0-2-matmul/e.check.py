import random
import numpy as np
import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def py_matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for t in range(k):
            a, Bt = Ai[t], B[t]
            for j in range(m):
                Ci[j] += a * Bt[j]
    return C


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("`batched_matmul` tanımlı", lambda: need(ns, "batched_matmul"))
    bmm = ns.get("batched_matmul")
    if bmm is None:
        return
    g = torch.Generator().manual_seed(0)
    As = torch.randn(32, 4, 6, generator=g)
    Bs = torch.randn(32, 6, 3, generator=g)
    R.test("(32,4,6)·(32,6,3) doğru", lambda: close(bmm(As, Bs), torch.stack([a @ b for a, b in zip(As, Bs)]), rtol=1e-4, atol=1e-5))
    R.test("çıktı shape'i (N, n, m)", lambda: ensure(tuple(bmm(As, Bs).shape) == (32, 4, 3), f"shape {tuple(bmm(As, Bs).shape)}"))

    # --- hız tablosu 1: tek 64×64 çarpım
    n = 64
    rng = random.Random(0)
    A = [[rng.random() for _ in range(n)] for _ in range(n)]
    B = [[rng.random() for _ in range(n)] for _ in range(n)]
    An, Bn = np.array(A), np.array(B)
    At, Bt = torch.tensor(A), torch.tensor(B)
    t_py = bench(lambda: py_matmul(A, B), repeat=3, number=1)
    t_np = bench(lambda: An @ Bn)
    t_th = bench(lambda: At @ Bt)
    R.table(f"Tek çarpım, {n}×{n} ({n**3:,} çarpma)", ["yöntem", "süre", "saf Python'a göre"],
            [["saf Python döngüsü", fmt_time(t_py), "1×"],
             ["NumPy  A @ B", fmt_time(t_np), f"{t_py / t_np:,.0f}× hızlı"],
             ["PyTorch  A @ B", fmt_time(t_th), f"{t_py / t_th:,.0f}× hızlı"]])

    # --- hız tablosu 2: 256 küçük çarpım, döngü vs batched
    As2, Bs2 = torch.randn(256, 8, 8), torch.randn(256, 8, 8)
    t_loop = bench(lambda: torch.stack([a @ b for a, b in zip(As2, Bs2)]))
    t_b = bench(lambda: bmm(As2, Bs2))
    t_at = bench(lambda: As2 @ Bs2)
    R.table("256 adet 8×8 çarpım", ["yöntem", "süre", "döngüye göre"],
            [["for + @ (256 kernel çağrısı)", fmt_time(t_loop), "1×"],
             ["senin batched_matmul", fmt_time(t_b), f"{t_loop / t_b:,.1f}× hızlı"],
             ["As @ Bs", fmt_time(t_at), f"{t_loop / t_at:,.1f}× hızlı"]],
            note="Küçük matrislerde maliyetin çoğu hesap değil, çağrı başına sabit maliyet (yorumlayıcı + dispatch).")
