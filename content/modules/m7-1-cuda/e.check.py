import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R):
    ext = need(ns, "ext", "derlenmiş modül")
    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(70, 45, generator=g), torch.randn(45, 33, generator=g)
    R.test("matmul_ijk", lambda: close(ext.matmul_ijk(A, B), A @ B, atol=1e-4))
    R.test("matmul_ikj", lambda: close(ext.matmul_ikj(A, B), A @ B, atol=1e-4))
    R.test("matmul_tiled (T = 16, kenarlar dahil)", lambda: close(ext.matmul_tiled(A, B, 16), A @ B, atol=1e-4))
    n = 512
    X, Y = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
    rows = []
    for name, fn in (("ijk (B'yi sütun boyunca)", lambda: ext.matmul_ijk(X, Y)), ("ikj (ardışık)", lambda: ext.matmul_ikj(X, Y)),
                     ("tiled ikj, T = 64", lambda: ext.matmul_tiled(X, Y, 64)), ("torch.matmul (BLAS)", lambda: X @ Y)):
        t = bench(fn, repeat=3, number=1)
        rows.append([name, fmt_time(t), f"{2 * n ** 3 / t / 1e9:.1f}"])
    R.table(f"n = {n}, tek thread C++ vs BLAS", ["yöntem", "süre", "GFLOP/s"], rows,
            note="Aynı FLOP, farklı bellek erişim düzeni. BLAS ayrıca SIMD, çok thread ve register tiling kullanır.")
