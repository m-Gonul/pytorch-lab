import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def loops(x, w, b):
    N, C, H, W = x.shape; Fo, _, KH, KW = w.shape
    y = torch.empty(N, Fo, H - KH + 1, W - KW + 1)
    for i in range(H - KH + 1):
        for j in range(W - KW + 1):
            y[:, :, i, j] = torch.einsum("nckl,fckl->nf", x[:, :, i:i + KH, j:j + KW], w) + b
    return y


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 10, 9, generator=g, dtype=torch.float64); w = torch.randn(4, 3, 3, 3, generator=g, dtype=torch.float64); b = torch.randn(4, generator=g, dtype=torch.float64)

    def eq():
        for s, p in [(1, 0), (2, 1), (1, 2)]:
            close(need(ns, "conv2d_strided")(x, w, b, s, p), F.conv2d(x, w, b, stride=s, padding=p), what=f"stride {s}, pad {p}")
    R.test("conv2d_strided = F.conv2d", eq)
    if "conv2d_strided" not in ns:
        return
    xb = torch.randn(16, 32, 32, 32, generator=g); wb = torch.randn(64, 32, 3, 3, generator=g); bb = torch.randn(64, generator=g)

    def unf():
        return (wb.reshape(64, -1) @ F.unfold(xb, 3, padding=1) + bb[:, None]).reshape(16, 64, 32, 32)
    xs, ws = xb[:2, :8, :16, :16], wb[:8, :8]
    t_loop = bench(lambda: loops(xs, ws, bb[:8]), repeat=3, number=1)
    t_vs = bench(lambda: ns["conv2d_strided"](xs, ws, bb[:8], 1, 0), repeat=3)
    t_unf, t_str, t_ref = bench(unf, repeat=3), bench(lambda: ns["conv2d_strided"](xb, wb, bb, 1, 1), repeat=3), bench(lambda: F.conv2d(xb, wb, bb, padding=1), repeat=3)
    R.table("Convolution yolları", ["yöntem", "girdi", "süre"],
            [["konum döngüsü + einsum", "(2, 8, 16, 16)", fmt_time(t_loop)], ["senin conv2d_strided", "(2, 8, 16, 16)", fmt_time(t_vs)],
             ["F.unfold + matmul (im2col, kopya)", "(16, 32, 32, 32)", fmt_time(t_unf)], ["senin conv2d_strided (view)", "(16, 32, 32, 32)", fmt_time(t_str)],
             ["F.conv2d (oneDNN/cuDNN)", "(16, 32, 32, 32)", fmt_time(t_ref)]],
            note="İm2col matrisi bu örnekte 9 kat bellek: 16·288·1024 eleman. Kütüphane kernel'ları bunu oluşturmadan (implicit GEMM, Winograd) hesaplar.")
