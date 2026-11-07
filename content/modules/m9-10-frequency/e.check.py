import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    for (B, Ci, Co, H, W, k, p) in [(2, 3, 4, 20, 17, 3, 1), (1, 2, 5, 32, 32, 7, 0), (3, 1, 1, 15, 40, 9, 4), (1, 4, 2, 24, 24, (3, 11), 0)]:
        def t(B=B, Ci=Ci, Co=Co, H=H, W=W, k=k, p=p):
            kh, kw = (k, k) if isinstance(k, int) else k
            x = torch.randn(B, Ci, H, W, generator=g, dtype=torch.float64)
            w = torch.randn(Co, Ci, kh, kw, generator=g, dtype=torch.float64)
            close(need(ns, "conv2d_fft")(x, w, p), F.conv2d(x, w, padding=p), atol=1e-9, what="F.conv2d")
        R.test(f"= F.conv2d (x {B}×{Ci}×{H}×{W}, w {Co}×{Ci}×{k}, padding {p})", t)
    if "conv2d_fft" not in ns:
        return
    rows = []
    x = torch.randn(1, 8, 128, 128)
    for k in (3, 7, 15, 31, 63):
        w = torch.randn(8, 8, k, k)
        t1 = bench(lambda: F.conv2d(x, w, padding=k // 2), repeat=3)
        t2 = bench(lambda: ns["conv2d_fft"](x, w, k // 2), repeat=3)
        rows.append([k, fmt_time(t1), fmt_time(t2), "FFT" if t2 < t1 else "doğrudan"])
    R.table("1 × 8 × 128 × 128 → 8 kanal, 'same' dolgu", ["k", "F.conv2d", "conv2d_fft", "hızlı olan"], rows,
            note="Doğrudan konvolüsyon k² ile, FFT neredeyse sabit büyür; kesişim donanıma ve kütüphaneye bağlıdır.")
