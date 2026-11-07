import torch
from labcheck import need, ensure, fmt_time


def check(ns, R_):
    def st():
        strides = [1, 2, 4, 8, 16, 32, 64]
        c = need(ns, "strided_cost")(1 << 25, strides)
        ensure(len(c) == len(strides) and c[4] > 2 * c[0], "adım büyüdükçe eleman başına maliyet artmalı")
        R_.table("x[::s].sum(), x 32 M float32 (128 MB)", ["adım s", "eleman başına (ns)", "satır başına kullanılan eleman"], [[s, f"{v:.2f}", max(1, 16 // s)] for s, v in zip(strides, c)],
                 note="CPU belleği 64 B satırlarla taşır; s ≥ 16'da her eleman ayrı satır getirir ve maliyet doyar — GPU'da warp başına sektör sayısının karşılığı. (Donanım önceden getirme (prefetch) ardışık adımları kısmen gizler.)")
    R_.test("adımlı erişim", st)

    def tp():
        N = 2048
        res = need(ns, "transpose_variants")(N)
        ensure(len(res) == 3, "üç yöntem")
        R_.table(f"{N}×{N} float32 transpoz kopyası (CPU)", ["yöntem", "süre"], [[k, fmt_time(v)] for k, v in res.items()],
                 note="PyTorch'un kopya çekirdeği zaten bloklu çalışır; Python döngüsü her satır için çağrı ek yükü ve adımlı yazmayla en yavaşı.")
    R_.test("transpoz", tp)
