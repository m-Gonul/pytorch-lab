import os
import time
import torch
from labcheck import close, need, ensure, fmt_time


def ref_mandel(H, W, maxit):
    c = torch.linspace(0, W - 1, W, dtype=torch.float64)
    r = torch.linspace(0, H - 1, H, dtype=torch.float64)
    cr = (-2.0 + 3.0 * c / (W - 1))[None].expand(H, W)
    ci = (1.5 * r / (H - 1))[:, None].expand(H, W)
    zr = torch.zeros(H, W, dtype=torch.float64)
    zi = torch.zeros_like(zr)
    k = torch.zeros(H, W, dtype=torch.int32)
    for _ in range(maxit):
        alive = zr * zr + zi * zi <= 4.0
        if not alive.any():
            break
        t = zr * zr - zi * zi + cr
        zi = torch.where(alive, 2 * zr * zi + ci, zi)
        zr = torch.where(alive, t, zr)
        k += alive.int()
    return k


def check(ns, R_):
    ext = need(ns, "build")()

    def corr():
        ref = ref_mandel(120, 180, 300)
        for p in (1, 3):
            for m in ("block", "cyclic", "dynamic"):
                img, work = ext.mandelbrot(120, 180, 300, p, m)
                ensure(img.dtype == torch.int32 and tuple(work.shape) == (p,), "dönüş tipleri")
                agree = float((img == ref).double().mean())
                ensure(agree > 0.999, f"{m}/{p}: referansla uyum {agree:.4f}")
                ensure(int(work.sum()) == int(img.long().sum()), "iş toplamı = iterasyon toplamı")
    R_.test("doğruluk", corr)

    def bal():
        H, W, it = 600, 900, 1000
        p = max(2, min(8, os.cpu_count() or 2))
        rows = []
        for m in ("block", "cyclic", "dynamic"):
            best = 1e9
            for _ in range(3):
                t0 = time.perf_counter()
                _, work = ext.mandelbrot(H, W, it, p, m)
                best = min(best, time.perf_counter() - t0)
            imb = float(work.max()) / float(work.double().mean())
            rows.append([m, fmt_time(best), f"{imb:.2f}", " ".join(f"{float(x) / 1e6:.0f}M" for x in work)])
            if m == "block":
                imb_block = imb
            if m == "cyclic":
                imb_cyc = imb
        t0 = time.perf_counter()
        ext.mandelbrot(H, W, it, 1, "block")
        t1 = time.perf_counter() - t0
        ensure(imb_block > 1.2 and imb_cyc < 1.1, "blok dağıtım dengesiz, döngüsel dengeli olmalı")
        R_.table(f"Mandelbrot {H}×{W}, maxit {it}, {p} thread (1 thread: {fmt_time(t1)})", ["dağıtım", "süre", "en çok iş / ortalama", "thread başına iterasyon"], rows,
                 note="Üst yarı düzlemde kümenin içi (pahalı satırlar) eksene yakın: ardışık bloklarda ilk thread'in işi çok fazla. Döngüsel dağıtım satırları serpiştirir; dinamik dağıtım hiçbir maliyet modeli bilmeden dengeler ama her satırda atomik sayaç maliyeti öder.")
    R_.test("yük dengesi", bal)
