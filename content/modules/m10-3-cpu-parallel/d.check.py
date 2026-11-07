import os
import time
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure, fmt_time


def best(f, r=3):
    b = 1e9
    for _ in range(r):
        t0 = time.perf_counter()
        f()
        b = min(b, time.perf_counter() - t0)
    return b


def gauss(K):
    g = torch.exp(-((torch.arange(K) - K // 2).float() ** 2) / (2 * (K / 5) ** 2))
    k = torch.outer(g, g)
    return (k / k.sum()).float()


def ref_conv(img, k):
    K = k.shape[0]
    acc = F.conv2d(img.float()[None, None], k[None, None], padding=K // 2)[0, 0]
    return torch.clamp((acc + 0.5).floor(), 0, 255).to(torch.uint8)


def check(ns, R_):
    ext = need(ns, "build")()
    P = max(2, min(8, os.cpu_count() or 2))
    g = torch.Generator().manual_seed(0)

    def corr():
        for n in (1, 7, 4096, 100003):
            x = torch.randint(-1000, 1000, (n,), dtype=torch.int32, generator=g)
            for p in (1, 3, P):
                ensure(torch.equal(ext.integral_array(x, p), torch.cumsum(x.long(), 0)), f"integral dizi n = {n}, p = {p}")
        img = torch.randint(0, 256, (123, 77), dtype=torch.uint8, generator=g)
        for p in (1, 2, 5):
            ensure(torch.equal(ext.integral_image(img, p), img.long().cumsum(0).cumsum(1)), f"integral görüntü p = {p}")
        for K in (3, 7, 11, 21):
            k = gauss(K)
            out = ext.conv2d(img, k, 3)
            ref = ref_conv(img, k)
            d = (out.int() - ref.int()).abs()
            ensure(int(d.max()) <= 1 and float((d == 0).double().mean()) > 0.99, f"konvolüsyon K = {K}")
            ensure(torch.equal(out, ext.conv2d(img, k, 1)), "thread sayısı sonucu değiştirmemeli")
        A = torch.randn(70, 90, generator=g)
        B = torch.randn(90, 50, generator=g)
        for mode in ("ijk", "ikj", "bt", "tiled"):
            close(ext.matmul(A, B, 3, mode).numpy(), (A @ B).numpy(), atol=1e-4, what=f"matmul {mode}")
    R_.test("doğruluk", corr)

    def timing():
        x = torch.randint(-1000, 1000, (4_000_000,), dtype=torch.int32, generator=g)
        img = torch.randint(0, 256, (2048, 2048), dtype=torch.uint8, generator=g)
        im1 = img[:1024, :1024].contiguous()
        A = torch.randn(512, 512, generator=g)
        B = torch.randn(512, 512, generator=g)
        jobs = [("integral dizi 4 M", lambda p: ext.integral_array(x, p)), ("integral görüntü 2048²", lambda p: ext.integral_image(img, p)),
                ("konv. 1024², 7×7", lambda p: ext.conv2d(im1, gauss(7), p)), ("konv. 1024², 21×21", lambda p: ext.conv2d(im1, gauss(21), p)),
                ("matmul 512 ijk", lambda p: ext.matmul(A, B, p, "ijk")), ("matmul 512 ikj", lambda p: ext.matmul(A, B, p, "ikj")),
                ("matmul 512 bt", lambda p: ext.matmul(A, B, p, "bt")), ("matmul 512 tiled", lambda p: ext.matmul(A, B, p, "tiled"))]
        rows = []
        for name, f in jobs:
            t1 = best(lambda: f(1), 2)
            tp = best(lambda: f(P), 3)
            rows.append([name, fmt_time(t1), fmt_time(tp), f"{t1 / tp:.2f}×"])
        R_.table(f"C++ std::thread, 1 vs {P} thread", ["çekirdek", "1 thread", f"{P} thread", "hızlanma"], rows,
                 note="Bellek-bağımlı çekirdekler (integral dizi/görüntü) bant genişliğinde doyar; hesap-bağımlı olanlar (büyük filtreli konvolüsyon, matmul) çekirdek sayısıyla ölçeklenir. Matmul'da döngü sırası thread sayısından daha büyük fark yaratabilir.")
    R_.test("ölçümler", timing)
