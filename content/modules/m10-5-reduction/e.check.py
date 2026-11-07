import os
import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R_):
    ext = need(ns, "build")()
    P = max(2, min(8, os.cpu_count() or 2))
    g = torch.Generator().manual_seed(0)

    def s():
        x = torch.rand(20_000_000, generator=g)
        ref = float(x.double().sum())
        rows = []
        for k in (1, 2, 4, 8):
            v = ext.sum_acc(x, 1, k)
            close(v, ref, rtol=1e-3, what=f"k = {k}")
            rows.append([f"1 thread, {k} biriktirici", fmt_time(bench(lambda: ext.sum_acc(x, 1, k), repeat=3, number=1)), f"{abs(v - ref) / ref:.1e}"])
        rows.append([f"{P} thread, 8 biriktirici", fmt_time(bench(lambda: ext.sum_acc(x, P, 8), repeat=3, number=1)), f"{abs(ext.sum_acc(x, P, 8) - ref) / ref:.1e}"])
        rows.append(["torch.sum (float32)", fmt_time(bench(lambda: x.sum(), repeat=3, number=1)), f"{abs(float(x.sum()) - ref) / ref:.1e}"])
        R_.table("20 M float32 toplamı", ["yöntem", "süre", "göreli hata"], rows,
                 note="Tek biriktiricide her toplama öncekinin sonucunu bekler (FP toplama gecikmesi ~4 çevrim): döngü gecikme-bağımlı. k biriktirici k bağımsız zincir = CPU'da 'ağaç'ın ilk seviyesi; hem hızlı hem daha doğru. -O2 kayan nokta toplamını kendisi yeniden sıralamaz (birleşmeli değil!).")
    R_.test("çoklu biriktirici", s)

    def h():
        img = torch.randint(0, 256, (4096, 4096), dtype=torch.uint8, generator=g)
        flat = torch.full((4096, 4096), 42, dtype=torch.uint8)
        rows = []
        for name, im in (("tekdüze rastgele", img), ("tek renk", flat)):
            ref = torch.bincount(im.flatten(), minlength=256)
            for c in (1, 4):
                ensure(torch.equal(ext.hist_copies(im, P, c), ref), f"{name}, {c} kopya")
                rows.append([name, c, fmt_time(bench(lambda: ext.hist_copies(im, P, c), repeat=3, number=1))])
        R_.table(f"16 M piksel histogramı, {P} thread (özel histogramlar)", ["görüntü", "thread başına alt-histogram", "süre"], rows,
                 note="Tek renkte her piksel aynı sayaca yazar: yükle–artır–yaz zinciri bir öncekinin bitmesini bekler (CPU'daki 'atomik çakışma' benzeri). 4 kopya zinciri 4'e böler. GPU'da aynı fikir: warp başına alt-histogram.")
    R_.test("çoklu alt-histogram", h)
