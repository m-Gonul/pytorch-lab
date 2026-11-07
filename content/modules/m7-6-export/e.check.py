import os
import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R):
    m = need(ns, "m", "model")
    run = need(ns, "run", "yüklenmiş paket")
    path = need(ns, "path", "paket yolu")

    def eq():
        ensure(os.path.exists(path), "paket dosyası yok")
        with torch.inference_mode():
            for B in (1, 9, 128):
                x = torch.randn(B, 64)
                close(run(x), m(x), atol=1e-5, what=f"batch {B}")
        return f"paket: {os.path.getsize(path):,} bayt"
    R.test("AOTInductor paketi = eager (farklı batch'ler)", eq)

    def reload():
        r2 = need(ns, "load_package")(path)
        x = torch.randn(3, 64)
        with torch.inference_mode():
            close(r2(x), m(x), atol=1e-5)
    R.test("paket yeniden yüklenebiliyor", reload)
    x1, x128 = torch.randn(1, 64), torch.randn(128, 64)
    with torch.inference_mode():
        rows = []
        for name, fn in (("eager", m), ("AOTInductor", run)):
            rows.append([name, fmt_time(bench(lambda: fn(x1), repeat=5)), fmt_time(bench(lambda: fn(x128), repeat=5))])
    R.table("Çıkarım gecikmesi (CPU)", ["", "batch 1", "batch 128"], rows,
            note="Küçük batch'te Python ve modül çağrı yükü baskındır; derlenmiş paket bunu ve ara tensörleri ortadan kaldırır.")
