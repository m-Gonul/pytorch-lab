import numpy as np
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def ref(frames, alpha, var0=100.0):
    B = [frames[0]]; V = [torch.full_like(frames[0], var0)]
    for t in range(1, len(frames)):
        V.append((1 - alpha) * V[-1] + alpha * (frames[t] - B[-1]) ** 2)
        B.append((1 - alpha) * B[-1] + alpha * frames[t])
    return torch.stack(B), torch.stack(V)


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız parça döngüsü", lambda: max_loops(src, 1, func="ema_scan"))
    g = torch.Generator().manual_seed(0)
    F = torch.rand(300, 24, 32, generator=g, dtype=torch.float64) * 255

    def sc():
        Bw, Vw = ref(F, 0.05)
        for L in (1, 7, 64, 300):
            close(need(ns, "ema_scan")(F, 0.05, L).numpy(), Bw.numpy(), atol=1e-8, what=f"L = {L}")
        B, V = need(ns, "ema_var_scan")(F, 0.05)
        close(V.numpy(), Vw.numpy(), atol=1e-6, what="varyans")
    R.test("ema_scan = döngülü özyineleme", sc)

    def big():
        try:
            out = ns["ema_scan"](F[:, :2, :2].repeat(10, 1, 1), 0.2, 64)
            ensure(torch.isfinite(out).all(), "uzun videoda taşma (parçalı tarama gerekli)")
        except Exception as e:
            raise AssertionError(f"uzun video: {e}")
    R.test("3000 kare, α = 0.2: taşma yok", big)

    def m():
        frames = torch.full((50, 10, 10), 100.0, dtype=torch.float64) + torch.randn(50, 10, 10, generator=g, dtype=torch.float64) * 3
        frames[30:, 4:6, 4:6] = 200
        mm = need(ns, "masks")(frames, 0.05, 3.0)
        ensure(bool(mm[30, 4:6, 4:6].all()), "nesne ön plan")
        ensure(mm[10:30].double().mean() < 0.05, "durağan sahne çoğunlukla arka plan")
    R.test("masks", m)
    if "ema_scan" not in ns:
        return
    V = torch.rand(200, 240, 320, dtype=torch.float64) * 255
    t1 = bench(lambda: ref(V, 0.05), repeat=1, number=1)
    t2 = bench(lambda: ns["ema_var_scan"](V, 0.05), repeat=3)
    R.table("200 × 240 × 320, B ve V", ["yöntem", "süre"], [["kare döngüsü", fmt_time(t1)], ["parçalı tarama (L = 64)", fmt_time(t2)]],
            note="CPU'da fark küçük olabilir (ikisi de bellek sınırlı); GPU'da kare döngüsü 200 küçük çekirdek başlatır, tarama 4 büyük çekirdek.")
