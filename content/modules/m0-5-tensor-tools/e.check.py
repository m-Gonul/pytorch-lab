import time
import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5000, generator=g)

    def ma():
        f = need(ns, "moving_average_fast")
        for w in (1, 3, 64, 5000):
            close(f(x, w), x.unfold(0, w, 1).double().mean(1), rtol=1e-5, atol=1e-5, what=f"w = {w}")
    R.test("moving_average_fast doğru (w = 1, 3, 64, n)", ma)
    if "time_fn" not in ns or "moving_average_fast" not in ns:
        R.test("`time_fn` tanımlı", lambda: need(ns, "time_fn"))
        return
    tf, maf = ns["time_fn"], ns["moving_average_fast"]

    def tt():
        y = torch.randn(200_000)
        t = tf(torch.sort, y)
        ensure(isinstance(t, float) and t > 0, f"pozitif float bekleniyordu: {t!r}")
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); torch.sort(y); ts.append(time.perf_counter() - t0)
        ref = sorted(ts)[3]
        ensure(0.2 < t / ref < 5, f"time_fn {fmt_time(t)}, bağımsız ölçüm {fmt_time(ref)} — tutarsız")
        return f"torch.sort(200k): {fmt_time(t)} (bağımsız ölçüm {fmt_time(ref)})"
    R.test("time_fn tutarlı bir medyan veriyor", tt)

    y = torch.randn(200_000)

    def loop(v, w):
        out, s = [], float(v[:w].sum())
        out.append(s / w)
        vv = v.tolist()
        for i in range(w, len(vv)):
            s += vv[i] - vv[i - w]
            out.append(s / w)
        return out

    rows = []
    for w in (5, 500):
        k = torch.full((1, 1, w), 1.0 / w)
        meths = [("Python döngüsü (kayan toplam)", lambda w=w: loop(y, w)),
                 ("unfold(…).mean(1)", lambda w=w: y.unfold(0, w, 1).mean(1)),
                 ("cumsum farkı", lambda w=w: (lambda c: (c[w:] - c[:-w]) / w)(torch.cat([y.new_zeros(1), y.cumsum(0)]))),
                 ("F.conv1d", lambda w=w, k=k: F.conv1d(y.view(1, 1, -1), k).view(-1)),
                 ("senin moving_average_fast", lambda w=w: maf(y, w))]
        times = [(n, tf(f)) for n, f in meths]
        best = min(t for _, t in times)
        rows += [[f"w = {w}", n, fmt_time(t), "en hızlı" if t == best else f"{t / best:,.1f}×"] for n, t in times]
    R.table("Hareketli ortalama, n = 200 000", ["pencere", "yöntem", "medyan süre", "en hızlıya göre"], rows,
            note="unfold O(n·w) iş yapar, cumsum O(n): w büyüdükçe fark açılır. Ölçmeden bilinmez.")
