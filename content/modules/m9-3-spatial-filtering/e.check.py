import torch
import torch.nn.functional as F
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("box_cumsum döngüsüz", lambda: max_loops(src, 0, func="box_cumsum"))
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 96, 80, generator=g)

    def eq():
        for k in (1, 3, 11, 31):
            ref = F.avg_pool2d(x[:, None], k, stride=1, padding=k // 2, count_include_pad=True)[:, 0]
            close(need(ns, "box_cumsum")(x, k), ref, atol=1e-5, what=f"k = {k}")
    R.test("box_cumsum = avg_pool2d (count_include_pad)", eq)
    if "box_cumsum" not in ns:
        return
    X = torch.rand(8, 512, 512, generator=g)
    rows = []
    for k in (3, 11, 21, 41):
        w2 = torch.ones(1, 1, k, k) / (k * k)
        w1 = torch.ones(1, 1, 1, k) / k
        t_full = bench(lambda: F.conv2d(X[:, None], w2, padding=k // 2), repeat=3)
        t_sep = bench(lambda: F.conv2d(F.conv2d(X[:, None], w1, padding=(0, k // 2)), w1.transpose(2, 3), padding=(k // 2, 0)), repeat=3)
        t_cs = bench(lambda: ns["box_cumsum"](X, k), repeat=3)
        rows.append([str(k), fmt_time(t_full), fmt_time(t_sep), fmt_time(t_cs)])
    R.table("8 × 512 × 512 kutu filtresi", ["k", "k × k conv2d", "separable", "cumsum (senin)"], rows,
            note="Doğrudan konvolüsyon k² ile, separable k ile büyür; cumsum yöntemi k'den bağımsızdır.")
