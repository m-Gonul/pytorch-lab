import time
import torch
from labcheck import need, ensure, fmt_time


def _boxes(g, n):
    c = torch.rand(n, 2, generator=g) * 600
    wh = torch.rand(n, 2, generator=g) * 60 + 10
    return torch.cat([c - wh / 2, c + wh / 2], 1)


def check(ns, R_):
    g = torch.Generator().manual_seed(0)

    def t_eq():
        b, s = _boxes(g, 300), torch.rand(300, generator=g)
        lab = torch.randint(0, 4, (300,), generator=g)
        keep = need(ns, "nms_loop")(b, s, 0.5)
        ensure(set(need(ns, "fast_nms")(b, s, 0.5).tolist()) <= set(keep.tolist()), "Fast NMS ⊆ açgözlü")
        ref = []
        for c in range(4):
            idx = torch.nonzero(lab == c)[:, 0]
            ref += idx[ns["nms_loop"](b[idx], s[idx], 0.5)].tolist()
        ensure(sorted(need(ns, "batched_nms")(b, s, lab, 0.5).tolist()) == sorted(ref), "ofset hilesi = sınıf başına NMS")
    R_.test("eşdeğerlikler", t_eq)

    def t_bench():
        rows = []
        for n in (500, 2000, 5000):
            b, s = _boxes(g, n), torch.rand(n, generator=g)
            r = [n]
            for f in (ns["nms_loop"], ns["fast_nms"]):
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    k = f(b, s, 0.5)
                    best = min(best, time.perf_counter() - t0)
                r += [fmt_time(best), len(k)]
            rows.append(r)
        R_.table("NMS süreleri (CPU, 600 × 600 sahnede rastgele kutular)", ["N", "açgözlü", "tutulan", "Fast NMS", "tutulan"], rows,
                 note="Açgözlü NMS tutulan kutu sayısı kadar sıralı adım atar (her adım küçük bir çekirdek çağrısı); Fast NMS tek bir N × N matris işlemidir: GPU'da paralel, ama bellek O(N²) ve biraz daha fazla kutu siler. Gerçek kütüphaneler (torchvision.ops.nms) açgözlü algoritmayı tek bir C++/CUDA çekirdeğinde bit maskeleriyle yapar.")
    R_.test("ölçüm", t_bench)
