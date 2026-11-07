import random
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def loop_td(n_runs, n_steps, alpha, seed=0):
    out = []
    for k in range(n_runs):
        rng = random.Random(seed * 1000 + k)
        V = [0.0] + [0.5] * 5 + [0.0]
        s = 3
        for _ in range(n_steps):
            s2 = s + (1 if rng.random() < 0.5 else -1)
            V[s] += alpha * ((1.0 if s2 == 6 else 0.0) + V[s2] - V[s])
            s = 3 if s2 in (0, 6) else s2
        out.append(V)
    return out


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: max_loops(src, 1, func="batched_td"))

    def t():
        V, rms = need(ns, "batched_td")(1000, 3000, 0.02, 5, 1)
        ensure(tuple(V.shape) == (1000, 7) and rms.shape[0] == 3000, "şekiller")
        close(V[:, 1:6].mean(0).numpy(), (torch.arange(1, 6, dtype=torch.float64) / 6).numpy(), atol=0.02, what="ortalama V ≈ gerçek değer")
        ensure(float(rms[-1]) < 0.06 and float(rms[-1]) < float(rms[100]), "hata azalmalı")
        ref = loop_td(200, 3000, 0.02)
        refm = torch.tensor(ref).mean(0)
        close(V[:200].mean(0).numpy(), refm.numpy(), atol=0.03, what="döngülü uygulamayla istatistiksel uyum")
        t1 = bench(lambda: loop_td(200, 3000, 0.02), repeat=1, number=1) * 5
        t2 = bench(lambda: ns["batched_td"](1000, 3000, 0.02), repeat=1, number=1)
        R_.table("1000 deney × 3000 adım", ["yöntem", "süre"], [["deney döngüsü (saf Python, tahmini)", fmt_time(t1)], ["tensör (deneyler paralel)", fmt_time(t2)]])
    R_.test("batched_td", t)
