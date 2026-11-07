import math
import random
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def loop_ucb(probs, T, n_runs):
    K = len(probs)
    for k in range(n_runs):
        rng = random.Random(k)
        n, s = [0] * K, [0.0] * K
        for t in range(1, T + 1):
            a = t - 1 if t <= K else max(range(K), key=lambda i: s[i] / n[i] + math.sqrt(2 * math.log(t) / n[i]))
            r = 1 if rng.random() < probs[a] else 0
            n[a] += 1
            s[a] += r


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: max_loops(src, 1, func="vec_bandit"))
    probs = [0.9, 0.8, 0.7, 0.5, 0.3]

    def lr():
        close(need(ns, "lai_robbins")([0.5, 0.4]), 0.1 / (0.4 * math.log(0.8) + 0.6 * math.log(0.6 / 0.5)), what="Lai–Robbins sabiti")
    R_.test("lai_robbins", lr)

    def v():
        rows, res = [], {}
        for m in ("eps", "ucb", "ts"):
            r = need(ns, "vec_bandit")(probs, 3000, 2000, m, 0.1, 1)
            res[m] = r
            rows.append([m, f"{float(r[299]):.1f}", f"{float(r[-1]):.1f}", f"{float(r[-1]) / math.log(3000):.1f}"])
        lb = ns["lai_robbins"](probs)
        rows.append(["Lai–Robbins alt sınır sabiti", "", f"{lb * math.log(3000):.1f}", f"{lb:.1f}"])
        ensure(float(res["ts"][-1]) < float(res["ucb"][-1]) and float(res["ts"][-1]) < float(res["eps"][-1]), "Thompson en iyi")
        ensure(abs(float(res["eps"][-1]) - float(res["eps"][1999]) * 1.5) < 0.2 * float(res["eps"][-1]), "ε: doğrusal")
        t_vec = bench(lambda: ns["vec_bandit"](probs, 3000, 2000, "ucb"), repeat=1, number=1)
        t_loop = bench(lambda: loop_ucb(probs, 3000, 20), repeat=1, number=1) * 100
        rows.append(["süre: 2000 deney paralel (UCB)", fmt_time(t_vec), "", ""])
        rows.append(["süre: deney döngüsü (tahmini)", fmt_time(t_loop), "", ""])
        R_.table("2000 deney × 3000 adım", ["yöntem", "pişmanlık t = 300", "t = 3000", "/ ln t"], rows,
                 note="Asimptotik alt sınır yalnız büyük t'de geçerlidir; Thompson'ın eğimi ona en yakın olandır (Chapelle & Li 2011).")
    R_.test("vec_bandit", v)
