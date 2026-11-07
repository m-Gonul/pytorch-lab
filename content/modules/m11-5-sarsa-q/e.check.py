import random
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def loop_q(n_envs, n_steps, alpha=0.5, eps=0.1):
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for k in range(n_envs):
        rng = random.Random(k)
        Q = [[0.0] * 4 for _ in range(48)]
        s = 36
        for _ in range(n_steps):
            q = Q[s]
            a = rng.randrange(4) if rng.random() < eps else q.index(max(q))
            r_, c_ = divmod(s, 12)
            r_ = min(max(r_ + moves[a][0], 0), 3); c_ = min(max(c_ + moves[a][1], 0), 11)
            if r_ == 3 and 1 <= c_ <= 10:
                s2, r = 36, -100.0
            else:
                s2, r = r_ * 12 + c_, -1.0
            Q[s][a] += alpha * (r + (0.0 if s2 == 47 else max(Q[s2])) - Q[s][a])
            s = 36 if s2 == 47 else s2


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: (max_loops(src, 1, func="batched_control"), max_loops(src, 1, func="greedy_returns")))

    def st():
        s2, r, d = need(ns, "step")(torch.tensor([36, 36, 35, 0]), torch.tensor([1, 0, 2, 0]))
        ensure(s2.tolist() == [36, 24, 47, 0] and r.tolist() == [-100, -1, -1, -1] and d.tolist() == [False, False, True, False], "vektörel ortam")
    R_.test("step", st)

    def ctl():
        rows = []
        res = {}
        for m in ("sarsa", "expected", "q"):
            Q, rew = need(ns, "batched_control")(1000, 8000, 0.5, 0.1, m, 2)
            G = need(ns, "greedy_returns")(Q)
            res[m] = (float(rew[-3000:].mean()), G)
            vals, cnt = torch.unique(G, return_counts=True)
            rows.append([m, f"{res[m][0]:.3f}", ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(vals, cnt))])
        ensure(int((res["q"][1] == -13).sum()) > 950, "Q-öğrenme: kenar yolu")
        ensure(int(((res["expected"][1] <= -15) & (res["expected"][1] >= -19)).sum()) > 900, "beklenen SARSA: güvenli yol")
        ensure(res["sarsa"][0] > res["q"][0] and res["expected"][0] > res["q"][0], "politika üstü yöntemler eğitimde daha çok ödül toplar")
        t1 = bench(lambda: loop_q(20, 8000), repeat=1, number=1) * 50
        t2 = bench(lambda: ns["batched_control"](1000, 8000, 0.5, 0.1, "q"), repeat=1, number=1)
        rows.append(["süre: ortam döngüsü (saf Python, tahmini)", fmt_time(t1), ""])
        rows.append(["süre: 1000 paralel ortam", fmt_time(t2), ""])
        R_.table("1000 ortam × 8000 adım, α = 0.5, ε = 0.1", ["yöntem", "son 3000 adımda ort. ödül/adım", "açgözlü getiri: ortam sayısı"], rows,
                 note="Beklenen SARSA, SARSA'nın a' örneklemesinden gelen varyansı yok eder: büyük α'da bile tüm ortamlarda aynı güvenli yolu bulur.")
    R_.test("SARSA, beklenen SARSA, Q-öğrenme", ctl)
