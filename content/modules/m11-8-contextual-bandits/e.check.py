import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def make(n, seed, env_seed=0, temp=0.8):
    ge = torch.Generator().manual_seed(env_seed)
    d, K = 5, 4
    theta = torch.randn(K, d, generator=ge, dtype=torch.float64)
    Wlog = temp * torch.randn(d, K, generator=ge, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    X[:, 0] = 1
    mu = torch.sigmoid(X @ theta.T)
    P = torch.softmax(X @ Wlog, 1)
    A = torch.multinomial(P, 1, generator=g).squeeze(1)
    R = (torch.rand(n, generator=g, dtype=torch.float64) < mu.gather(1, A[:, None]).squeeze(1)).double()
    return X, A, R, P.gather(1, A[:, None]).squeeze(1), mu, theta


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: (max_loops(src, 0, func="estimates"), max_loops(src, 0, func="fit_reward_model"), max_loops(src, 0, func="policy_actions")))

    def frm():
        X, A, R, p, mu, theta = make(3000, 1)
        Th = need(ns, "fit_reward_model")(X, A, R, 4, 1.0)
        for k in range(4):
            Xk = X[A == k]
            ref = torch.linalg.solve(Xk.T @ Xk + torch.eye(5, dtype=torch.float64), Xk.T @ R[A == k])
            close(Th[k].numpy(), ref.numpy(), atol=1e-9, what=f"eylem {k} sırt regresyonu")
    R_.test("fit_reward_model", frm)

    def ope():
        theta = make(10, 0)[5]
        g = torch.Generator().manual_seed(7)
        W = torch.randn(200, 4, 5, generator=g, dtype=torch.float64)
        W[:100] = theta[None] + 0.7 * W[:100]
        errs = {k: [] for k in ("DM", "IPS", "SNIPS", "DR")}
        found = {k: 0 for k in errs}
        for s in range(40):
            X, A, R, p, mu, _ = make(2000, 100 + s)
            pa = need(ns, "policy_actions")(W, X)
            ensure(tuple(pa.shape) == (200, 2000), "şekil")
            true = mu.gather(1, pa.T).T.mean(1)
            est = need(ns, "estimates")(W, X, A, R, p, ns["fit_reward_model"](X, A, R, 4, 1.0))
            for k, v in est.items():
                errs[k].append(v - true)
                found[k] += int(int(v.argmax()) == int(true.argmax()))
        stats = {}
        for k, v in errs.items():
            E = torch.stack(v)
            m, se = E.mean(0), E.std(0) / 40 ** 0.5
            stats[k] = (float(E.pow(2).mean().sqrt()), float(m.abs().mean()), float((m.abs() / se).mean()))
        ensure(stats["IPS"][2] < 1.3 and stats["DR"][2] < 1.3, f"IPS ve DR yansız: |ortalama hata| / SE ≈ 0.8 olmalı ({stats['IPS'][2]:.2f}, {stats['DR'][2]:.2f})")
        ensure(stats["DM"][2] > 2.0, "yanlış belirtilmiş DM yanlı olmalı")
        ensure(stats["DR"][0] < stats["IPS"][0] and stats["SNIPS"][0] < stats["IPS"][0], "DR ve SNIPS IPS'ten düşük RMSE")
        X, A, R, p, mu, _ = make(2000, 100)
        Th = ns["fit_reward_model"](X, A, R, 4, 1.0)
        t_vec = bench(lambda: ns["estimates"](W, X, A, R, p, Th), repeat=3, number=1)
        t_loop = bench(lambda: [ns["estimates"](W[i:i + 1], X, A, R, p, Th) for i in range(200)], repeat=1, number=1)
        rows = [[k, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.2f}", f"{found[k]}/40"] for k, v in stats.items()]
        rows.append(["süre: 200 politika tek tensör", fmt_time(t_vec), "", "", ""])
        rows.append(["süre: politika döngüsü", fmt_time(t_loop), "", "", ""])
        R_.table("200 doğrusal politika × 40 kayıt kümesi (n = 2000, softmax kayıt politikası, sigmoid ödül)", ["kestirici", "RMSE", "|ort. hata|", "|ort. hata| / SE", "en iyi politikayı buldu"], rows,
                 note="Yansız bir kestiricide |ortalama hata| / SE ≈ √(2/π) ≈ 0.8 beklenir. DM'nin doğrusal ödül modeli sigmoid gerçeği yanlış belirtir: düşük varyanslı ama yanlı. IPS yansızdır, küçük p'lerde varyansı büyür. DR, DM'yi IPS ile düzeltir: yansız ve IPS'ten düşük varyanslı.")
    R_.test("DM, IPS, SNIPS, DR", ope)
