import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="fit_model"), max_loops(src, 1, func="plan"), max_loops(src, 0, func="policy_value")))
    rng = np.random.default_rng(1)
    nS, nA, gamma = 15, 3, 0.9
    P = rng.random((nA, nS, nS)) ** 4
    P /= P.sum(-1, keepdims=True)
    R = rng.random((nS, nA))

    def data(n, seed):
        r = np.random.default_rng(seed)
        S_ = r.integers(0, nS, n)
        A_ = r.integers(0, nA, n)
        cdf = P[A_, S_].cumsum(1)
        S2_ = (r.random(n)[:, None] > cdf).sum(1).clip(max=nS - 1)
        return S_, A_, R[S_, A_], S2_

    def fm():
        S_, A_, R__, S2_ = data(200000, 0)
        Ph, Rh = need(ns, "fit_model")(S_, A_, R__, S2_, nS, nA)
        close(Ph.sum(-1), np.ones((nA, nS)), what="satır toplamları")
        ensure(np.abs(Ph - P).max() < 0.02 and np.abs(Rh - R).max() < 1e-12, "çok veride gerçek modele yakınsar")
        Ph2, _ = ns["fit_model"](S_[:3], A_[:3], R__[:3], S2_[:3], nS, nA)
        ensure(np.allclose(Ph2.sum(-1), 1), "görülmemiş çiftlerde de geçerli dağılım")
    R_.test("fit_model", fm)

    def cmp():
        Qs = need(ns, "plan")(P, R, gamma)
        Vs = need(ns, "policy_value")(P, R, Qs.argmax(1), gamma)
        rows = []
        res = {}
        for n in (300, 1000, 3000, 10000):
            vm = vq = vq10 = 0
            for seed in range(5):
                S_, A_, R__, S2_ = data(n, seed + 10)
                Ph, Rh = ns["fit_model"](S_, A_, R__, S2_, nS, nA)
                vm += ns["policy_value"](P, R, ns["plan"](Ph, Rh, gamma).argmax(1), gamma).mean()
                vq += ns["policy_value"](P, R, need(ns, "q_learning_replay")(S_, A_, R__, S2_, nS, nA, gamma, 1).argmax(1), gamma).mean()
                vq10 += ns["policy_value"](P, R, ns["q_learning_replay"](S_, A_, R__, S2_, nS, nA, gamma, 10).argmax(1), gamma).mean()
            res[n] = (vm / 5, vq / 5, vq10 / 5)
            rows.append([n, f"{vm / 5:.3f}", f"{vq / 5:.3f}", f"{vq10 / 5:.3f}"])
        ensure(res[1000][0] > res[1000][1] and Vs.mean() - res[10000][0] < 0.05, "modelli planlama örnek verimli")
        R_.table(f"Açgözlü politikanın gerçek ortalama değeri (optimal {Vs.mean():.3f}), 5 tekrar", ["geçiş sayısı", "model + planlama", "Q-öğrenme (1 geçiş)", "Q-öğrenme (10 tekrar)"], rows,
                 note="Aynı veri: model her geçişi tüm planlama boyunca kullanır; Q-öğrenme her örnekten tek küçük adım alır. Tekrar (replay) aradaki farkı kapatır — Dyna ile tamponun yakınlığı.")
    R_.test("model tabanlı vs modelsiz", cmp)
