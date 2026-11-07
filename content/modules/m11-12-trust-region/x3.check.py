import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(3)
        S, A, gamma = 6, 3, 0.9
        P = rng.random((A, S, S)) ** 2
        P /= P.sum(-1, keepdims=True)
        R = rng.random((S, A))
        d0 = np.full(S, 1 / S)
        pi = rng.dirichlet(np.ones(A), S)
        pi2 = rng.dirichlet(np.ones(A), S)
        e0 = need(ns, "eta")(P, R, d0, pi, gamma)
        close(need(ns, "surrogate_L")(P, R, d0, pi, pi, gamma), e0, what="L_π(π) = η(π)")
        rows, viol = [], 0
        for a in (0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0):
            pn = (1 - a) * pi + a * pi2
            e, Lv, b = ns["eta"](P, R, d0, pn, gamma), ns["surrogate_L"](P, R, d0, pi, pn, gamma), need(ns, "bound")(P, R, d0, pi, pn, gamma)
            viol += int(b > e + 1e-12)
            rows.append([a, f"{e:.5f}", f"{Lv:.5f}", f"{b:.4f}", f"{need(ns, 'kl_max')(pi, pn):.5f}"])
        ensure(viol == 0, "η ≥ sınır her karışımda")
        h = 1e-6
        de = (ns["eta"](P, R, d0, (1 - h) * pi + h * pi2, gamma) - e0) / h
        dL = (ns["surrogate_L"](P, R, d0, pi, (1 - h) * pi + h * pi2, gamma) - e0) / h
        close(de, dL, rtol=1e-4, what="L ve η birinci mertebede aynı")
        R_.table("π̃ = (1 − α)π + απ′ (S = 6, A = 3, γ = 0.9)", ["α", "η(π̃)", "L_π(π̃)", "alt sınır", "KL^max"], rows,
                 note="Sınır doğru ama çok kötümser: (1 − γ)⁻² = 100 çarpanı yüzünden yalnız minik adımlarda anlamlı. TRPO bu yüzden ceza yerine KL kısıtı kullanır ve δ'yı deneysel seçer.")
    R_.test("monoton iyileşme sınırı", t)
