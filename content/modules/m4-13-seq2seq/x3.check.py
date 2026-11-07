import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        V, d = 2000, 16
        W = rng.normal(0, 0.6, (V, d))
        h = rng.normal(size=d)
        t_ = 7
        z = W @ h
        p = np.exp(z - z.max())
        p /= p.sum()
        g = need(ns, "full_grad")(h, W, t_)
        close(g, W.T @ p - W[t_], atol=1e-10, what="tam gradyan")
        eps = 1e-6
        e0 = np.zeros(d)
        e0[3] = eps
        loss = lambda hh: -(W @ hh)[t_] + np.log(np.exp(W @ hh).sum())
        close((loss(h + e0) - loss(h)) / eps, g[3], atol=1e-4, what="sayısal")
        q = (1.0 / np.arange(1, V + 1))
        q /= q.sum()
        logq = np.log(q)
        rows = []
        errs = {}
        for m in (20, 100, 500):
            for corr in (False, True):
                acc = np.zeros(d)
                R = 300
                for _ in range(R):
                    S = rng.choice(V, m, p=q)
                    S = S[S != t_]
                    acc += need(ns, "sampled_grad")(h, W, t_, S, logq, corr)
                err = np.linalg.norm(acc / R - g) / np.linalg.norm(g)
                errs[(m, corr)] = err
                rows.append([m, "evet" if corr else "hayır", f"{err:.3f}"])
        ensure(errs[(500, True)] < errs[(500, False)] and errs[(500, True)] < 0.25, "log Q düzeltmesi yanlılığı azaltır")
        R_.table("V = 2000, Zipf önerisi: 300 örneklenmiş gradyanın ortalamasının tam gradyana göreli hatası", ["örnek sayısı", "log Q düzeltmesi", "göreli hata"], rows,
                 note="Örneklenmiş softmax, tam normalizasyonu yalnız hedef + m örnek üzerinden yapar. Öneri dağılımı düzgün değilse (sık kelimeler daha çok örneklenir) logit'lerden log q çıkarmak önem örneklemesi düzeltmesidir (Jean ve ark. 2015); düzeltmesiz tahminci sık kelimelere doğru yanlıdır. m büyüdükçe her iki hata da azalır.")
    R_.test("örneklenmiş softmax", t)
