import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from labcheck import close, need, ensure


def check(ns, R):
    def cr():
        D, F = need(ns, "cascade_rates")([(0.99, 0.3)] * 10)
        close(D, 0.99 ** 10, what="D"); close(F, 0.3 ** 10, what="F")
        ensure(need(ns, "stages_needed")(0.3, 1e-6) == 12 and ns["stages_needed"](0.5, 1e-3) == 10, "aşama sayısı")
    R.test("kaskad aritmetiği", cr)

    def rc():
        rng = np.random.default_rng(0)
        for _ in range(5):
            y = rng.integers(0, 2, 300); s = np.round(rng.normal(y * 1.2, 1.0), 1)
            fpr, tpr, auc = need(ns, "roc")(s, y)
            close(auc, roc_auc_score(y, s), atol=1e-12, what="AUC")
            f2, t2, _ = roc_curve(y, s, drop_intermediate=False)
            close(np.array(fpr), f2, atol=1e-12, what="fpr"); close(np.array(tpr), t2, atol=1e-12, what="tpr")
    R.test("roc = sklearn", rc)
