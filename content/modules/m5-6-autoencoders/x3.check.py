import numpy as np
from sklearn.datasets import load_digits
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("fit_pca", "recon_error", "roc_auc")])

    def t():
        close(need(ns, "roc_auc")([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, what="sklearn örneği")
        close(ns["roc_auc"]([1, 1, 1, 1], [0, 1, 0, 1]), 0.5, what="eşit skorlar")
        X, y = load_digits(return_X_y=True)
        X = X / 16
        normal = y <= 4
        rng = np.random.default_rng(0)
        idx = rng.permutation(np.flatnonzero(normal))
        tr = idx[:600]
        test = np.concatenate([idx[600:], np.flatnonzero(~normal)])
        rows = []
        aucs = {}
        for k in (1, 3, 8, 20, 40, 64):
            m, C = need(ns, "fit_pca")(X[tr], k)
            e = need(ns, "recon_error")(X[test], m, C)
            aucs[k] = ns["roc_auc"](e, ~normal[test])
            rows.append([k, f"{aucs[k]:.3f}"])
        ensure(aucs[20] > 0.85 and aucs[1] < aucs[20] and aucs[40] < aucs[20], f"AUC {aucs}")
        R_.table("0–4 rakamlarıyla eğitilen doğrusal AE (PCA), 5–9 'anomali'", ["kod boyutu k", "ROC AUC"], rows,
                 note="Yalnız normal veriyle eğitilmiş AE onları iyi yeniden kurar, görmediği yapıdaki girdileri kötü: hata anomali skorudur. Kod çok küçükse normal örnekler de kötü kurulur, kod büyüdükçe anomaliler de iyi kurulmaya başlar; k = d'de (darboğaz yok) her şey mükemmel kurulur, 'hata' yalnız kayan nokta gürültüsüdür ve AUC anlamsızlaşır — darboğaz, tespiti mümkün kılan kısıttır.")
    R_.test("anomali tespiti", t)
