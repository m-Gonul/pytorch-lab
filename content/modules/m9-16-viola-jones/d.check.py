import numpy as np
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    X = data.lfw_subset()
    y = np.r_[np.ones(100), np.zeros(100)]
    Rt, St, At = ns["feature_table"](3000, 0)
    F = torch.tensor(ns["normalized_features"](X, Rt, St, At))
    rng = np.random.default_rng(0)
    perm = rng.permutation(200); tr, te = perm[:150], perm[150:]

    def bs():
        w = torch.tensor(rng.random(150)); w /= w.sum()
        yt = torch.tensor(y[tr])
        err, thr, pol = need(ns, "best_stumps")(F[tr][:, :50], yt, w)
        for j in range(0, 50, 7):
            f = F[tr][:, j].numpy()
            best = min((w.numpy() * (((p * f) <= (p * t)).astype(float) != y[tr])).sum() for t in np.unique(f) for p in (1, -1))
            close(float(err[j]), best, atol=1e-12, what=f"öznitelik {j}")
    R.test("best_stumps = kaba kuvvet", bs)

    def ab():
        H = need(ns, "adaboost")(F[tr], torch.tensor(y[tr]), 30)
        ensure(len(H) == 30, "30 zayıf sınıflandırıcı")
        rows = []
        for k in (1, 3, 10, 30):
            s, half = need(ns, "strong_score")(F[te], H, k)
            acc = float(((s >= half).double().numpy() == y[te]).mean())
            rows.append([k, f"{acc:.2f}"])
        ensure(float(rows[-1][1]) >= 0.9, f"30 öznitelikle test doğruluğu {rows[-1][1]}")
        j = H[0][0]
        r = Rt[j][:, :4]
        R.table("lfw_subset (150 eğitim / 50 test), 3000 rastgele öznitelik", ["öznitelik sayısı", "test doğruluğu"], rows,
                note=f"İlk seçilen öznitelik dikdörtgenleri (satır, sütun): {[tuple(x) for x in r if x[2] > 0]} — genelde göz bandı / yanak karşıtlığı.")
    R.test("AdaBoost", ab)
