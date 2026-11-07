import torch
from labcheck import close, need, ensure


def spam(g, n=400, V=30):
    y = torch.randint(0, 2, (n,), generator=g)
    p0 = torch.softmax(torch.randn(V, generator=g), 0)
    p1 = torch.softmax(torch.randn(V, generator=g), 0)
    X = torch.stack([torch.multinomial(p1 if c else p0, 20, replacement=True, generator=g).bincount(minlength=V) for c in y]).double()
    return X, y


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    X, y = spam(g)

    def fit():
        lp, lt = need(ns, "fit_mnb")(X, y, 2, alpha=0.5)
        ensure(tuple(lt.shape) == (2, 30), f"log_theta shape {tuple(lt.shape)}")
        for c in range(2):
            N = X[y == c].sum(0)
            close(lt[c], torch.log((N + 0.5) / (N.sum() + 0.5 * 30)), what=f"sınıf {c}")
            close(lp[c].exp(), (y == c).double().mean(), what="prior")
        close(lt.exp().sum(1), torch.ones(2, dtype=lt.dtype), what="θ satır toplamı")
    R.test("fit_mnb: Laplace'lı θ ve prior", fit)

    def pred():
        m = need(ns, "fit_mnb")(X, y, 2)
        L = need(ns, "predict_mnb")(m, X)
        close(L.exp().sum(1), torch.ones(400, dtype=L.dtype), what="normalize")
        acc = (need(ns, "predict_mnb")(m, X).argmax(1) == y).double().mean().item()
        ensure(acc > 0.85, f"eğitim doğruluğu {acc:.2f}")
        return f"eğitim doğruluğu {acc:.3f}"
    R.test("predict_mnb normalize, doğruluk > 0.85", pred)
