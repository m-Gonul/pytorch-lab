import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def acc_one(Xtr, ytr, Xva, yva, C, a):
    N = F.one_hot(ytr, C).to(Xtr.dtype).T @ Xtr
    lt = torch.log(N + a) - torch.log(N.sum(1, keepdim=True) + a * Xtr.shape[1])
    lp = torch.log(torch.bincount(ytr, minlength=C).to(Xtr.dtype) / len(ytr))
    return ((Xva @ lt.T + lp).argmax(1) == yva).to(Xtr.dtype).mean()


def data(g, n, V, C):
    y = torch.randint(0, C, (n,), generator=g)
    P = torch.softmax(torch.randn(C, V, generator=g) * 1.5, 1)
    X = torch.stack([torch.multinomial(P[c], 15, replacement=True, generator=g).bincount(minlength=V) for c in y]).double()
    return X, y


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    Xtr, ytr = data(g, 300, 80, 3)
    Xva, yva = data(g, 200, 80, 3)
    alphas = torch.logspace(-4, 2, 20, dtype=torch.float64)

    def eq():
        A = need(ns, "alpha_sweep")(Xtr, ytr, Xva, yva, 3, alphas)
        ensure(tuple(A.shape) == (20,), f"shape {tuple(A.shape)}")
        ref = torch.stack([acc_one(Xtr, ytr, Xva, yva, 3, float(a)) for a in alphas])
        close(A, ref)
        k = int(A.argmax())
        return f"en iyi α = {float(alphas[k]):.3g}, doğrulama doğruluğu {float(A[k]):.3f}"
    R.test("20 α: tek tek eğitimle aynı", eq)
    if "alpha_sweep" not in ns:
        return
    Xb, yb = data(g, 2000, 500, 5)
    al = torch.logspace(-4, 2, 20, dtype=torch.float64)
    t_loop = bench(lambda: [acc_one(Xb, yb, Xb, yb, 5, float(a)) for a in al], repeat=3)
    t_vec = bench(lambda: ns["alpha_sweep"](Xb, yb, Xb, yb, 5, al), repeat=3)
    R.table("20 α, n = 2000, V = 500", ["yöntem", "süre", "döngüye göre"],
            [["for α: eğit + tahmin", fmt_time(t_loop), "1×"], ["senin alpha_sweep (einsum)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]],
            note="Sayım matrisi N tek kez hesaplanır; α yalnız logaritmaya giren bir sabit.")
