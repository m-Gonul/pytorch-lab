import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)).double()
    x = torch.randn(6, 4, dtype=torch.float64); y = torch.randint(0, 3, (6,))

    def psg():
        G = need(ns, "per_sample_grads")(model, x, y)
        ensure(set(G) == {k for k, _ in model.named_parameters()}, "parametre adlarıyla sözlük")
        for i in range(6):
            model.zero_grad()
            F.cross_entropy(model(x[i:i + 1]), y[i:i + 1]).backward()
            for k, p in model.named_parameters():
                close(G[k][i], p.grad, what=f"örnek {i}, {k}")
    R.test("per_sample_grads = örnek başı backward", psg)

    def jac():
        W = torch.randn(3, 5, dtype=torch.float64)
        f = lambda v: torch.tanh(W @ v) * v[:3]
        v = torch.randn(5, dtype=torch.float64)
        Jr, Jf = need(ns, "jacobians")(f, v)
        ref = torch.autograd.functional.jacobian(f, v)
        close(Jr, ref, what="jacrev"); close(Jf, ref, what="jacfwd")
        ensure(tuple(Jr.shape) == (3, 5), "Jacobian (3, 5)")
    R.test("jacrev = jacfwd = autograd.functional.jacobian", jac)

    def hv():
        A = torch.randn(6, 6, dtype=torch.float64)
        f = lambda v: (torch.sin(A @ v) ** 2).sum() + (v ** 4).sum()
        x0, v = torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
        H = torch.autograd.functional.hessian(f, x0)
        close(need(ns, "hvp")(f, x0, v), H @ v)
    R.test("hvp = H v", hv)
