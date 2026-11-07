import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def sbm(seed):
    g = torch.Generator().manual_seed(seed)
    C, n = 3, 40
    N = C * n
    y = torch.arange(C).repeat_interleave(n)
    P = torch.where(y[:, None] == y[None, :], 0.15, 0.01)
    A = (torch.rand(N, N, generator=g) < P).float().triu(1)
    A = A + A.T
    ei = A.nonzero().T
    X = torch.cat([F.one_hot(y, C).float() + 1.5 * torch.randn(N, C, generator=g), torch.randn(N, 13, generator=g)], 1)
    tr = torch.zeros(N, dtype=torch.bool)
    for c in range(C):
        tr[(y == c).nonzero()[:5, 0]] = True
    return X, ei, y, tr, A


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.l1 = nn.Linear(i, h); self.l2 = nn.Linear(h, o)

    def forward(self, x, ei):
        return self.l2(F.dropout(F.relu(self.l1(x)), 0.5, self.training))


def check(ns, R):
    X, ei, y, tr, A = sbm(0)
    N = X.shape[0]

    def prop():
        At = A + torch.eye(N); d = At.sum(1)
        Ah = At / (d[:, None] * d[None, :]).sqrt()
        x = torch.randn(N, 4, dtype=torch.float64)
        close(need(ns, "gcn_propagate")(x, ei, N), Ah.double() @ x)
    R.test("gcn_propagate = Â x", prop)

    def grad():
        x = torch.randn(N, 3, requires_grad=True)
        need(ns, "gcn_propagate")(x, ei, N).sum().backward()
        ensure(x.grad is not None and x.grad.abs().sum() > 0, "x'e gradyan akmalı")
    R.test("gradyan akıyor", grad)

    def train():
        accs = {}
        for name, M in (("GCN", need(ns, "GCN", "sınıf")), ("MLP", MLP)):
            torch.manual_seed(0)
            m = M(16, 16, 3)
            m = need(ns, "train_node_clf")(m, X, ei, y, tr) or m
            m.eval()
            with torch.no_grad():
                accs[name] = (m(X, ei).argmax(1) == y)[~tr].float().mean().item()
        ensure(accs["GCN"] > 0.8, f"GCN test doğruluğu {accs['GCN']:.3f}")
        ensure(accs["GCN"] > accs["MLP"] + 0.2, "GCN, MLP'yi açıkça geçmeli")
        return f"15 etiketle test doğruluğu: GCN {accs['GCN']:.3f}, MLP {accs['MLP']:.3f}"
    R.test("az etiketle düğüm sınıflandırma", train)
