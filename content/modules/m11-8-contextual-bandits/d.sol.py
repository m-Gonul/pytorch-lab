import torch
import torch.nn as nn


def xor_means(x):
    p = (x[:, 0] * x[:, 1] > 0).double()
    return torch.stack([torch.full_like(p, 0.6), p, 1 - p], 1)


class BLR:
    def __init__(self, K, d, lam=1.0, sigma=0.5):
        self.K, self.d, self.lam, self.s2 = K, d, lam, sigma ** 2
        self.P = torch.eye(d, dtype=torch.float64).repeat(K, 1, 1) / lam ** 2
        self.b = torch.zeros(K, d, dtype=torch.float64)

    def update(self, a, z, r):
        self.P[a] += torch.outer(z, z) / self.s2
        self.b[a] += z * r / self.s2

    def refit(self, Z, A, R):
        oh = torch.nn.functional.one_hot(A, self.K).double()
        self.P = torch.eye(self.d, dtype=torch.float64) / self.lam ** 2 + torch.einsum("nk,ni,nj->kij", oh, Z, Z) / self.s2
        self.b = torch.einsum("nk,ni,n->ki", oh, Z, R) / self.s2

    def sample(self, z, g):
        S = torch.linalg.inv(self.P)
        S = (S + S.transpose(1, 2)) / 2
        mu = (S @ self.b[:, :, None]).squeeze(-1)
        w = mu + (torch.linalg.cholesky(S) @ torch.randn(self.K, self.d, 1, generator=g, dtype=torch.float64)).squeeze(-1)
        return w @ z


def _env(T, g):
    X = torch.rand(T, 2, generator=g, dtype=torch.float64) * 2 - 1
    return X, xor_means(X)


def run_lints(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    X, M = _env(T, g)
    blr = BLR(3, 3)
    reg, out = 0.0, []
    for t in range(T):
        z = torch.cat([X[t], torch.ones(1, dtype=torch.float64)])
        a = int(blr.sample(z, g).argmax())
        r = M[t, a] + 0.1 * torch.randn((), generator=g, dtype=torch.float64)
        blr.update(a, z, r)
        reg += float(M[t].max() - M[t, a])
        out.append(reg)
    return out


def run_neural_linear(T, seed=0, retrain=50, iters=300):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X, M = _env(T, g)
    body = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU()).double()
    head = nn.Linear(50, 3).double()
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-2)
    blr = BLR(3, 51)
    A, R = [], []
    reg, out = 0.0, []
    for t in range(T):
        with torch.no_grad():
            z = torch.cat([body(X[t:t + 1])[0], torch.ones(1, dtype=torch.float64)])
        a = int(blr.sample(z, g).argmax())
        r = M[t, a] + 0.1 * torch.randn((), generator=g, dtype=torch.float64)
        A.append(a)
        R.append(float(r))
        blr.update(a, z, r)
        reg += float(M[t].max() - M[t, a])
        out.append(reg)
        if (t + 1) % retrain == 0:
            Xa, Aa, Ra = X[:t + 1], torch.tensor(A), torch.tensor(R, dtype=torch.float64)
            for _ in range(iters):
                idx = torch.randint(0, t + 1, (64,), generator=g)
                pred = head(body(Xa[idx])).gather(1, Aa[idx, None]).squeeze(1)
                loss = ((pred - Ra[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                Z = torch.cat([body(Xa), torch.ones(t + 1, 1, dtype=torch.float64)], 1)
            blr.refit(Z, Aa, Ra)
    return out


print(run_lints(300)[-1])
