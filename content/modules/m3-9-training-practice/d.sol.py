import torch
import torch.nn as nn
from sklearn.datasets import load_digits


def _data():
    X, y = load_digits(return_X_y=True)
    return torch.tensor(X / 16.0, dtype=torch.float32), torch.tensor(y)


def _mlp(sizes):
    L = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        L += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*L[:-1])


def _train(m, X, y, epochs, lr=1e-3, bs=32, params=None, Xv=None, yv=None, g=None):
    opt = torch.optim.Adam(params if params is not None else m.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        p = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), bs):
            idx = p[i:i + bs]
            loss = nn.functional.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xv is not None:
            with torch.no_grad():
                hist.append((float(nn.functional.cross_entropy(m(X), y)), float(nn.functional.cross_entropy(m(Xv), yv)), float((m(Xv).argmax(1) == yv).float().mean())))
    return hist


def capacity_runs(epochs=40, seed=0):
    X, y = _data()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    tr, va = perm[:1000], perm[1000:]
    out = {}
    for sizes in ([64, 300, 100, 10], [64, 400, 500, 300, 100, 10], [64, 600, 700, 500, 400, 300, 100, 10]):
        torch.manual_seed(seed)
        out["-".join(map(str, sizes))] = _train(_mlp(sizes), X[tr], y[tr], epochs, Xv=X[va], yv=y[va], g=g)
    return out


def _world(n, g, A1, A2):
    z = torch.randn(n, 4, generator=g)
    return torch.sin(2 * torch.tanh(2 * z @ A1) @ A2) + 0.05 * torch.randn(n, 64, generator=g), z


def _fit(m, X, Y, epochs, lr=1e-3, params=None, mse=False, bs=32, g=None):
    opt = torch.optim.Adam(params if params is not None else m.parameters(), lr=lr)
    for _ in range(epochs):
        p = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            idx = p[i:i + bs]
            out = m(X[idx])
            loss = ((out - Y[idx]) ** 2).mean() if mse else nn.functional.cross_entropy(out, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def transfer(n_target=100, seed=0):
    import copy
    g = torch.Generator().manual_seed(seed)
    A1 = torch.randn(4, 32, generator=g)
    A2 = torch.randn(32, 64, generator=g) / 4
    Xs, Zs = _world(5000, g, A1, A2)
    Xt, Zt = _world(n_target, g, A1, A2)
    Xe, Ze = _world(3000, g, A1, A2)
    yt, ye = ((Zt[:, 2] + Zt[:, 3]) > 0).long(), ((Ze[:, 2] + Ze[:, 3]) > 0).long()
    body_fn = lambda: nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU())
    torch.manual_seed(seed)
    body = body_fn()
    _fit(nn.Sequential(body, nn.Linear(32, 4)), Xs, Zs, 30, mse=True, g=g)
    rbody = body_fn()
    _fit(nn.Sequential(rbody, nn.Linear(32, 4)), Xs, torch.randint(0, 4, (5000,), generator=g), 30, g=g)
    torch.manual_seed(seed + 1)
    scratch = nn.Sequential(body_fn(), nn.Linear(32, 2))
    _fit(scratch, Xt, yt, 200, bs=8, g=g)
    frozen = nn.Sequential(copy.deepcopy(body), nn.Linear(32, 2))
    _fit(frozen, Xt, yt, 200, bs=8, params=frozen[1].parameters(), g=g)
    fine = nn.Sequential(copy.deepcopy(body), nn.Linear(32, 2))
    _fit(fine, Xt, yt, 100, bs=8, params=fine[1].parameters(), g=g)
    _fit(fine, Xt, yt, 100, lr=1e-4, bs=8, g=g)
    rnd = nn.Sequential(copy.deepcopy(rbody), nn.Linear(32, 2))
    _fit(rnd, Xt, yt, 200, bs=8, params=rnd[1].parameters(), g=g)
    with torch.no_grad():
        return {k: float((m(Xe).argmax(1) == ye).float().mean()) for k, m in (("sıfırdan", scratch), ("dondurulmuş", frozen), ("ince ayar", fine), ("ilgisiz kaynak", rnd))}
