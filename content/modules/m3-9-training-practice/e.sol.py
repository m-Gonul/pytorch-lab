import math
import torch
import torch.nn as nn
from sklearn.datasets import load_digits


def _data():
    X, y = load_digits(return_X_y=True)
    return torch.tensor(X / 16.0, dtype=torch.float32), torch.tensor(y)


def _folds(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    fs = torch.tensor_split(idx, k)
    return [(torch.cat([fs[j] for j in range(k) if j != i]), fs[i]) for i in range(k)]


def batched_cv(widths, k=5, epochs=60, seed=0):
    X, y = _data()
    folds = _folds(len(y), k, seed)
    H = max(widths)
    cfg = [(w, f) for w in widths for f in range(k)]
    E = len(cfg)
    torch.manual_seed(seed)
    W1 = (torch.rand(E, 64, H) * 2 - 1) / 8
    b1 = torch.zeros(E, 1, H)
    W2 = (torch.rand(E, H, 10) * 2 - 1) / math.sqrt(H)
    b2 = torch.zeros(E, 1, 10)
    mask = torch.stack([(torch.arange(H) < w).float() for w, _ in cfg])[:, None, :]
    params = [p.requires_grad_() for p in (W1, b1, W2, b2)]
    opt = torch.optim.Adam(params, lr=3e-3)
    ntr = min(len(folds[f][0]) for f in range(k))
    Xtr = torch.stack([X[folds[f][0][:ntr]] for _, f in cfg])
    ytr = torch.stack([y[folds[f][0][:ntr]] for _, f in cfg])

    def fwd(Xb):
        h = torch.relu(torch.baddbmm(b1, Xb, W1)) * mask
        return torch.baddbmm(b2, h, W2)

    for _ in range(epochs):
        loss = nn.functional.cross_entropy(fwd(Xtr).reshape(-1, 10), ytr.reshape(-1)) * E
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = {}
    with torch.no_grad():
        accs = []
        for i, (w, f) in enumerate(cfg):
            va = folds[f][1]
            h = torch.relu(X[va] @ W1[i] + b1[i]) * mask[i]
            accs.append(float(((h @ W2[i] + b2[i]).argmax(1) == y[va]).float().mean()))
        for w in widths:
            a = torch.tensor([accs[i] for i, (ww, _) in enumerate(cfg) if ww == w])
            out[w] = (float(a.mean()), float(a.std()))
    return out


def loop_cv(widths, k=5, epochs=60, seed=0):
    X, y = _data()
    folds = _folds(len(y), k, seed)
    out = {}
    for w in widths:
        accs = []
        for f in range(k):
            tr, va = folds[f]
            torch.manual_seed(seed)
            m = nn.Sequential(nn.Linear(64, w), nn.ReLU(), nn.Linear(w, 10))
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for _ in range(epochs):
                loss = nn.functional.cross_entropy(m(X[tr]), y[tr])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                accs.append(float((m(X[va]).argmax(1) == y[va]).float().mean()))
        a = torch.tensor(accs)
        out[w] = (float(a.mean()), float(a.std()))
    return out
