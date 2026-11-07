import math
import torch
import torch.nn as nn


def make_data(fn, n, seed):
    g = torch.Generator().manual_seed(seed)
    if fn == "f1":
        X = torch.rand(n, 1, generator=g) * 2 - 1
        Y = torch.sin(3 * math.pi * X) / (1 + X ** 2)
    elif fn == "f2":
        X = torch.rand(n, 2, generator=g) * 2 - 1
        Y = torch.exp(-(X ** 2).sum(1, keepdim=True) / 0.1)
    else:
        X = torch.rand(n, 3, generator=g) - 0.5
        r = X[:, 0] ** 2 + X[:, 1] ** 2
        Y = torch.stack([torch.exp(-(r - 2 * X[:, 2]) / 0.1), torch.exp(-(r + 2 * X[:, 2]) / 0.1)], 1)
    return X, Y


def mlp(d_in, d_out, depth, width=64):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def fit(model, X, Y, steps=2000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def depth_table(fns, depths, n=2000):
    out = {}
    for fn in fns:
        X, Y = make_data(fn, n, 0)
        Xt, Yt = make_data(fn, 2000, 1)
        for dep in depths:
            torch.manual_seed(0)
            m = mlp(X.shape[1], Y.shape[1], dep)
            fit(m, X, Y)
            with torch.no_grad():
                out[(fn, dep)] = float(((m(Xt) - Yt) ** 2).mean())
    return out


def penultimate_separability(model, X, y):
    with torch.no_grad():
        H = model[:-1](X)
    w = torch.zeros(H.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(500):
        loss = nn.functional.binary_cross_entropy_with_logits(H @ w + b, y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((((H @ w + b) > 0).long() == y).float().mean())
