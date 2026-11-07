import random


def make_groups(n_groups, per_group, seed):
    rng = random.Random(seed)
    X, y, g = [], [], []
    for k in range(n_groups):
        c = [rng.gauss(0, 1) for _ in range(5)]
        lab = int(c[0] + rng.gauss(0, 2) > 0)
        for _ in range(per_group):
            X.append([ci + rng.gauss(0, 0.05) for ci in c])
            y.append(lab)
            g.append(k)
    return X, y, g


def nn1_predict(Xtr, ytr, Xte):
    out = []
    for x in Xte:
        best = min(range(len(Xtr)), key=lambda i: sum((a - b) ** 2 for a, b in zip(Xtr[i], x)))
        out.append(ytr[best])
    return out


def split_random(n, frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = int(n * frac)
    return idx[:k], idx[k:]


def split_groups(g, frac, seed):
    groups = sorted(set(g))
    random.Random(seed).shuffle(groups)
    tr_g = set(groups[:int(len(groups) * frac)])
    tr = [i for i, gi in enumerate(g) if gi in tr_g]
    te = [i for i, gi in enumerate(g) if gi not in tr_g]
    return tr, te


def accuracy_both(X, y, g, seed):
    res = []
    for tr, te in (split_random(len(X), 0.7, seed), split_groups(g, 0.7, seed)):
        pred = nn1_predict([X[i] for i in tr], [y[i] for i in tr], [X[i] for i in te])
        res.append(sum(p == y[i] for p, i in zip(pred, te)) / len(te))
    return tuple(res)


X, y, g = make_groups(40, 5, 0)
print(accuracy_both(X, y, g, 0))
