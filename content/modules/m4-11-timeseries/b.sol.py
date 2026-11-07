def difference(y, d=1):
    for _ in range(d):
        y = [b - a for a, b in zip(y, y[1:])]
    return list(y)


def undifference(y, dz, d):
    if d == 0:
        return list(dz)
    f = undifference(difference(y, 1), dz, d - 1)
    out, last = [], y[-1]
    for v in f:
        last = last + v
        out.append(last)
    return out


def arma_predict(y, alpha, theta):
    p, q = len(alpha), len(theta)
    preds, eps = [], []
    for t in range(len(y) + 1):
        v = sum(alpha[i] * y[t - 1 - i] for i in range(p) if t - 1 - i >= 0)
        v += sum(theta[i] * eps[t - 1 - i] for i in range(q) if t - 1 - i >= 0)
        preds.append(v)
        if t < len(y):
            eps.append(y[t] - v)
    return preds


def _solve(A, b):
    n = len(b)
    M = [row[:] + [bv] for row, bv in zip(A, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def ar_acf(phi, nlags):
    p = len(phi)
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for k in range(1, p + 1):
        A[k - 1][k - 1] += 1.0
        for i in range(1, p + 1):
            lag = abs(k - i)
            if lag == 0:
                b[k - 1] += phi[i - 1]
            else:
                A[k - 1][lag - 1] -= phi[i - 1]
    rho = [1.0] + _solve(A, b)
    for k in range(p + 1, nlags + 1):
        rho.append(sum(phi[i - 1] * rho[k - i] for i in range(1, p + 1)))
    return rho[:nlags + 1]


def rnn_params(sizes, n_in=1, double_bias=False):
    tot, prev = 0, n_in
    for h in sizes:
        tot += prev * h + h * h + h * (2 if double_bias else 1)
        prev = h
    return tot


def receptive_field(k, dilations):
    return 1 + sum((k - 1) * d for d in dilations)


print(difference([1, 4, 9, 16, 25, 36], 2))
