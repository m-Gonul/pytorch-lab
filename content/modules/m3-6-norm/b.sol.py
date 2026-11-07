import math


def bn_forward(X, gamma, beta, eps=1e-5):
    N, D = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / N for j in range(D)]
    var = [sum((r[j] - mu[j]) ** 2 for r in X) / N for j in range(D)]
    std = [math.sqrt(v + eps) for v in var]
    xh = [[(r[j] - mu[j]) / std[j] for j in range(D)] for r in X]
    out = [[gamma[j] * row[j] + beta[j] for j in range(D)] for row in xh]
    return out, (xh, std, gamma)


def bn_backward(dout, cache):
    xh, std, gamma = cache
    N, D = len(xh), len(xh[0])
    dgamma = [sum(dout[i][j] * xh[i][j] for i in range(N)) for j in range(D)]
    dbeta = [sum(dout[i][j] for i in range(N)) for j in range(D)]
    dX = [[0.0] * D for _ in range(N)]
    for j in range(D):
        dxh = [gamma[j] * dout[i][j] for i in range(N)]
        s1 = sum(dxh)
        s2 = sum(dxh[i] * xh[i][j] for i in range(N))
        for i in range(N):
            dX[i][j] = (N * dxh[i] - s1 - xh[i][j] * s2) / (N * std[j])
    return dX, dgamma, dbeta


def rms_norm(x, g, eps=1e-6):
    r = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / r * gi for v, gi in zip(x, g)]


out, cache = bn_forward([[1.0], [2.0], [3.0]], [2.0], [1.0], eps=0.0)
print(out)                                   # [[-1.449], [1.0], [3.449]]
print(bn_backward([[1.0], [0.0], [0.0]], cache))
