import math


def log_normal(x, mu, var):
    return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)


def lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(a - m) for a in v))


def e_step(xs, pis, mus, vars):
    R = []
    for x in xs:
        logs = [math.log(p) + log_normal(x, m, v) for p, m, v in zip(pis, mus, vars)]
        z = lse(logs)
        R.append([math.exp(a - z) for a in logs])
    return R


def m_step(xs, R, reg=1e-6):
    n, K = len(xs), len(R[0])
    pis, mus, vs = [], [], []
    for k in range(K):
        Nk = sum(r[k] for r in R)
        mu = sum(r[k] * x for r, x in zip(R, xs)) / Nk
        var = sum(r[k] * (x - mu) ** 2 for r, x in zip(R, xs)) / Nk + reg
        pis.append(Nk / n); mus.append(mu); vs.append(var)
    return pis, mus, vs


def log_likelihood(xs, pis, mus, vars):
    return sum(lse([math.log(p) + log_normal(x, m, v) for p, m, v in zip(pis, mus, vars)]) for x in xs)


def em(xs, pis, mus, vars, iters):
    hist = [log_likelihood(xs, pis, mus, vars)]
    for _ in range(iters):
        pis, mus, vars = m_step(xs, e_step(xs, pis, mus, vars))
        hist.append(log_likelihood(xs, pis, mus, vars))
    return pis, mus, vars, hist


R = e_step([1, 2, 4], [0.5, 0.5], [1, 4], [1, 1])
print([round(r[0], 4) for r in R])          # [0.989, 0.8176, 0.011]
print(m_step([1, 2, 4], R))
