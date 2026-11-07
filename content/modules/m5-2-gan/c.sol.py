import numpy as np


def value_expect(p, q, D):
    return (p * np.log(D)).sum() + (q * np.log(1 - D)).sum()


def optimal_d(p, q):
    s = p + q
    return np.where(s > 0, p / np.where(s > 0, s, 1), 0.5)


def jsd(p, q):
    m = 0.5 * (p + q)
    kl = lambda a: np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(m > 0, m, 1)), 0).sum()
    return 0.5 * kl(p) + 0.5 * kl(q)


def gen_logit_grads(l):
    s = 1 / (1 + np.exp(-l))
    return -s, -(1 - s)


x = np.linspace(-4, 6, 50)
p = np.exp(-0.5 * x ** 2); p /= p.sum()
q = np.exp(-0.5 * (x - 2) ** 2); q /= q.sum()
print(value_expect(p, q, optimal_d(p, q)), -np.log(4) + 2 * jsd(p, q))
print(gen_logit_grads(np.array([-5.0, 0.0, 5.0])))
