import numpy as np


def value_expect(p, q, D):
    ...


def optimal_d(p, q):
    ...


def jsd(p, q):
    ...


def gen_logit_grads(l):
    ...


x = np.linspace(-4, 6, 50)
p = np.exp(-0.5 * x ** 2); p /= p.sum()
q = np.exp(-0.5 * (x - 2) ** 2); q /= q.sum()
print(value_expect(p, q, optimal_d(p, q)), -np.log(4) + 2 * jsd(p, q))
print(gen_logit_grads(np.array([-5.0, 0.0, 5.0])))
