import math


def sig(t):
    return 1 / (1 + math.exp(-t))


def grads(theta, psi):
    ...


def simulate(theta, psi, h, steps, gamma=0.0, alternating=False):
    ...


for kw in [{}, {"alternating": True}, {"gamma": 1.0}]:
    th, ps = simulate(1.0, 1.0, 0.1, 3000, **kw)
    print(kw, round(th, 4), round(ps, 4), round(math.hypot(th, ps), 4))
