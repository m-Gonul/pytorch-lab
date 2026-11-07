import math


def sig(t):
    return 1 / (1 + math.exp(-t))


def grads(theta, psi):
    s = sig(-psi * theta)
    return psi * s, theta * s


def simulate(theta, psi, h, steps, gamma=0.0, alternating=False):
    for _ in range(steps):
        g_th, g_ps = grads(theta, psi)
        if alternating:
            theta = theta - h * g_th
            g_ps = grads(theta, psi)[1]
            psi = psi + h * (g_ps - gamma * psi)
        else:
            theta, psi = theta - h * g_th, psi + h * (g_ps - gamma * psi)
    return theta, psi


for kw in [{}, {"alternating": True}, {"gamma": 1.0}]:
    th, ps = simulate(1.0, 1.0, 0.1, 3000, **kw)
    print(kw, round(th, 4), round(ps, 4), round(math.hypot(th, ps), 4))
