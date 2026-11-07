import numpy as np


def softmax(theta):
    e = np.exp(theta - theta.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def kl(p, q):
    return (p * (np.log(p) - np.log(q))).sum(-1)


def fisher(pi):
    return pi[:, :, None] * np.eye(pi.shape[1])[None] - pi[:, :, None] * pi[:, None, :]


def fvp(pi, v):
    return pi * v - pi * (pi * v).sum(-1, keepdims=True)


def policy_grad(theta, r):
    pi = softmax(theta)
    return pi * (r - (pi * r).sum(-1, keepdims=True))


def cg(Avp, b, iters=20, tol=1e-12):
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    rr = (r * r).sum(-1, keepdims=True)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / np.maximum((p * Ap).sum(-1, keepdims=True), 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rr2 = (r * r).sum(-1, keepdims=True)
        if rr2.max() < tol:
            break
        p = r + rr2 / np.maximum(rr, 1e-300) * p
        rr = rr2
    return x


def natural_step(theta, r, delta, damping=1e-3):
    pi = softmax(theta)
    g = policy_grad(theta, r)
    s = cg(lambda v: fvp(pi, v) + damping * v, g)
    beta = np.sqrt(2 * delta / np.maximum((s * fvp(pi, s)).sum(-1, keepdims=True), 1e-300))
    return theta + beta * s


th = np.zeros((1, 3)); r = np.array([[1.0, 0.0, 0.5]])
print(natural_step(th, r, 0.01))
