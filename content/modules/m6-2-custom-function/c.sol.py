import numpy as np


def ce_forward(logits, y):
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    N = logits.shape[0]
    loss = -logp[np.arange(N), y].mean()
    return loss, (np.exp(logp), y)


def ce_backward(cache):
    p, y = cache
    N = p.shape[0]
    d = p.copy()
    d[np.arange(N), y] -= 1
    return d / N


def dir_gradcheck(f, grad, x, rng, k=10, eps=1e-6):
    g = grad(x)
    worst = 0.0
    for _ in range(k):
        v = rng.normal(size=x.shape)
        num = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
        ana = (g * v).sum()
        worst = max(worst, abs(num - ana) / max(abs(ana), 1e-12))
    return worst


rng = np.random.default_rng(0)
Z, y = rng.normal(size=(5, 4)), rng.integers(0, 4, 5)
loss, cache = ce_forward(Z, y)
print(loss, ce_backward(cache).sum(1))     # satır toplamları 0
