import numpy as np


def num_grad_np(f, x, h=1e-5):
    """f: (k, n) → (k,). x: (n,). Döngüsüz merkezi fark gradyanı → (n,)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    E = h * np.eye(n)
    fx = f(np.concatenate([x + E, x - E]))
    return (fx[:n] - fx[n:]) / (2 * h)


def num_jacobian_np(g, x, h=1e-5):
    """g: (k, n) → (k, m). x: (n,). → (m, n) Jacobian."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    E = h * np.eye(n)
    G = g(np.concatenate([x + E, x - E]))      # (2n, m)
    return ((G[:n] - G[n:]) / (2 * h)).T       # (n, m) → (m, n)


f = lambda X: (X[:, 0] + X[:, 1]) * X[:, 1]
g = lambda X: np.stack([X[:, 0] * X[:, 1], X[:, 0] + X[:, 1] ** 2], axis=1)
print(num_grad_np(f, np.array([2.0, 3.0])))        # [3. 8.]
print(num_jacobian_np(g, np.array([2.0, 3.0])))    # [[3. 2.] [1. 6.]]
