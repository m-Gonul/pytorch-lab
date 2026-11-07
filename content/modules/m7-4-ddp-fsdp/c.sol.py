import numpy as np


def worker_grads(w, X, y, K):
    return [2 * Xk.T @ (Xk @ w - yk) / len(yk) for Xk, yk in zip(np.array_split(X, K), np.array_split(y, K))]


def allreduce_mean(grads):
    return np.mean(grads, axis=0)


def adam_update(w, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(vh) + eps), m, v


def zero1_step(w, g, states, t, lr=1e-2):
    K = len(states)
    ws, gs = np.array_split(w, K), np.array_split(g, K)
    new_w, new_states = [], []
    for wk, gk, (mk, vk) in zip(ws, gs, states):
        wk, mk, vk = adam_update(wk, gk, mk, vk, t, lr)
        new_w.append(wk)
        new_states.append((mk, vk))
    return np.concatenate(new_w), new_states


rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(64, 6)), rng.normal(size=64), rng.normal(size=6)
g = allreduce_mean(worker_grads(w, X, y, 4))
print(g - 2 * X.T @ (X @ w - y) / 64)
