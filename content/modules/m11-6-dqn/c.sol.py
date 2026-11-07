import numpy as np


def baird_features():
    X = np.zeros((7, 8))
    X[np.arange(6), np.arange(6)] = 2
    X[:6, 7] = 1
    X[6, 6], X[6, 7] = 1, 2
    return X


def expected_td(w0, alpha, gamma, n_iter, mu):
    X = baird_features()
    w = np.array(w0, float)
    hist = [np.linalg.norm(w)]
    for _ in range(n_iter):
        v = X @ w
        w = w + alpha * (mu * (gamma * v[6] - v)) @ X
        hist.append(np.linalg.norm(w))
    return w, np.array(hist)


def expected_tdc(w0, alpha, beta, gamma, n_iter, mu):
    X = baird_features()
    w = np.array(w0, float)
    u = np.zeros(8)
    hist = [np.linalg.norm(w)]
    for _ in range(n_iter):
        v = X @ w
        delta = gamma * v[6] - v
        xu = X @ u
        w = w + alpha * ((mu * delta) @ X - gamma * X[6] * (mu @ xu))
        u = u + beta * (mu * (delta - xu)) @ X
        hist.append(np.linalg.norm(w))
    return w, np.array(hist)


w0 = [1, 1, 1, 1, 1, 1, 10, 1]
print(expected_td(w0, 0.01, 0.99, 500, np.full(7, 1 / 7))[1][-1])
