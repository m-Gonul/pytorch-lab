import numpy as np


def learning_curve(sizes, degree, lam=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    xv = rng.uniform(-1, 1, 2000)
    yv = np.sin(3 * xv) + 0.2 * rng.normal(size=2000)
    tr, va = [], []
    for n in sizes:
        x = rng.uniform(-1, 1, n)
        y = np.sin(3 * x) + 0.2 * rng.normal(size=n)
        V = np.vander(x, degree + 1)
        c = np.linalg.solve(V.T @ V + lam * np.eye(degree + 1), V.T @ y)
        tr.append(float(np.mean((V @ c - y) ** 2)))
        va.append(float(np.mean((np.vander(xv, degree + 1) @ c - yv) ** 2)))
    return tr, va


def diagnose(train_err, val_err, noise_var):
    if val_err - train_err > 0.5 * max(train_err, noise_var):
        return "yüksek varyans"
    return "yüksek yanlılık" if train_err > 2 * noise_var else "iyi"


print(learning_curve([20, 100], 3))
