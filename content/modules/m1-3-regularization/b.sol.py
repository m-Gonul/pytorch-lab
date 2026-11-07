def soft_threshold(z, t):
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


def ridge2(X, y, lam):
    """İki özellik, bias yok. → [w1, w2]"""
    a = b = d = c0 = c1 = 0.0
    for (x0, x1), yi in zip(X, y):
        a += x0 * x0; b += x0 * x1; d += x1 * x1
        c0 += x0 * yi; c1 += x1 * yi
    a += lam; d += lam
    det = a * d - b * b
    return [(d * c0 - b * c1) / det, (-b * c0 + a * c1) / det]


def lasso_cd(X, y, lam, iters=100):
    """Amaç: (1/2n)‖y − Xw‖² + λ‖w‖₁, bias yok. → w (liste)"""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    col_sq = [sum(X[i][j] ** 2 for i in range(n)) / n for j in range(d)]
    for _ in range(iters):
        for j in range(d):
            rho = 0.0
            for i in range(n):
                r = y[i] - sum(X[i][k] * w[k] for k in range(d) if k != j)
                rho += X[i][j] * r / n
            w[j] = soft_threshold(rho, lam) / col_sq[j] if col_sq[j] > 0 else 0.0
    return w


X = [[1, 0], [0, 1], [1, 1]]
y = [1, 2, 3]
print(ridge2(X, y, 0.0), ridge2(X, y, 1.0))     # [1, 2] [0.875, 1.375]
print(soft_threshold(1.5, 0.5), soft_threshold(-0.3, 0.5))
print(lasso_cd(X, y, 0.3))
