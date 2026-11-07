def soft_threshold(z, t):
    ...


def ridge2(X, y, lam):
    """İki özellik, bias yok. → [w1, w2]"""
    ...


def lasso_cd(X, y, lam, iters=100):
    """Amaç: (1/2n)‖y − Xw‖² + λ‖w‖₁, bias yok. → w (liste)"""
    ...


X = [[1, 0], [0, 1], [1, 1]]
y = [1, 2, 3]
print(ridge2(X, y, 0.0), ridge2(X, y, 1.0))     # [1, 2] [0.875, 1.375]
print(soft_threshold(1.5, 0.5), soft_threshold(-0.3, 0.5))
print(lasso_cd(X, y, 0.3))
