def hinge_grad(X, y, w, b, lam):
    n, d = len(X), len(w)
    L, gw, gb = 0.0, [2 * lam * wj for wj in w], 0.0
    for xi, yi in zip(X, y):
        m = yi * (sum(wj * xj for wj, xj in zip(w, xi)) + b)
        if m < 1:
            L += (1 - m) / n
            for j in range(d):
                gw[j] -= yi * xi[j] / n
            gb -= yi / n
    L += lam * sum(wj * wj for wj in w)
    return L, gw, gb


def fit_svm(X, y, lam=0.01, lr=0.1, steps=500):
    w, b = [0.0] * len(X[0]), 0.0
    for _ in range(steps):
        _, gw, gb = hinge_grad(X, y, w, b, lam)
        w = [wj - lr * g for wj, g in zip(w, gw)]
        b -= lr * gb
    return w, b


def multiclass_hinge(scores, y, delta=1.0):
    return sum(max(0.0, s - scores[y] + delta) for j, s in enumerate(scores) if j != y)


print(hinge_grad([[2, 1], [0, 1], [1, 1]], [1, -1, 1], [1.0, -1.0], 0.0, 0.0))
print(multiclass_hinge([3.2, 5.1, -1.7], 0))   # 2.9
