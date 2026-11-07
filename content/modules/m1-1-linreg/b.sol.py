def gd_step(xs, ys, w, b, lr):
    """MSE için bir gradient descent adımı → (w_yeni, b_yeni, güncellemeden önceki kayıp)."""
    n = len(xs)
    loss = gw = gb = 0.0
    for x, y in zip(xs, ys):
        e = w * x + b - y
        loss += e * e / n
        gw += 2 * e * x / n
        gb += 2 * e / n
    return w - lr * gw, b - lr * gb, loss


def fit(xs, ys, lr=0.05, steps=2000):
    w = b = 0.0
    for _ in range(steps):
        w, b, _ = gd_step(xs, ys, w, b, lr)
    return w, b


def closed_form(xs, ys):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs)
    w = sxy / sxx
    return w, ym - w * xm


print(gd_step([1, 2, 3], [2, 4, 5], 0.0, 0.0, 0.1))   # (1.666…, 0.733…, 15.0)
print(fit([1, 2, 3], [2, 4, 5]))                      # ≈ (1.5, 0.667)
print(closed_form([1, 2, 3], [2, 4, 5]))              # (1.5, 0.666…)
