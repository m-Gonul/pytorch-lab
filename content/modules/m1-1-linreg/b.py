def gd_step(xs, ys, w, b, lr):
    """MSE için bir gradient descent adımı → (w_yeni, b_yeni, güncellemeden önceki kayıp)."""
    ...


def fit(xs, ys, lr=0.05, steps=2000):
    ...


def closed_form(xs, ys):
    ...


print(gd_step([1, 2, 3], [2, 4, 5], 0.0, 0.0, 0.1))   # (1.666…, 0.733…, 15.0)
print(fit([1, 2, 3], [2, 4, 5]))                      # ≈ (1.5, 0.667)
print(closed_form([1, 2, 3], [2, 4, 5]))              # (1.5, 0.666…)
