def fit_stump(xs, rs):
    ...


def gbm_fit(xs, ys, rounds, lr):
    ...


def gbm_predict(model, x):
    ...


m = gbm_fit([1, 2, 3, 4], [1, 1, 3, 3], 2, 0.5)
print([gbm_predict(m, x) for x in [1, 2, 3, 4]])   # [1.25, 1.25, 2.75, 2.75]
