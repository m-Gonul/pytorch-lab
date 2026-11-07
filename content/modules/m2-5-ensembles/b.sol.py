def fit_stump(xs, rs):
    n = len(xs)
    mean = sum(rs) / n
    vals = sorted(set(xs))
    best = (None, mean, mean, 0.0)               # (t, sol, sağ, kazanç)
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2
        L = [r for x, r in zip(xs, rs) if x <= t]
        R = [r for x, r in zip(xs, rs) if x > t]
        gain = sum(L) ** 2 / len(L) + sum(R) ** 2 / len(R)
        if best[0] is None or gain > best[3]:
            best = (t, sum(L) / len(L), sum(R) / len(R), gain)
    return best[:3]


def gbm_fit(xs, ys, rounds, lr):
    F0 = sum(ys) / len(ys)
    F = [F0] * len(xs)
    stumps = []
    for _ in range(rounds):
        r = [y - f for y, f in zip(ys, F)]
        st = fit_stump(xs, r)
        stumps.append(st)
        F = [f + lr * stump_eval(st, x) for f, x in zip(F, xs)]
    return {"F0": F0, "lr": lr, "stumps": stumps}


def stump_eval(st, x):
    t, left, right = st
    return left if t is None or x <= t else right


def gbm_predict(model, x):
    return model["F0"] + model["lr"] * sum(stump_eval(st, x) for st in model["stumps"])


m = gbm_fit([1, 2, 3, 4], [1, 1, 3, 3], 2, 0.5)
print([gbm_predict(m, x) for x in [1, 2, 3, 4]])   # [1.25, 1.25, 2.75, 2.75]
