import numpy as np


def kfold_indices(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


def poly_cv(x, y, degrees, k=5, seed=0):
    out = {}
    folds = kfold_indices(len(x), k, seed)
    for d in degrees:
        errs = []
        for tr, va in folds:
            c = np.linalg.lstsq(np.vander(x[tr], d + 1), y[tr], rcond=None)[0]
            errs.append(np.mean((np.vander(x[va], d + 1) @ c - y[va]) ** 2))
        out[d] = float(np.mean(errs))
    return out


def early_stop(val_losses, patience):
    v = np.asarray(val_losses)
    run_min = np.minimum.accumulate(v)
    improved = np.r_[True, v[1:] < run_min[:-1]]
    last_imp = np.maximum.accumulate(np.where(improved, np.arange(len(v)), 0))
    stop = np.flatnonzero(np.arange(len(v)) - last_imp >= patience)
    s = int(stop[0]) if len(stop) else len(v) - 1
    return s, int(last_imp[s])


def standardize_leak(Xtr, Xte):
    A = np.vstack([Xtr, Xte])
    m, s = A.mean(0), A.std(0) + 1e-12
    return (Xtr - m) / s, (Xte - m) / s


def standardize_ok(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - m) / s, (Xte - m) / s


print(early_stop([1.0, 0.8, 0.7, 0.72, 0.75, 0.71, 0.9], 2))
