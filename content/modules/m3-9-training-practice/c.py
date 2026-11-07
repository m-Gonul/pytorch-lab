import numpy as np


def kfold_indices(n, k, seed):
    ...


def poly_cv(x, y, degrees, k=5, seed=0):
    ...


def early_stop(val_losses, patience):
    ...


def standardize_leak(Xtr, Xte):
    ...


def standardize_ok(Xtr, Xte):
    ...


print(early_stop([1.0, 0.8, 0.7, 0.72, 0.75, 0.71, 0.9], 2))
