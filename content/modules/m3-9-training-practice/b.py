import random


def make_groups(n_groups, per_group, seed):
    ...


def nn1_predict(Xtr, ytr, Xte):
    ...


def split_random(n, frac, seed):
    ...


def split_groups(g, frac, seed):
    ...


def accuracy_both(X, y, g, seed):
    ...


X, y, g = make_groups(40, 5, 0)
print(accuracy_both(X, y, g, 0))
