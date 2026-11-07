import numpy as np


def tree_reduce(x, op, identity):
    ...


def naive_sum32(x):
    ...


def tree_sum32(x):
    ...


def argmax_tree(x):
    ...


def chain_product_tree(M):
    ...


print(tree_reduce(np.array([3, 1, 7, 0, 4, 1, 6, 3]), np.maximum, -np.inf))
