import numpy as np


def tree_reduce(x, op, identity):
    x = np.asarray(x)
    while x.shape[0] > 1:
        if x.shape[0] % 2:
            x = np.concatenate([x, np.full((1,) + x.shape[1:], identity, dtype=x.dtype)])
        x = op(x[0::2], x[1::2])
    return x[0]


def naive_sum32(x):
    return np.cumsum(np.asarray(x, np.float32), dtype=np.float32)[-1]


def tree_sum32(x):
    return tree_reduce(np.asarray(x, np.float32), np.add, np.float32(0))


def argmax_tree(x):
    v = np.asarray(x, float)
    i = np.arange(len(v))
    while len(v) > 1:
        if len(v) % 2:
            v = np.append(v, -np.inf)
            i = np.append(i, len(i) + 10 ** 9)
        a, b = v[0::2], v[1::2]
        ia, ib = i[0::2], i[1::2]
        take_b = (b > a) | ((b == a) & (ib < ia))
        v, i = np.where(take_b, b, a), np.where(take_b, ib, ia)
    return int(i[0])


def chain_product_tree(M):
    M = np.asarray(M, float)
    while M.shape[0] > 1:
        if M.shape[0] % 2:
            M = np.concatenate([M, np.eye(2)[None]])
        M = M[0::2] @ M[1::2]
    return M[0]


print(tree_reduce(np.array([3, 1, 7, 0, 4, 1, 6, 3]), np.maximum, -np.inf))
