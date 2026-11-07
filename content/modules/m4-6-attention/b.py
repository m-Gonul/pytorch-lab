import math


def softmax_row(xs):
    ...


def attention(Q, K, V, causal=False):
    ...


Q = K = [[1, 0], [0, 1], [1, 1]]
V = [[1, 2], [3, 4], [5, 6]]
print(attention(Q, K, V))               # 1. satır [3, 4]
print(attention(Q, K, V, causal=True))  # 1. satır [1, 2]
