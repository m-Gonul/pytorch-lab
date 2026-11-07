import math


def softmax_row(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def attention(Q, K, V, causal=False):
    dk = len(Q[0])
    out = []
    for i, q in enumerate(Q):
        scores = []
        for j, k in enumerate(K):
            if causal and j > i:
                scores.append(-math.inf)
            else:
                scores.append(sum(a * b for a, b in zip(q, k)) / math.sqrt(dk))
        w = softmax_row(scores)
        out.append([sum(w[j] * V[j][c] for j in range(len(V))) for c in range(len(V[0]))])
    return out


Q = K = [[1, 0], [0, 1], [1, 1]]
V = [[1, 2], [3, 4], [5, 6]]
print(attention(Q, K, V))               # 1. satır [3, 4]
print(attention(Q, K, V, causal=True))  # 1. satır [1, 2]
