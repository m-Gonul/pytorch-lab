import math


def gcn_weights(n, edges):
    deg = [1] * n
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    w = {(i, i): 1 / deg[i] for i in range(n)}
    for i, j in edges:
        w[(i, j)] = w[(j, i)] = 1 / math.sqrt(deg[i] * deg[j])
    return w


def gcn_layer(X, n, edges, W):
    XW = [[sum(x[k] * W[k][c] for k in range(len(x))) for c in range(len(W[0]))] for x in X]
    out = [[0.0] * len(W[0]) for _ in range(n)]
    for (i, j), w in gcn_weights(n, edges).items():
        for c in range(len(W[0])):
            out[i][c] += w * XW[j][c]
    return out


def mean_neighbors(X, n, edges):
    d = len(X[0])
    tot = [[0.0] * d for _ in range(n)]
    cnt = [0] * n
    for i, j in edges:
        for a, b in ((i, j), (j, i)):
            cnt[a] += 1
            for c in range(d):
                tot[a][c] += X[b][c]
    return [[v / cnt[i] if cnt[i] else 0.0 for v in tot[i]] for i in range(n)]


print(gcn_layer([[1.0], [2.0], [3.0]], 3, [(0, 1), (1, 2)], [[1.0]]))   # [[1.3165], [2.2997], [2.3165]]
print(mean_neighbors([[1.0], [2.0], [3.0]], 3, [(0, 1), (1, 2)]))      # [[2], [2], [2]]
