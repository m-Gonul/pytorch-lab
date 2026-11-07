import math


def gcn_weights(n, edges):
    ...


def gcn_layer(X, n, edges, W):
    ...


def mean_neighbors(X, n, edges):
    ...


print(gcn_layer([[1.0], [2.0], [3.0]], 3, [(0, 1), (1, 2)], [[1.0]]))   # [[1.3165], [2.2997], [2.3165]]
print(mean_neighbors([[1.0], [2.0], [3.0]], 3, [(0, 1), (1, 2)]))      # [[2], [2], [2]]
