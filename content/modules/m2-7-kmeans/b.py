def assign(points, centers):
    ...


def update(points, labels, centers):
    ...


def kmeans(points, centers, iters):
    ...


P = [(0, 0), (1, 0), (4, 4), (5, 4)]
print(kmeans(P, [(0.0, 0.0), (1.0, 0.0)], 2))   # ([(0.5, 0.0), (4.5, 4.0)], [0, 0, 1, 1], 1.0)
