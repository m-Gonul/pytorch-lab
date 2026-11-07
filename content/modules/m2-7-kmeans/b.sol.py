def sqd(p, c):
    return sum((a - b) ** 2 for a, b in zip(p, c))


def assign(points, centers):
    labels = []
    for p in points:
        best = 0
        for k in range(1, len(centers)):
            if sqd(p, centers[k]) < sqd(p, centers[best]):
                best = k
        labels.append(best)
    return labels


def update(points, labels, centers):
    d = len(points[0])
    new = []
    for k, c in enumerate(centers):
        members = [p for p, l in zip(points, labels) if l == k]
        if not members:
            new.append(tuple(c))
        else:
            new.append(tuple(sum(p[j] for p in members) / len(members) for j in range(d)))
    return new


def kmeans(points, centers, iters):
    for _ in range(iters):
        centers = update(points, assign(points, centers), centers)
    labels = assign(points, centers)
    inertia = sum(sqd(p, centers[l]) for p, l in zip(points, labels))
    return centers, labels, inertia


P = [(0, 0), (1, 0), (4, 4), (5, 4)]
print(kmeans(P, [(0.0, 0.0), (1.0, 0.0)], 2))   # ([(0.5, 0.0), (4.5, 4.0)], [0, 0, 1, 1], 1.0)
