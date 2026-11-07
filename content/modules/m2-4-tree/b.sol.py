def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for c in labels:
        counts[c] = counts.get(c, 0) + 1
    return 1.0 - sum((k / n) ** 2 for k in counts.values())


def best_split(X, y):
    """→ (j, t, skor) ya da None"""
    n, d = len(X), len(X[0])
    best = None
    for j in range(d):
        vals = sorted(set(row[j] for row in X))
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            L = [y[i] for i in range(n) if X[i][j] <= t]
            R = [y[i] for i in range(n) if X[i][j] > t]
            s = len(L) / n * gini(L) + len(R) / n * gini(R)
            if best is None or s < best[2]:
                best = (j, t, s)
    return best


def majority(y):
    counts = {}
    for c in y:
        counts[c] = counts.get(c, 0) + 1
    return min(counts, key=lambda c: (-counts[c], c))


def build_tree(X, y, max_depth):
    if max_depth == 0 or len(set(y)) == 1:
        return {"leaf": majority(y)}
    sp = best_split(X, y)
    if sp is None:
        return {"leaf": majority(y)}
    j, t, _ = sp
    li = [i for i in range(len(X)) if X[i][j] <= t]
    ri = [i for i in range(len(X)) if X[i][j] > t]
    return {"j": j, "t": t,
            "left": build_tree([X[i] for i in li], [y[i] for i in li], max_depth - 1),
            "right": build_tree([X[i] for i in ri], [y[i] for i in ri], max_depth - 1)}


def predict_tree(tree, x):
    while "leaf" not in tree:
        tree = tree["left"] if x[tree["j"]] <= tree["t"] else tree["right"]
    return tree["leaf"]


X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 1, 1, 1, 0]
print(best_split(X, y))             # (0, 2.5, 0.25)
tree = build_tree(X, y, 3)
print(tree)
print([predict_tree(tree, x) for x in X])
