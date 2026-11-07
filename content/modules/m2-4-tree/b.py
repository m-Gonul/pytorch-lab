def gini(labels):
    ...


def best_split(X, y):
    """→ (j, t, skor) ya da None"""
    ...


def build_tree(X, y, max_depth):
    ...


def predict_tree(tree, x):
    ...


X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 1, 1, 1, 0]
print(best_split(X, y))             # (0, 2.5, 0.25)
tree = build_tree(X, y, 3)
print(tree)
print([predict_tree(tree, x) for x in X])
