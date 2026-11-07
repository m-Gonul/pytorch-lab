def hinge_grad(X, y, w, b, lam):
    ...


def fit_svm(X, y, lam=0.01, lr=0.1, steps=500):
    ...


def multiclass_hinge(scores, y, delta=1.0):
    ...


print(hinge_grad([[2, 1], [0, 1], [1, 1]], [1, -1, 1], [1.0, -1.0], 0.0, 0.0))
print(multiclass_hinge([3.2, 5.1, -1.7], 0))   # 2.9
