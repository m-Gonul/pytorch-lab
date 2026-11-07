import numpy as np


def svm_loss(W, X, y, reg=0.0, delta=1.0):
    """→ (loss, dW). Döngü yok."""
    ...


rng = np.random.default_rng(0)
W = rng.normal(size=(5, 3)) * 0.01
X = rng.normal(size=(10, 5))
y = rng.integers(0, 3, 10)
print(svm_loss(W, X, y, reg=0.1))
