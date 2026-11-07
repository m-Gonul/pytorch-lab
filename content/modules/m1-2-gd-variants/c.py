import numpy as np


def train(X, y, lr, batch_size, epochs, beta=0.0, seed=0):
    """→ (w, b, losses: (epochs,)). En fazla 2 katman döngü (epoch, batch)."""
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + 0.1 * rng.normal(size=500)
w, b, losses = train(X, y, lr=0.05, batch_size=32, epochs=30, beta=0.9)
print(w, b, losses[-1])
