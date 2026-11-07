import numpy as np


def cosine_matrix(X):
    """X: (n, d) → (n, n) kosinüs benzerlik matrisi. Döngü yok."""
    X = np.asarray(X, dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T


def most_similar(X, i):
    """i. satıra en benzer başka satırın indeksi."""
    s = cosine_matrix(X)[i].copy()
    s[i] = -np.inf
    return int(np.argmax(s))


X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
print(np.round(cosine_matrix(X), 3))
print(most_similar(X, 0))  # 1
