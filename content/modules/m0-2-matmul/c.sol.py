import numpy as np


def matmul_np(A, B):
    """Döngüsüz matris çarpımı."""
    return np.asarray(A) @ np.asarray(B)


def center_columns(X):
    """Her sütundan kendi ortalamasını çıkar (broadcasting)."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=0)


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(matmul_np(A, B))
print(center_columns(np.array([[1.0, 10.0], [3.0, 30.0]])))  # [[-1,-10],[1,10]]
