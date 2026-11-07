def matmul(A, B):
    """A: n×k, B: k×m (iç içe listeler). n×m liste döndür."""
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"iç boyutlar uyuşmuyor: {k} ≠ {k2}")
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


print(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))  # [[19, 22], [43, 50]]
