def matmul(A, B):
    """A: n×k, B: k×m (iç içe listeler). n×m liste döndür."""
    # 1) shape'leri oku, iç boyut uyuşmazsa ValueError
    # 2) sıfırlarla dolu n×m sonuç oluştur
    # 3) üç iç içe döngü ile C[i][j] += A[i][t] * B[t][j]
    raise NotImplementedError


print(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))  # [[19, 22], [43, 50]]
