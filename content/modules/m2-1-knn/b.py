def sqdist_loops(Xte, Xtr):
    """→ nte × ntr kare mesafe listesi (iki döngü)."""
    ...


def knn_predict(D, ytr, k):
    """D: nte × ntr. → tahmin listesi. Eşit mesafe: küçük indeks önce; eşit oy: küçük etiket."""
    ...


Xtr = [[0, 0], [1, 3], [3, 1], [4, 4]]
ytr = [0, 1, 0, 1]
D = sqdist_loops([[1, 2]], Xtr)
print(D)                                     # [[5, 1, 5, 13]]
print(knn_predict(D, ytr, 1), knn_predict(D, ytr, 3))   # [1] [0]
