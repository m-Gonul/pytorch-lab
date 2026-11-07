def sqdist_loops(Xte, Xtr):
    """→ nte × ntr kare mesafe listesi (iki döngü)."""
    D = []
    for a in Xte:
        row = []
        for b in Xtr:
            s = 0
            for ai, bi in zip(a, b):
                s += (ai - bi) ** 2
            row.append(s)
        D.append(row)
    return D


def knn_predict(D, ytr, k):
    """D: nte × ntr. → tahmin listesi. Eşit mesafe: küçük indeks önce; eşit oy: küçük etiket."""
    preds = []
    for row in D:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))
        votes = {}
        for j in order[:k]:
            votes[ytr[j]] = votes.get(ytr[j], 0) + 1
        best = None
        for c in sorted(votes):
            if best is None or votes[c] > votes[best]:
                best = c
        preds.append(best)
    return preds


Xtr = [[0, 0], [1, 3], [3, 1], [4, 4]]
ytr = [0, 1, 0, 1]
D = sqdist_loops([[1, 2]], Xtr)
print(D)                                     # [[5, 1, 5, 13]]
print(knn_predict(D, ytr, 1), knn_predict(D, ytr, 3))   # [1] [0]
