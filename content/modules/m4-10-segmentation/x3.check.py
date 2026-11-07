import numpy as np
from collections import deque
from labcheck import need, ensure


def _bfs(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    K = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not lab[i, j]:
                K += 1
                q = deque([(i, j)])
                lab[i, j] = K
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and not lab[x, y]:
                            lab[x, y] = K
                            q.append((x, y))
    return lab, K


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        for _ in range(5):
            m = rng.random((30, 40)) < 0.45
            lab, K = need(ns, "connected_components")(m)
            ref, Kr = _bfs(m)
            ensure(K == Kr and set(lab[~m].ravel()) <= {0}, "bileşen sayısı")
            pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
            ensure(len(pairs) == K, "bölümleme referansla aynı (etiket adları hariç)")
        sem = np.zeros((20, 30), int)
        sem[2:8, 2:8] = 1
        sem[2:8, 8:14] = 1
        sem[12:18, 2:8] = 1
        sem[12:18, 20:26] = 2
        inst = need(ns, "instances_from_semantic")(sem, 3)
        ensure(sorted((c, int(m.sum())) for c, m in inst) == [(1, 36), (1, 72), (2, 36)], "örnekler")
        R_.note("Değen iki kare (sınıf 1) tek bir bileşen olarak çıkar: anlamsal maske, aynı sınıftan bitişik nesneleri ayırmaya yetmez. Örnek segmentasyonu bu yüzden ya tespitten başlar (Mask R-CNN: her kutu için ayrı maske) ya da ayırma sınırlarını ayrıca öğrenir (U-Net'in ağırlık haritası — ödev).")
    R_.test("bağlantılı bileşenler", t)
