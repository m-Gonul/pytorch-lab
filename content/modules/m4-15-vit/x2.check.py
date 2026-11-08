import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        N, k, D = 16, 12, 8
        patches = rng.normal(size=(N, k))
        E, cls = rng.normal(size=(k, D)) / 3, rng.normal(size=D)
        pos = rng.normal(size=(N + 1, D))
        W = [rng.normal(size=(D, D)) / 3 for _ in range(3)]
        perm = rng.permutation(N)
        a = need(ns, "vit_forward")(patches, E, cls, None, W)
        b = ns["vit_forward"](patches[perm], E, cls, None, W)
        close(b[0], a[0], atol=1e-12, what="konumsuz: [class] çıktısı yama sırasından bağımsız")
        close(b[1:], a[1:][perm], atol=1e-12, what="yama çıktıları aynı permütasyonla yer değiştirir")
        c = ns["vit_forward"](patches, E, cls, pos, W)
        d = ns["vit_forward"](patches[perm], E, cls, pos, W)
        ensure(np.abs(c[0] - d[0]).max() > 1e-3, "konumlu: sıra fark eder")
        R_.note("Konum embedding'i olmayan öz-attention permütasyona eşdeğerdir: yamaları karıştırınca çıktılar aynı şekilde karışır, [class] (ve GAP) değişmez. Böyle bir ViT için bir yüz ile parçaları karıştırılmış yüz aynı görüntüdür — d aşamasındaki konumsuz ViT'in düşük doğruluğunun nedeni. CNN'de bu bilgi konvolüsyonun yerelliğiyle zaten vardır.")
    R_.test("permütasyon eşdeğerliği", t)
