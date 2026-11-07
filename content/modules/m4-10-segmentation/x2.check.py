import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz konvolüsyon", lambda: max_loops(src, 0, func="dilated_conv2d"))

    def t():
        rng = np.random.default_rng(0)
        x = rng.normal(size=(12, 13))
        w = rng.normal(size=(3, 3))
        y = need(ns, "dilated_conv2d")(x, w, 2)
        xp = np.pad(x, 2)
        ref = sum(w[u, v] * xp[2 * u:2 * u + 12, 2 * v:2 * v + 13] for u in range(3) for v in range(3))
        close(y, ref, atol=1e-10, what="d = 2")
        close(ns["dilated_conv2d"](x, w, 1)[1:-1, 1:-1], sum(w[u, v] * x[u:u + 10, v:v + 11] for u in range(3) for v in range(3)), atol=1e-10, what="d = 1 normal konv.")
        ensure(need(ns, "receptive_field")([3] * 4, [1, 2, 4, 8]) == 31, "1, 2, 4, 8 → 31")
        rows = []
        for ds in ([1, 1, 1, 1], [2, 2, 2], [1, 2, 4, 8], [1, 2, 5], [1, 2, 3]):
            Fp = need(ns, "footprint")(ds)
            rows.append([str(ds), Fp.shape[0], int(Fp.sum()), f"{Fp.mean():.2f}"])
        ensure(not ns["footprint"]([2, 2, 2]).all() and ns["footprint"]([1, 2, 5]).all(), "ızgaralanma: [2,2,2] delikli, [1,2,5] tam")
        R_.table("3×3 genişletilmiş konvolüsyon yığınları", ["genişletmeler", "receptive field", "etkileyen piksel", "doluluk"], rows,
                 note="Genişletme, parametre ve çözünürlük kaybı olmadan receptive field'ı üstel büyütür (Yu & Koltun; DeepLab ASPP). Ama aynı d'yi tekrarlamak ızgaralanma (gridding) yaratır: merkez çıktı yalnız d'nin katlarındaki pikselleri görür. HDC (Wang ve ark.): ortak böleni 1 olan genişletmeler (1, 2, 5) alanı tam kaplar.")
    R_.test("dilated konvolüsyon", t)
