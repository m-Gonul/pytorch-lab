import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rows = []
    for name, A in [("dikdörtgen", np.pad(np.ones((12, 24), bool), 4)),
                    ("madeni paralar", ndimage.binary_opening(data.coins()[40:160, 40:200] > 110, np.ones((3, 3))))]:
        def t(A=A, name=name):
            for Bn, B in [("kare", np.ones((3, 3), bool)), ("artı", ndimage.generate_binary_structure(2, 1))]:
                S, parts = need(ns, "skeleton")(A, B)
                ensure(np.all(S <= A), "iskelet A'nın içinde olmalı")
                E = A.copy(); K = 0
                while ndimage.binary_erosion(E, B).any():
                    E = ndimage.binary_erosion(E, B); K += 1
                ensure(len(parts) == K + 1, f"{len(parts)} parça, {K + 1} bekleniyordu")
                ensure(np.array_equal(need(ns, "reconstruct")(parts, B), A), "geri çatma A'yı vermedi")
                rows.append([name, Bn, int(A.sum()), int(S.sum()), K])
        R.test(f"iskelet ve geri çatma ({name})", t)
    if rows:
        R.table("Lantuéjoul iskeleti", ["nesne", "SE", "|A|", "|S|", "K"], rows, note="İskelet bağlantılı olmak zorunda değildir; ama S_k'ler ve k'lar A'yı tam olarak geri verir.")
