import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    img = data.coffee()[::2, ::2]

    def sp():
        lab, sp = need(ns, "superpixel_kmeans")(img, 300, 4)
        ensure(lab.shape == img.shape[:2] and lab.max() < 4, "k = 4 etiket")
        for s in np.unique(sp):
            ensure(len(np.unique(lab[sp == s])) == 1, "bir süper pikselin tüm pikselleri aynı etiketi almalı")
        pl = need(ns, "pixel_kmeans")(img, 4)
        comps = lambda L: sum(ndimage.label(L == k)[1] for k in np.unique(L))
        a, b = comps(lab), comps(pl)
        ensure(a < b, f"süper piksel tabanlı {a} bölge, piksel tabanlı {b}")
        R.table("coffee, k = 4", ["yöntem", "bağlı bölge sayısı", "kümelenen öğe"], [["piksel k-means", b, img.shape[0] * img.shape[1]], ["süper piksel + k-means", a, int(sp.max() + 1)]],
                note="Süper pikseller hem kümelenecek öğe sayısını ~300 kat azaltır hem de tuz-biber benzeri tek piksellik bölgeleri ortadan kaldırır.")
    R.test("süper piksel + k-means", sp)
