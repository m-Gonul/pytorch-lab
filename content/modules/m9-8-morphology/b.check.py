import numpy as np
import cv2
from skimage import data
from labcheck import no_imports, need, ensure


def blobs():
    img = (data.coins()[60:140, 20:120] > 120).astype(np.uint8)
    return img


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/cv2 kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "skimage"}))
    A = blobs()
    sym = np.ones((3, 3), np.uint8)
    asym = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], np.uint8)

    def ero():
        for B in (sym, asym, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)):
            got = np.array(need(ns, "erode")(A.tolist(), B.tolist()), np.uint8)
            want = cv2.erode(A, B, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            ensure(np.array_equal(got, want), f"SE {B.tolist()}: cv2.erode ile farklı")
    R.test("erode = cv2.erode (dışarısı 0)", ero)

    def dil():
        got = np.array(need(ns, "dilate")(A.tolist(), asym.tolist()), np.uint8)
        flipped = cv2.dilate(A, asym[::-1, ::-1].copy(), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        plain = cv2.dilate(A, asym, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        ensure(np.array_equal(got, flipped), "ders kitabı genişlemesi = cv2.dilate(A, B[::-1, ::-1]) olmalı")
        ensure(np.array_equal(np.array(ns["dilate"](A.tolist(), sym.tolist()), np.uint8), cv2.dilate(A, sym)), "simetrik SE")
        return f"asimetrik SE: cv2.dilate(A, B) ile {int((got != plain).sum())} piksel farklı, B 180° döndürülünce 0"
    R.test("dilate (ders kitabı, yansıtılmış B)", dil)

    def dual():
        Ac = 1 - A
        lhs = 1 - np.array(ns["erode"](A.tolist(), asym.tolist()))
        rhs = np.array(ns["dilate"](Ac.tolist(), asym[::-1, ::-1].tolist()))
        inner = (slice(2, -2), slice(2, -2))
        ensure(np.array_equal(lhs[inner], rhs[inner]), "(A ⊖ B)ᶜ = Aᶜ ⊕ B̂ (kenardan uzakta)")
    R.test("ikilik", dual)

    def hmt():
        I = np.zeros((12, 12), np.uint8); I[2:5, 2:5] = 1; I[6, 2] = 1; I[8:11, 6:10] = 1; I[3, 9] = 1
        B = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        got = np.array(need(ns, "hit_or_miss")(I.tolist(), B))
        ensure(set(zip(*np.nonzero(got))) == {(6, 2), (3, 9)}, "tek piksellik nesneler bulunmalı")
        C = [[-1, 0, -1], [1, 1, 0], [-1, 1, -1]]
        got2 = np.array(ns["hit_or_miss"](I.tolist(), C))
        k = np.array(C); k = np.where(k == -1, 0, np.where(k == 0, -1, 1)).astype(np.int8)
        want = cv2.morphologyEx(I, cv2.MORPH_HITMISS, k)
        ensure(np.array_equal(got2[1:-1, 1:-1], want[1:-1, 1:-1]), "cv2 MORPH_HITMISS ile farklı (iç bölge)")
        return f"sağ üst köşe dedektörü: {list(zip(*np.nonzero(got2)))}"
    R.test("hit_or_miss (umursamaz elemanlarla)", hmt)

    def bnd():
        got = np.array(need(ns, "boundary")(A.tolist()), np.uint8)
        ensure(np.array_equal(got, A - cv2.erode(A, sym, borderType=cv2.BORDER_CONSTANT, borderValue=0)), "sınır")
    R.test("boundary", bnd)
