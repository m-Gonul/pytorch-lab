from skimage.feature import haar_like_feature_coord
from labcheck import need, ensure


def check(ns, R):
    def t():
        rows = []
        for kind, (nx, ny) in {"type-2-x": (2, 1), "type-2-y": (1, 2), "type-3-x": (3, 1), "type-3-y": (1, 3), "type-4": (2, 2)}.items():
            for W, H in [(24, 24), (19, 19), (25, 20)]:
                want = len(haar_like_feature_coord(W, H, kind)[0])
                got = need(ns, "count_type")(W, H, nx, ny)
                ensure(got == want, f"{kind} {W}×{H}: {got}, skimage {want}")
            rows.append([kind, need(ns, "count_type")(24, 24, nx, ny)])
        ensure(need(ns, "total_features")(24, 24) == 162336, "24 × 24 toplam 162 336")
        R.table("24 × 24 pencere", ["tür", "öznitelik"], rows + [["toplam", 162336]])
    R.test("öznitelik sayıları = skimage", t)
