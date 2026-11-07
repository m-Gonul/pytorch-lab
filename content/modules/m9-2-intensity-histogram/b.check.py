import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "skimage", "scipy"}))
    R.test("histogram", lambda: ensure(list(need(ns, "histogram")([[0, 0, 1], [255, 1, 1]]))[:2] == [2, 3] and need(ns, "histogram")([[0, 0, 1], [255, 1, 1]])[255] == 1, "sayımlar"))
    crops = {"camera": data.camera()[100:228, 100:228], "moon": data.moon()[200:328, 200:328]}

    def ocv():
        for name, im in crops.items():
            got = np.array(need(ns, "equalize_opencv")(im.tolist()), np.uint8)
            ensure(np.array_equal(got, cv2.equalizeHist(im)), f"{name}: cv2.equalizeHist ile farklı ({int(np.abs(got.astype(int) - cv2.equalizeHist(im)).sum())})")
        ensure(need(ns, "equalize_opencv")([[7, 7], [7, 7]]) == [[7, 7], [7, 7]], "tek renkli görüntü değişmemeli")
    R.test("equalize_opencv = cv2.equalizeHist (birebir)", ocv)

    def cls():
        rows = []
        for name, im in crops.items():
            got = np.array(need(ns, "equalize_class")(im.tolist()), np.int64)
            c = np.bincount(im.ravel(), minlength=256).cumsum()
            ensure(np.array_equal(got, np.round(255 * c / im.size).astype(np.int64)[im]), f"{name}: ders formülü")
            rows.append([name, str(int(np.abs(got - cv2.equalizeHist(im)).sum()))])
        R.table("Toplam mutlak fark: ders formülü − OpenCV", ["görüntü", "Σ|fark|"], rows,
                note="moon'da en koyu seviyede çok piksel var: c_min büyük, iki formül ayrışıyor.")
    R.test("equalize_class (ders formülü)", cls)
