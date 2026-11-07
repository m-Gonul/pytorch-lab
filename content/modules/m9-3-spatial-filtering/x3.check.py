import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    crop = data.camera()[100:160, 300:380].astype(np.float32)

    def eq():
        for s in (3, 15, 31):
            got = np.array(need(ns, "box_running")(crop.tolist(), s), np.float32)
            close(got, cv2.boxFilter(crop, -1, (s, s), normalize=True, borderType=cv2.BORDER_CONSTANT), atol=1e-2, what=f"{s}×{s}")
    R.test("box_running = cv2.boxFilter", eq)

    def cnt():
        res = {}
        for s in (3, 31):
            ns["ops"] = 0
            need(ns, "box_running")(crop.tolist(), s)
            res[s] = ns["ops"] / crop.size
        ensure(res[3] > 0, "ops sayacı artırılmalı (global ops)")
        ensure(res[31] < 1.6 * res[3], f"piksel başına işlem çekirdekle büyümemeli: {res}")
        return f"piksel başına toplama/çıkarma: 3×3 → {res[3]:.2f}, 31×31 → {res[31]:.2f}"
    R.test("işlem sayısı çekirdek boyutundan bağımsız", cnt)
