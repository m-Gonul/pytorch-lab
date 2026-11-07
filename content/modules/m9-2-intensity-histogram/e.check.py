import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: max_loops(src, 0, func="equalize_batch"))

    def eq():
        ims = [data.camera(), data.moon(), data.coins()[:303, :384][:256, :256], data.page()[:172, :256]]
        ims = [cv2.resize(i, (256, 256)) for i in ims] + [np.full((256, 256), 9, np.uint8)]
        X = torch.tensor(np.stack(ims))
        got = need(ns, "equalize_batch")(X).numpy()
        for i, im in enumerate(ims):
            ensure(np.array_equal(got[i], cv2.equalizeHist(im)), f"görüntü {i} cv2 ile farklı")
    R.test("her görüntü = cv2.equalizeHist (tek renkli dahil)", eq)
    if "equalize_batch" not in ns:
        return
    g = torch.Generator().manual_seed(0)
    X = (torch.rand(512, 128, 128, generator=g) ** 2 * 255).to(torch.uint8)
    Xn = X.numpy()
    t_loop = bench(lambda: [cv2.equalizeHist(Xn[i]) for i in range(512)], repeat=3, number=1)
    t_vec = bench(lambda: ns["equalize_batch"](X), repeat=3)
    R.table("512 görüntü, 128 × 128", ["yöntem", "süre"], [["cv2.equalizeHist döngüsü", fmt_time(t_loop)], ["tek bincount (torch)", fmt_time(t_vec)]],
            note="OpenCV'nin tek görüntülük C++ çekirdeği çok hızlı; batch'li sürüm GPU'da ve veri artırma hattında anlam kazanır.")
