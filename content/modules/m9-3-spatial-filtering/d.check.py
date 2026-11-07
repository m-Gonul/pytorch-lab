import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera()[:200, :240].astype(np.float32)
    rng = np.random.default_rng(1)
    K = rng.normal(size=(5, 3)).astype(np.float32)

    def borders():
        for name, bt in (("constant", cv2.BORDER_CONSTANT), ("replicate", cv2.BORDER_REPLICATE), ("reflect101", cv2.BORDER_REFLECT_101)):
            got = need(ns, "filter2d")(torch.tensor(img), torch.tensor(K), name).numpy()
            close(got, cv2.filter2D(img, -1, K, borderType=bt), atol=2e-3, what=name)
    R.test("filter2d = cv2.filter2D (üç kenar modu)", borders)

    def rec():
        torch.manual_seed(0)
        x = torch.rand(128, 128)
        w_true = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32) * 0.5
        w_true[0, 0] = 0.3
        y = need(ns, "filter2d")(x, w_true)
        w = need(ns, "recover_kernel")(x, y, 3)
        close(w, w_true, atol=2e-2)
        return "bulunan çekirdek:\n" + np.array2string(w.numpy(), precision=3)
    R.test("recover_kernel: beyaz gürültü girdisiyle çekirdeği bul", rec)
