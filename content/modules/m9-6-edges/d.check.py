import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    def dk():
        k = need(ns, "dog_kernel")(1.5, 9)
        x = np.arange(9) - 4
        close(np.asarray(k, dtype=np.float64), x / 2.25 * np.exp(-x ** 2 / 4.5), atol=1e-6)
    R.test("dog_kernel", dk)

    def learn():
        w = need(ns, "learn_edge_filter")().double()
        ensure(w.shape == (21,), "21 katsayı")
        best = max((float(np.corrcoef(w.numpy(), ns["dog_kernel"](s, 21).double().numpy())[0, 1]), s) for s in np.arange(0.5, 6.01, 0.1))
        anti = float(np.corrcoef(w.numpy(), -w.flip(0).numpy())[0, 1])
        ensure(best[0] > 0.95, f"Gauss türeviyle en iyi korelasyon {best[0]:.3f} (σ = {best[1]:.1f}); 0.95 üstü bekleniyordu")
        ensure(anti > 0.9, f"tek (antisimetrik) değil: {anti:.3f}")
        R.table("Öğrenilen filtre", ["ölçü", "değer"], [["Gauss türeviyle korelasyon", f"{best[0]:.3f}"], ["en iyi σ", f"{best[1]:.1f}"], ["antisimetri", f"{anti:.3f}"]],
                note="Kimse 'Gauss türevi' demedi: gürültülü basamakları bulmayı öğrenen doğrusal filtre kendiliğinden ona yakınsıyor (Canny 1986).")
    R.test("öğrenilen filtre ≈ Gauss türevi", learn)

    def sob():
        img = data.camera()[100:180, 50:150].astype(np.float64)
        gx, gy = need(ns, "sobel_torch")(torch.tensor(img))
        close(gx.numpy(), cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE), atol=1e-9, what="gx")
        close(gy.numpy(), cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE), atol=1e-9, what="gy")
        ensure(ns["__source__"].count("conv2d") == 1, "tek conv2d çağrısı")
    R.test("sobel_torch = cv2.Sobel", sob)
