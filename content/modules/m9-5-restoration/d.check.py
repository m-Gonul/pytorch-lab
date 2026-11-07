import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    def shape():
        m = need(ns, "Denoiser", "sınıf")()
        x = torch.rand(2, 1, 20, 24)
        ensure(tuple(m(x).shape) == (2, 1, 20, 24), "aynı boyutta çıkış")
        convs = [l for l in m.modules() if isinstance(l, torch.nn.Conv2d)]
        ensure(len(convs) == 4 and convs[-1].out_channels == 1, "4 konvolüsyon, son katman 1 kanal")
    R.test("Denoiser yapısı", shape)

    def train():
        tr = torch.tensor(data.camera(), dtype=torch.float32) / 255
        te = data.coins()[:256, :256].astype(np.float32) / 255
        sig = 25 / 255
        m = need(ns, "train_denoiser")(tr, sig)
        noisy = te + sig * np.random.default_rng(5).normal(size=te.shape).astype(np.float32)
        with torch.no_grad():
            out = m(torch.tensor(noisy)[None, None])[0, 0].clamp(0, 1).numpy()
        P = lambda a: 10 * np.log10(1 / np.mean((a - te) ** 2))
        p_n, p_g, p_m, p_c = P(noisy), P(cv2.GaussianBlur(noisy, (5, 5), 0)), P(cv2.medianBlur((np.clip(noisy, 0, 1) * 255).astype(np.uint8), 5).astype(np.float32) / 255), P(out)
        ensure(p_c > p_g + 1.0, f"CNN {p_c:.2f} dB, Gauss {p_g:.2f} dB: en az 1 dB fark bekleniyordu")
        R.table("coins, σ = 25 (görülmemiş görüntü)", ["yöntem", "PSNR (dB)"],
                [["gürültülü", f"{p_n:.2f}"], ["Gauss 5×5", f"{p_g:.2f}"], ["median 5×5", f"{p_m:.2f}"], ["senin CNN (500 adım)", f"{p_c:.2f}"]])
    R.test("öğrenilen gürültü giderici klasik filtreleri geçiyor", train)
