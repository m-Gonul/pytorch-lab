import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    clean_np = data.camera()[128:384, 128:384]
    noisy_np = np.clip(clean_np + rng.normal(0, 15, clean_np.shape), 0, 255).astype(np.uint8)

    def cv():
        for d, sc, ss in [(7, 30.0, 5.0), (9, 50.0, 3.0)]:
            got = need(ns, "bilateral_t")(torch.tensor(noisy_np, dtype=torch.float64), d // 2, sc, ss).numpy()
            want = cv2.bilateralFilter(noisy_np, d, sc, ss)
            diff = np.abs(np.clip(np.rint(got), 0, 255) - want)
            ensure(diff.max() <= 1 and (diff > 0).mean() < 0.005, f"d = {d}: en büyük fark {diff.max()}, %{(diff > 0).mean() * 100:.2f} piksel")
    R.test("bilateral_t ≈ cv2.bilateralFilter", cv)

    def grad():
        x = torch.tensor(noisy_np[:32, :32], dtype=torch.float64)
        p = torch.tensor([30.0, 2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(lambda a: ns["bilateral_t"](x, 2, a[0], a[1]).sum(), (p,))
    R.test("σ'lara göre türev (gradcheck)", grad)

    def fit():
        clean = torch.tensor(clean_np, dtype=torch.float32)
        noisy = torch.tensor(noisy_np, dtype=torch.float32)
        P = lambda a: 10 * torch.log10(255 ** 2 / ((a - clean) ** 2).mean()).item()
        grid = max((P(ns["bilateral_t"](noisy, 3, sc, ss)), sc, ss) for sc in (10, 20, 30, 40, 60) for ss in (0.5, 1, 1.5, 2, 3))
        sc, ss = need(ns, "fit_bilateral")(noisy, clean)
        pl = P(ns["bilateral_t"](noisy, 3, sc, ss))
        ensure(pl >= grid[0] - 0.1, f"öğrenilen {pl:.2f} dB, ızgara {grid[0]:.2f} dB")
        R.table("σ = 15 Gauss gürültüsü (camera 256²), r = 3", ["yöntem", "σ_r", "σ_s", "PSNR (dB)"],
                [["gürültülü", "—", "—", f"{P(noisy):.2f}"], ["ızgara arama (25 nokta)", grid[1], grid[2], f"{grid[0]:.2f}"], ["gradyanla (150 adım)", f"{sc:.1f}", f"{ss:.2f}", f"{pl:.2f}"]],
                note="σ_r ≈ 2.5 × gürültü σ'sı: kenar sayılacak genlik gürültünün birkaç katı olmalı.")
    R.test("σ'ları gradyanla bul", fit)
