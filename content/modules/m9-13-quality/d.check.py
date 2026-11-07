import numpy as np
import torch
from skimage import data
from skimage.metrics import structural_similarity as sk_ssim
from labcheck import close, need, ensure


def check(ns, R):
    ref = torch.tensor(data.camera()[128:384, 128:384], dtype=torch.float64)

    def eq():
        y = ref + torch.randn(ref.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 20
        want = sk_ssim(ref.numpy(), y.numpy(), data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        close(float(need(ns, "ssim_t")(ref, y)), want, atol=1e-10, what="skimage SSIM")
        yy = y.clone().requires_grad_(True)
        ns["ssim_t"](ref, yy).backward()
        ensure(yy.grad is not None and torch.isfinite(yy.grad).all(), "türev")
    R.test("ssim_t = skimage ve türevlenebilir", eq)

    def sph():
        hi = need(ns, "mse_sphere")(ref, 210.0, +1)
        lo = ns["mse_sphere"](ref, 210.0, -1)
        m = lambda a: float(((a - ref) ** 2).mean())
        ensure(abs(m(hi) - 210) < 0.5 and abs(m(lo) - 210) < 0.5, "MSE küresinde kalmalı")
        sh, sl = float(ns["ssim_t"](ref, hi)), float(ns["ssim_t"](ref, lo))
        ensure(sh > 0.9 and sl < 0.3, f"en iyi {sh:.3f}, en kötü {sl:.3f}")
        R.table("MSE = 210 küresinde", ["görüntü", "MSE", "SSIM"], [["SSIM'i en büyükle", f"{m(hi):.1f}", f"{sh:.3f}"], ["SSIM'i en küçükle", f"{m(lo):.1f}", f"{sl:.3f}"]],
                note="Aynı MSE: en iyisinde hata düz bölgelere değil yapının içine 'gizlenir', en kötüsünde yapıyı bozan desenler oluşur.")
    R.test("MSE küresinde en büyük / en küçük SSIM", sph)
