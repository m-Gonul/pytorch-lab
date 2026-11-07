import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = torch.tensor(data.astronaut()[100:220, 150:300], dtype=torch.float64).permute(2, 0, 1)[None] / 255

    def eig():
        Fv, th = need(ns, "di_zenzo")(img)
        np_img = img[0].permute(1, 2, 0).numpy()
        gx = np.stack([cv2.Sobel(np_img[..., c], cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE) for c in range(3)], -1)
        gy = np.stack([cv2.Sobel(np_img[..., c], cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE) for c in range(3)], -1)
        gxx, gyy, gxy = (gx * gx).sum(-1), (gy * gy).sum(-1), (gx * gy).sum(-1)
        lmax = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2))
        close(Fv[0].numpy() ** 2, lmax, atol=1e-9, what="F² = en büyük özdeğer")
        close(ns["channel_sum_grad"](img)[0].numpy(), np.sqrt(gx ** 2 + gy ** 2).sum(-1), atol=1e-9, what="kanal toplamı")
    R.test("di_zenzo: F² = λ_max", eig)

    def toy():
        n = 9
        step_x = torch.zeros(n, n, dtype=torch.float64); step_x[:, n // 2:] = 1
        step_y = step_x.T.clone()
        a = torch.stack([step_x, step_x, step_x])[None]
        e = torch.stack([step_x, step_x, step_y])[None]
        c = n // 2
        sa, se = ns["channel_sum_grad"](a)[0, c, c].item(), ns["channel_sum_grad"](e)[0, c, c].item()
        fa, fe = need(ns, "di_zenzo")(a)[0][0, c, c].item(), ns["di_zenzo"](e)[0][0, c, c].item()
        ensure(abs(sa - se) < 1e-9, "kanal toplamları eşit olmalı")
        ensure(fa > fe * 1.15, f"vektör gradyanı (a) {fa:.2f}, (e) {fe:.2f}: (a) daha güçlü olmalı")
        R.table("Notlardaki örnek: merkez piksel", ["görüntü", "kanal toplamı", "Di Zenzo F"], [["(a) üç kanal aynı kenar", f"{sa:.2f}", f"{fa:.2f}"], ["(e) B kanalı dik kenar", f"{se:.2f}", f"{fe:.2f}"]],
                note="Kanal toplamı ikisini ayırt edemiyor; vektör gradyanı (a)'daki renk kenarının daha güçlü olduğunu görüyor.")
    R.test("kanal toplamı yanılır, vektör gradyanı yanılmaz", toy)

    def seg():
        im = torch.tensor(data.astronaut(), dtype=torch.float64).permute(2, 0, 1)[None] / 255
        samples = im[0, :, 20:60, 360:420].reshape(3, -1).T
        m = need(ns, "mahalanobis_mask")(im, samples, 3.0)
        a = samples.mean(0).numpy(); C = np.cov(samples.numpy().T)
        z = im[0].permute(1, 2, 0).numpy() - a
        d = np.sqrt(np.einsum("hwi,ij,hwj->hw", z, np.linalg.inv(C), z))
        ensure(np.array_equal(m[0].numpy(), d <= 3.0), "Mahalanobis maskesi")
        b = need(ns, "box_mask")(im, samples, 1.25)
        s = samples.std(0).numpy()
        ensure(np.array_equal(b[0].numpy(), (np.abs(z) <= 1.25 * s).all(-1)), "kutu maskesi")
        return f"Mahalanobis (D0 = 3): %{m.double().mean() * 100:.1f}, kutu (1.25σ): %{b.double().mean() * 100:.1f} piksel"
    R.test("renk bölütleme maskeleri", seg)
