import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera().astype(np.float32) / 255

    def flat():
        asym = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], np.uint8)
        w = torch.where(torch.tensor(asym) > 0, 0.0, -1e4)
        got = need(ns, "dilate_t")(torch.tensor(img)[None, None], w)[0, 0].numpy()
        close(got, cv2.dilate(img, asym[::-1, ::-1].copy()), atol=1e-6, what="düz asimetrik SE")
        bN = torch.tensor([[0.0, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.0]])
        from scipy import ndimage
        want = ndimage.grey_dilation(img.astype(np.float64), structure=bN.numpy().astype(np.float64), mode="constant", cval=-1e4)
        close(ns["dilate_t"](torch.tensor(img)[None, None], bN)[0, 0].numpy(), want, atol=1e-5, what="düz olmayan SE")
    R.test("dilate_t", flat)

    def learn():
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        w = need(ns, "learn_se")(img, cross)
        on, off = w[torch.tensor(cross) > 0], w[torch.tensor(cross) == 0]
        err = float((ns["dilate_t"](torch.tensor(img)[None, None], w)[0, 0] - torch.tensor(cv2.dilate(img, cross))).abs().max())
        ensure(on.abs().max() < 0.05, f"artının elemanları 0 olmalı: {on.tolist()}")
        ensure(off.max() < -0.3, f"köşeler belirgin negatif olmalı: {off.tolist()}")
        ensure(err < 0.05, f"en büyük hata {err:.3f}")
        R.table("Öğrenilen yapı elemanı (hedef: 3 × 3 artı)", ["", "sol", "orta", "sağ"], [[n] + [f"{v:+.3f}" for v in row] for n, row in zip(["üst", "orta", "alt"], w.tolist())],
                note=f"En büyük hata {err:.4f}. Köşeler −∞'a gitmez: doğal görüntüde komşu farkları nadiren 0.5'i aşar, köşe 'yeterince' negatif olunca hiç kazanmaz ve gradyan almaz.")
    R.test("yapı elemanını öğren", learn)

    def th():
        u8 = data.coins()
        ramp = np.linspace(0, 60, u8.shape[1], dtype=np.float32)[None, :]
        f = np.clip(u8 + ramp, 0, 255).astype(np.float32)
        got = need(ns, "tophat_t")(torch.tensor(f)[None, None], 31)[0, 0].numpy()
        close(got, cv2.morphologyEx(f, cv2.MORPH_TOPHAT, np.ones((31, 31), np.uint8)), atol=1e-4, what="top-hat")
        l, r = f[:, :60].mean() - f[:, -60:].mean(), got[:, :60].mean() - got[:, -60:].mean()
        return f"sol–sağ ortalama farkı: girdi {l:.1f}, top-hat {r:.1f} (aydınlatma eğimi gitti)"
    R.test("tophat_t = cv2 MORPH_TOPHAT", th)
