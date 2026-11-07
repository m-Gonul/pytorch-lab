import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = torch.tensor(data.camera(), dtype=torch.float64)[::4, ::4] / 255
    H, W = img.shape

    def vs_scipy():
        th = math.radians(17)
        A = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64) * 0.9
        t = torch.tensor([12.0, -7.5], dtype=torch.float64)
        got = need(ns, "warp_affine")(img, A, t).numpy()
        M = np.array([[A[1, 1], A[1, 0]], [A[0, 1], A[0, 0]]])
        ref = ndimage.affine_transform(img.numpy(), M, offset=[t[1].item(), t[0].item()], order=1, mode="constant", cval=0.0)
        inner = np.zeros_like(ref, bool); inner[8:-8, 8:-8] = True
        S = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1) @ A.numpy().T + t.numpy()
        ok = inner & (S[..., 0] > 1) & (S[..., 0] < W - 2) & (S[..., 1] > 1) & (S[..., 1] < H - 2)
        close(got[ok], ref[ok], atol=1e-9, what="iç pikseller")
    R.test("warp_affine = scipy.ndimage.affine_transform (iç bölge)", vs_scipy)

    def ident():
        close(need(ns, "warp_affine")(img, torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)), img, atol=1e-9)
        sh = ns["warp_affine"](img, torch.eye(2, dtype=torch.float64), torch.tensor([5.0, 0.0], dtype=torch.float64))
        close(sh[:, :-5], img[:, 5:], atol=1e-9, what="5 piksel sola kaydırma")
    R.test("birim dönüşüm ve tam sayı kaydırma", ident)

    def grad():
        A = torch.eye(2, dtype=torch.float64).requires_grad_(True)
        t = torch.tensor([0.3, 0.2], dtype=torch.float64, requires_grad=True)
        need(ns, "warp_affine")(img, A, t).sum().backward()
        ensure(A.grad is not None and t.grad is not None and t.grad.abs().sum() > 0, "A ve t gradyan almalı")
    R.test("türevlenebilir", grad)

    def reg():
        fixed = img.float()
        th, tx, ty = math.radians(8), 4.0, -3.0
        c, s = math.cos(th), math.sin(th)
        A = torch.tensor([[c, -s], [s, c]])
        ctr = torch.tensor([(W - 1) / 2, (H - 1) / 2])
        tt = ctr - A @ ctr + torch.tensor([tx, ty])
        Ai = torch.linalg.inv(A)
        moving = need(ns, "warp_affine")(fixed, Ai, -Ai @ tt)
        f = ndimage.gaussian_filter(fixed.numpy(), 2); m = ndimage.gaussian_filter(moving.numpy(), 2)
        est = need(ns, "register")(torch.tensor(f), torch.tensor(m))
        ensure(abs(math.degrees(est[0]) - 8) < 0.5 and abs(est[1] - tx) < 0.5 and abs(est[2] - ty) < 0.5,
               f"bulunan θ = {math.degrees(est[0]):.2f}°, t = ({est[1]:.2f}, {est[2]:.2f}); gerçek 8°, (4, −3)")
        return f"θ = {math.degrees(est[0]):.2f}° (8°), t = ({est[1]:.2f}, {est[2]:.2f}) ((4, −3))"
    R.test("register: döndürme + kaydırmayı gradient descent ile bul", reg)
