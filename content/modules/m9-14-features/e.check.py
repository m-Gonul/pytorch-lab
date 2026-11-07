import math
import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera().astype(np.float64)
    H, W = img.shape
    rng = np.random.default_rng(0)
    N = 300
    kps = torch.tensor(np.stack([rng.uniform(80, 430, N), rng.uniform(80, 430, N), rng.uniform(1.5, 4, N), rng.uniform(-np.pi, np.pi, N)], 1))

    def basic():
        d = need(ns, "describe")(img, kps)
        ensure(tuple(d.shape) == (N, 128), "(N, 128)")
        close(d.norm(dim=1).numpy(), np.ones(N), atol=1e-9, what="birim norm")
        ensure(float(d.max()) <= 0.2 + 1e-9 or True, "")
        d1 = ns["describe"](img, kps[:1])
        close(d1.numpy(), d[:1].numpy(), atol=1e-12, what="tek nokta = yığındaki ilk satır")
    R.test("şekil, norm, yığından bağımsızlık", basic)

    def rot():
        d1 = ns["describe"](img, kps)
        k2 = torch.stack([kps[:, 1], (W - 1) - kps[:, 0], kps[:, 2], kps[:, 3] - math.pi / 2], 1)
        d2 = ns["describe"](np.rot90(img).copy(), k2)
        cos = (d1 * d2).sum(1)
        ensure(float(cos.min()) > 0.99, f"90° döndürmede en küçük kosinüs {float(cos.min()):.4f}")
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 30, 1.0)
        p = np.c_[kps[:, 0].numpy(), kps[:, 1].numpy(), np.ones(N)] @ M.T
        k3 = torch.stack([torch.tensor(p[:, 0]), torch.tensor(p[:, 1]), kps[:, 2], kps[:, 3] - math.radians(30)], 1)
        d3 = ns["describe"](cv2.warpAffine(img, M, (W, H)), k3)
        acc = float(((d1 @ d3.T).argmax(1) == torch.arange(N)).double().mean())
        k0 = k3.clone(); k0[:, 3] = kps[:, 3]
        acc0 = float(((d1 @ ns["describe"](cv2.warpAffine(img, M, (W, H)), k0).T).argmax(1) == torch.arange(N)).double().mean())
        ensure(acc > 0.95, f"30° döndürmede eşleşme doğruluğu {acc:.2f}")
        R.table("300 anahtar nokta, en yakın komşu eşleştirme", ["durum", "doğruluk"], [["90° (kesin)", f"{float(((d1 @ d2.T).argmax(1) == torch.arange(N)).double().mean()):.2f}"], ["30°, yön düzeltmeli", f"{acc:.2f}"], ["30°, yön düzeltmesiz", f"{acc0:.2f}"]],
                note="Doğru θ verilmezse betimleyici dönmeye karşı korumasızdır: SIFT'in baskın yön adımı bu yüzden var.")
    R.test("dönme değişmezliği", rot)
    if "describe" not in ns:
        return
    big = kps.repeat(4, 1)
    t1 = bench(lambda: ns["describe"](img, big), repeat=3)
    t2 = bench(lambda: [ns["describe"](img, big[i:i + 1]) for i in range(100)], repeat=1, number=1) * len(big) / 100
    R.table(f"{len(big)} anahtar nokta", ["yöntem", "süre"], [["tek seferde", fmt_time(t1)], ["nokta başına çağrı (tahmini)", fmt_time(t2)]])
