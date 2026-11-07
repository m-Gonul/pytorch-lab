import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure, bench, fmt_time, uses_any


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("cv2.resize kullanılmıyor", lambda: ensure(not uses_any(src, ["resize"]) or "def resize" in src and "cv2" not in src, "cv2 kullanma"))
    img = data.camera().astype(np.float32)
    for H2, W2 in ((200, 300), (700, 900), (512, 512)):
        R.test(f"bilineer {H2}×{W2} = cv2", (lambda H2=H2, W2=W2: close(need(ns, "resize_bilinear")(img, H2, W2), cv2.resize(img, (W2, H2), interpolation=cv2.INTER_LINEAR), atol=1e-2)))
    for H2, W2 in ((200, 300), (700, 900)):
        R.test(f"bikübik {H2}×{W2} = cv2", (lambda H2=H2, W2=W2: close(need(ns, "resize_cubic")(img, H2, W2), cv2.resize(img, (W2, H2), interpolation=cv2.INTER_CUBIC), atol=1e-2)))
    if "resize_cubic" in ns:
        small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        rows = []
        for name, fn in (("en yakın", lambda: cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)),
                         ("bilineer", lambda: ns["resize_bilinear"](small, 512, 512)), ("bikübik", lambda: ns["resize_cubic"](small, 512, 512))):
            up = fn()
            psnr = 10 * np.log10(255 ** 2 / np.mean((up - img) ** 2))
            rows.append([name, f"{psnr:.2f} dB"])
        R.table("128 → 512 büyütme, orijinale göre PSNR", ["yöntem", "PSNR"], rows,
                note="Bikübik kenarları daha keskin korur; en yakın komşu blokludur.")
