import numpy as np
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    def ws():
        img = data.camera()[:100, :120]
        m, s = need(ns, "window_stats")(img, 24)
        ensure(m.shape == (77, 97), f"şekil {m.shape}")
        win = np.lib.stride_tricks.sliding_window_view(img.astype(float), (24, 24))
        close(m, win.mean((-2, -1)), atol=1e-9, what="ortalama")
        close(s, win.std((-2, -1)), atol=1e-6, what="std")
    R.test("window_stats", ws)

    def dt():
        rows = []
        for H, W in [(16, 16), (640, 480), (1920, 1080), (8000, 8000)]:
            a, b = need(ns, "min_dtype")(H, W)
            rows.append([f"{W} × {H}", np.dtype(a).name, np.dtype(b).name])
        ensure(rows[0][1] == "uint16" and rows[1][1] == "uint32" and rows[1][2] == "uint64" and rows[3][1] == "uint64", f"{rows}")
        R.table("Integral görüntü için gereken tip (8 bit giriş)", ["boyut", "ii", "kare ii"], rows,
                note="uint8 görüntünün kümülatif toplamını uint8'de tutmak (alıştırma 5) neredeyse her pikselde taşar.")
    R.test("min_dtype", dt)
