import numpy as np
import cv2
from scipy import ndimage
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera()
    f = img.astype(np.float64)
    r = need(ns, "pipeline")(img)

    def steps():
        L = ndimage.correlate(f, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float), mode="mirror")
        close(r["L"], L, atol=1e-6, what="L")
        close(r["S1"], f - L, atol=1e-6, what="S1")
        gx = ndimage.correlate(f, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float), mode="mirror")
        gy = ndimage.correlate(f, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float), mode="mirror")
        G = np.abs(gx) + np.abs(gy)
        close(r["G"], G, atol=1e-6, what="G")
        Gs = ndimage.uniform_filter(G, 5, mode="mirror")
        close(r["Gs"], Gs, atol=1e-6, what="Gs")
        close(r["P"], -L * Gs / Gs.max(), atol=1e-6, what="P")
        close(r["S2"], f - L * Gs / Gs.max(), atol=1e-6, what="S2")
        ensure(r["out"].dtype == np.uint8, "out uint8")
    R.test("tüm ara adımlar", steps)

    def noise():
        flat = r["S1"][ndimage.uniform_filter(f, 15) > 0] - f[ndimage.uniform_filter(f, 15) > 0]
        n1 = np.abs(r["S1"] - f)[ndimage.uniform_filter(r["G"], 9) < np.percentile(r["G"], 30)].mean()
        n2 = np.abs(r["S2"] - f)[ndimage.uniform_filter(r["G"], 9) < np.percentile(r["G"], 30)].mean()
        ensure(n2 < 0.5 * n1, f"düz bölgelerde ayrıntı eklemesi S1'de {n1:.2f}, S2'de {n2:.2f}: maske düz bölgeleri korumalı")
        return f"düz bölgelerde ortalama değişim: Laplasyen keskinleştirme {n1:.2f}, maskeli {n2:.2f} gri seviye"
    R.test("maske gürültüyü düz bölgelerde bastırıyor", noise)
