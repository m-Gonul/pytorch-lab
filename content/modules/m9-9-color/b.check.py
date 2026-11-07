import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "skimage"}))
    rng = np.random.default_rng(0)
    cols = rng.random((300, 3)).tolist() + [[0.3, 0.3, 0.3], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

    def fwd():
        for r, g, b in cols:
            h, s, i = need(ns, "rgb_to_hsi")(r, g, b)
            close(i, (r + g + b) / 3, what="I")
            if r + g + b > 0:
                close(s, 1 - 3 * min(r, g, b) / (r + g + b), atol=1e-12, what="S")
        close(ns["rgb_to_hsi"](1, 0, 0)[0], 0.0, what="kırmızı H")
        close(ns["rgb_to_hsi"](0, 1, 0)[0], 120.0, atol=1e-9, what="yeşil H")
        close(ns["rgb_to_hsi"](0, 0, 1)[0], 240.0, atol=1e-9, what="mavi H")
        close(ns["rgb_to_hsi"](1, 1, 0)[0], 60.0, atol=1e-9, what="sarı H")
        ensure(ns["rgb_to_hsi"](0.3, 0.3, 0.3)[0] == 0 and ns["rgb_to_hsi"](0, 0, 0)[1] == 0, "gri ve siyah özel durumları")
    R.test("rgb_to_hsi", fwd)

    def rt():
        err = 0.0
        for r, g, b in cols[:300]:
            back = need(ns, "hsi_to_rgb")(*ns["rgb_to_hsi"](r, g, b))
            err = max(err, max(abs(x - y) for x, y in zip(back, (r, g, b))))
        ensure(err < 1e-9, f"gidiş-dönüş hatası {err:.2e}")
        return f"300 rastgele renkte en büyük hata {err:.1e}"
    R.test("hsi_to_rgb gidiş-dönüş", rt)

    def cm():
        close(need(ns, "rgb_to_cmyk")(0.2, 0.5, 0.9), (0.8 / 0.9 - 0.1 / 0.9, 0.5 / 0.9 - 0.1 / 0.9, 0.0, 0.1), atol=1e-12, what="(0.2, 0.5, 0.9)")
        ensure(ns["rgb_to_cmyk"](0, 0, 0) == (0.0, 0.0, 0.0, 1.0), "siyah: K = 1, diğerleri 0")
        err = max(max(abs(x - y) for x, y in zip(need(ns, "cmyk_to_rgb")(*ns["rgb_to_cmyk"](r, g, b)), (r, g, b))) for r, g, b in cols)
        ensure(err < 1e-12, f"gidiş-dönüş {err:.1e}")
    R.test("CMYK gidiş-dönüş", cm)
