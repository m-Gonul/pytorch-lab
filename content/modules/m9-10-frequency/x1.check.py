import numpy as np
from skimage import data, color
from labcheck import close, need, ensure


def check(ns, R):
    a = data.camera().astype(np.float64)
    b = (color.rgb2gray(data.astronaut()) * 255)

    def sw():
        out = need(ns, "swap_phase")(a, b)
        c = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        ca, cb = c(out, a), c(out, b)
        ensure(cb > ca + 0.2, f"faz b'den: b ile korelasyon {cb:.2f}, a ile {ca:.2f}")
        return f"|F_camera| + ∠F_astronaut → astronaut ile korelasyon {cb:.2f}, camera ile {ca:.2f}: yapıyı faz taşır"
    R.test("swap_phase", sw)

    def sp():
        s = need(ns, "spectrum_log")(a)
        ensure(s.dtype == np.uint8 and s.max() == 255 and s.min() == 0, "0–255 uint8")
        ensure(np.unravel_index(np.argmax(s), s.shape) == (256, 256), "DC merkezde olmalı")
    R.test("spectrum_log merkezli", sp)

    def sh():
        close(need(ns, "shift")(a, 7, -3), np.roll(a, (-3, 7), (0, 1)), atol=1e-8, what="tamsayı kayma = np.roll")
        o = a[:511, :511]
        h = ns["shift"](ns["shift"](o, 0.5, 0.25), 0.5, 0.75)
        close(h, np.roll(o, (1, 1), (0, 1)), atol=1e-6, what="iki kesirli kayma = (1, 1) (tek boyutlu görüntü: Nyquist bileşeni yok)")
    R.test("öteleme teoremi", sh)
