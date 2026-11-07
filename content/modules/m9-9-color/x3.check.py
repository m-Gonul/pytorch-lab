import numpy as np
from skimage import color, data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.astronaut()[::4, ::4] / 255.0

    def xyz():
        close(need(ns, "rgb_to_xyz")(img), color.rgb2xyz(img), atol=1e-9, what="XYZ")
    R.test("rgb_to_xyz = skimage.color.rgb2xyz", xyz)

    def lab():
        close(need(ns, "xyz_to_lab")(ns["rgb_to_xyz"](img)), color.rgb2lab(img), atol=1e-6, what="Lab")
    R.test("Lab = skimage.color.rgb2lab", lab)

    def chrom():
        prim = np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]]])
        x, y = need(ns, "chromaticity")(ns["rgb_to_xyz"](prim))
        rows = [[n, f"{x[0, i]:.3f}", f"{y[0, i]:.3f}"] for i, n in enumerate(["R", "G", "B", "beyaz (D65)"])]
        close(x[0, 3], 0.3127, atol=2e-3, what="D65 x"); close(y[0, 3], 0.3290, atol=2e-3, what="D65 y")
        R.table("sRGB ana renklerinin renklilik koordinatları", ["renk", "x", "y"], rows, note="Bu üç nokta sRGB gamut üçgeninin köşeleri; at nalının tamamını kapsamaz.")
    R.test("chromaticity", chrom)

    def de():
        a = color.rgb2lab(np.array([[[0.5, 0.2, 0.2]]])); b = color.rgb2lab(np.array([[[0.52, 0.2, 0.2]]]))
        c = color.rgb2lab(np.array([[[0.2, 0.5, 0.2]]])); d = color.rgb2lab(np.array([[[0.2, 0.52, 0.2]]]))
        close(need(ns, "delta_e76")(a, b), color.deltaE_cie76(a, b), what="ΔE")
        return f"R'de +0.02: ΔE = {float(ns['delta_e76'](a, b)[0, 0]):.2f}; G'de +0.02: ΔE = {float(ns['delta_e76'](c, d)[0, 0]):.2f} (RGB'de eşit adım, algıda değil)"
    R.test("delta_e76", de)
