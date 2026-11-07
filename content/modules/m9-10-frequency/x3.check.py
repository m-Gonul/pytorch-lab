import numpy as np
from scipy import fft
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)

    def mat():
        C = need(ns, "dct_matrix")(8)
        close(C @ C.T, np.eye(8), atol=1e-12, what="ortonormal")
        X = rng.random((8, 12))
        close(need(ns, "dct2")(X), fft.dctn(X, norm="ortho"), atol=1e-12, what="dctn")
        close(need(ns, "idct2")(ns["dct2"](X)), X, atol=1e-12, what="geri")
    R.test("DCT-II = scipy.fft.dctn (ortho)", mat)

    def comp():
        img = data.camera()[::2, ::2]
        P = lambda a: 10 * np.log10(255 ** 2 / np.mean((a - img) ** 2))
        rows = []
        for frac in (0.01, 0.05, 0.2):
            d = P(need(ns, "keep_largest")(img, frac, "dct")); f = P(ns["keep_largest"](img, frac, "dft"))
            rows.append([f"%{frac * 100:g}", f"{d:.2f}", f"{f:.2f}"])
            ensure(d > f, f"%{frac * 100:g} katsayıda DCT {d:.2f} dB, DFT {f:.2f} dB")
        R.table("En büyük katsayıları tutma (camera 256²)", ["katsayı oranı", "DCT PSNR", "DFT PSNR"], rows,
                note="DFT bir karmaşık katsayı = iki gerçel sayı olduğu için adil karşılaştırmada fark daha da büyür.")
    R.test("enerji yoğunlaştırma: DCT > DFT", comp)
